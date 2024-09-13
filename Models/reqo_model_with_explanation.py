import torch
from torch.nn import Linear, ModuleList, MaxPool2d, Dropout, Sequential, Sigmoid, Softplus
from torch_geometric.nn import TransformerConv, BatchNorm, GRUAggregation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from .DirGNNConv import DirGNNConv

class Feature_encoder(torch.nn.Module):
    def __init__(self, encoder_params):
        super(Feature_encoder, self).__init__()
        self.node_type_num = 26
        self.column_num = encoder_params["encoder_column_num"]
        self.table_num = encoder_params["encoder_table_num"]

        # Node encoding
        self.node_type_embedding_dim = encoder_params["encoder_node_type_embedding_dim"]
        self.column_embedding_dim = encoder_params["encoder_column_embedding_dim"]
        self.node_type_enc = Linear(self.node_type_num, self.node_type_embedding_dim)
        self.column_layers = ModuleList([
            Linear(8, self.column_embedding_dim, bias=False) for _ in range(self.column_num)
        ])

    def forward(self, node_features, table_columns_number):
        node_num = node_features.size(0)
        node_features = torch.split(node_features, [self.node_type_num, 2, self.table_num, self.column_num * 8], dim=1)
        node_type_enc = torch.relu(self.node_type_enc(node_features[0]))
        node_stats_enc = node_features[1]
        node_table_used = node_features[2]

        # Reshape and transpose once for column encoding
        node_column_enc = node_features[3].view(node_num, self.column_num, 8).transpose(0, 1)
        node_column_new_enc = torch.stack([torch.relu(layer(node_column_enc[i])) for i, layer in enumerate(self.column_layers)])

        node_table_enc = []
        start_idx = 0
        for num in table_columns_number:
            pooled = MaxPool2d((num, 1))(node_column_new_enc[start_idx:start_idx + num].transpose(0, 1)).squeeze()
            node_table_enc.append(pooled)
            start_idx += num
        node_table_enc = torch.cat(node_table_enc, dim=1)

        # Concatenate all features
        encoded_node_features = torch.cat([node_type_enc,node_stats_enc,  node_table_used, node_table_enc], dim=1)
        return encoded_node_features

class BiGG(torch.nn.Module):
    def __init__(self, encoder_params, node_feature_dim):
        super(BiGG, self).__init__()
        # BIGG tree model
        self.node_feature_dim = node_feature_dim
        n_heads = encoder_params["encoder_attention_heads"]
        dropout_rate = encoder_params["encoder_gnn_dropout_rate"]
        embedding_dim = encoder_params["encoder_gnn_embedding_dim"]
        self.n_conv_layers = encoder_params["encoder_conv_layers"]
        dirgnn_alpha = encoder_params["encoder_dirgnn_alpha"]

        # Bidirectional GNN layers
        self.conv_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        current_dim = node_feature_dim
        for i in range(self.n_conv_layers):
            conv = TransformerConv(current_dim, embedding_dim, heads=n_heads, dropout=dropout_rate, concat=False)
            self.conv_layers.append(DirGNNConv(conv=conv, alpha=dirgnn_alpha, root_weight=True))
            self.bn_layers.append(BatchNorm(embedding_dim))
            current_dim = embedding_dim

        # GRU aggregation layer
        self.aggr_layer = GRUAggregation(embedding_dim, embedding_dim)

    def forward(self, node_features, edge_index, batch_subtree_index, subtree_index, subtree_labels):
        x = node_features

        num = 0
        for i in range(len(batch_subtree_index)):
            num += len(subtree_index[batch_subtree_index[i]])
        assert  num == len(x), f"Number of nodes mismatch! Expected {num}, but got {len(x)}"

        batch_index_all, global_tree_index, batch_global_labels, batch_local_labels = self.prepare_batches_for_subtrees(batch_subtree_index, subtree_index, subtree_labels)

        for i in range(self.n_conv_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            x = torch.relu(x)
        x = self.aggr_layer(x, batch_index_all)

        y = x[global_tree_index]
        M = torch.zeros(batch_local_labels.shape[0], x.shape[1]*2, device=device)

        sg_num = 0
        for i, g_idx in enumerate(global_tree_index):
            subtree_num = g_idx - sg_num + 1
            subtree_x = x[sg_num:g_idx + 1]
            if subtree_num > 1:
                global_x = x[g_idx].unsqueeze(0)
                M[sg_num:sg_num + subtree_num] = torch.cat((subtree_x, global_x.expand(subtree_num, -1)), dim=1)
            else:
                M[sg_num] = torch.cat((x[g_idx], x[g_idx]), dim=0)
            sg_num += subtree_num

        return y, M, batch_global_labels, batch_local_labels

    def prepare_batches_for_subtrees(self, batch_subtree_index, subtree_index, subtree_labels):
        batch_size = len(batch_subtree_index)
        batch_index_all, global_tree_index, batch_global_labels, batch_local_labels = [], [], [], []
        num = 0
        for i in range(batch_size):
            batch_subtree_labels_per_tree = subtree_labels[batch_subtree_index[i]].to(device)
            assert torch.max(batch_subtree_labels_per_tree) == batch_subtree_labels_per_tree[-1], "Last element is not max value!"
            batch_global_labels.append(torch.max(batch_subtree_labels_per_tree))
            batch_local_labels.append(batch_subtree_labels_per_tree / torch.max(batch_subtree_labels_per_tree))
            batch_index_per_tree = subtree_index[batch_subtree_index[i]].to(device)
            batch_index_all.extend(batch_index_per_tree + num)
            num += int(batch_index_per_tree.max().item() + 1)
            global_tree_index.append(num - 1)

        return torch.tensor(batch_index_all, device=device), global_tree_index, torch.tensor(batch_global_labels, device=device), torch.cat(batch_local_labels)

    def predict_without_explainer(self, node_features, edge_index, batch_index):
        x = node_features
        for i in range(self.n_conv_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            x = torch.relu(x)
        x = self.aggr_layer(x, batch_index)
        return x


class Explainer(torch.nn.Module):
    def __init__(self, explainer_params, embedding_dim):
        super(Explainer, self).__init__()
        # Explainer model to estimate the contribution of each subtree embedding to the global embedding
        self.explainer_layers = ModuleList([])
        self.n_explainer_layers = explainer_params["explainer_fcn_layers"]
        explanation_embedding_dim = explainer_params["explainer_explanation_embedding_dim"]
        self.fcn_dropout_rate = explainer_params["explainer_fcn_dropout_rate"]
        self.explainer_layers.append(Linear(embedding_dim, explanation_embedding_dim))
        for i in range(self.n_explainer_layers - 1):
            if i != self.n_explainer_layers - 2:
                self.explainer_layers.append(Linear(explanation_embedding_dim, int(explanation_embedding_dim / 2)))
                explanation_embedding_dim = int(explanation_embedding_dim / 2)
            else:
                self.explainer_layers.append(Linear(explanation_embedding_dim, 1))
        self.dropout = Dropout(p=self.fcn_dropout_rate)

    def forward(self, x):
        for i in range(self.n_explainer_layers - 1):
            x = self.dropout(torch.relu(self.explainer_layers[i](x)))
        x = torch.sigmoid(self.explainer_layers[-1](x))
        return x


class Estimator(torch.nn.Module):
    def __init__(self, estimator_params, embedding_dim):
        super(Estimator, self).__init__()
        # Estimator
        self.fcn_layers = ModuleList([])
        self.fcn_layers_for_e = ModuleList([])
        self.fcn_layers_for_v = ModuleList([])
        self.n_fcn_layers = estimator_params["estimator_fcn_layers"]
        estimation_embedding_dim = estimator_params["estimator_estimation_embedding_dim"]
        self.fcn_dropout_rate = estimator_params["estimator_fcn_dropout_rate"]
        self.fcn_layers.append(Linear(embedding_dim, estimation_embedding_dim))
        for i in range(self.n_fcn_layers - 1):
            if i == 0:
                self.fcn_layers.append(Linear(estimation_embedding_dim, int(estimation_embedding_dim / 4)))
                estimation_embedding_dim = int(estimation_embedding_dim / 4)
            else:
                self.fcn_layers.append(Linear(estimation_embedding_dim, int(estimation_embedding_dim / 2)))
                estimation_embedding_dim = int(estimation_embedding_dim / 2)

        estimation_embedding_dim_e = estimation_embedding_dim
        estimation_embedding_dim_v = estimation_embedding_dim
        for i in range(3):
            # Branch for latency estimation
            if i != 2:
                self.fcn_layers_for_e.append(Linear(estimation_embedding_dim_e, int(estimation_embedding_dim_e / 2)))
                estimation_embedding_dim_e = int(estimation_embedding_dim_e / 2)
            else:
                self.fcn_layers_for_e.append(Linear(estimation_embedding_dim_e, 1))
            # Branch for variance(uncertainty) quantification
            if i != 2:
                self.fcn_layers_for_v.append(Linear(estimation_embedding_dim_v, int(estimation_embedding_dim_v / 2)))
                estimation_embedding_dim_v = int(estimation_embedding_dim_v / 2)
            else:
                self.fcn_layers_for_v.append(Linear(estimation_embedding_dim_v, 1))
        self.fcn_layers_for_v_activation = Softplus()

        self.fs = Sequential(Linear(2, 8), Linear(8, 1), Sigmoid())
        self.dropout = Dropout(p=self.fcn_dropout_rate)

    def forward(self, x):
        for i in range(self.n_fcn_layers-1):
            x = self.dropout(torch.relu(self.fcn_layers[i](x)))
        x = torch.relu(self.fcn_layers[-1](x))

        x_e = x
        for i in range(2):
            x_e = self.dropout(torch.relu(self.fcn_layers_for_e[i](x_e)))
        x_e = torch.sigmoid(self.fcn_layers_for_e[-1](x_e))

        x_v = x
        for i in range(2):
            x_v = self.dropout(torch.relu(self.fcn_layers_for_v[i](x_v)))
        x_v = self.fcn_layers_for_v_activation(self.fcn_layers_for_v[-1](x_v))

        # Integrate the estimated latency and quantified variance(uncertainty)
        x_iv = self.fs(torch.cat([x_e, x_v], dim=1))

        return x_e, x_v, x_iv

class Reqo(torch.nn.Module):
    def __init__(self, encoder_params, estimator_params, explainer_params):
        super(Reqo, self).__init__()
        self.node_feature_dim = encoder_params["encoder_node_type_embedding_dim"] + 2 + encoder_params["encoder_table_num"] + encoder_params["encoder_table_num"] * encoder_params["encoder_column_embedding_dim"]
        self.embedding_dim = encoder_params["encoder_gnn_embedding_dim"]
        self.feature_encoder = Feature_encoder(encoder_params)
        self.bigg = BiGG(encoder_params, self.node_feature_dim)
        self.estimator = Estimator(estimator_params, self.embedding_dim)
        self.explainer = Explainer(explainer_params, self.embedding_dim*2)


    def forward(self, batch, table_columns_number, subtree_index, subtree_labels):
        encoded_tree = self.feature_encoder(batch.x.float(), table_columns_number)
        global_output, local_output, global_labels, local_labels = self.bigg(encoded_tree, batch.edge_index, batch.y.long(), subtree_index, subtree_labels)
        pred, va, iv = self.estimator(global_output)
        expl = self.explainer(local_output)
        return pred, va, iv, expl, global_labels, local_labels

    def predict_without_explainer(self, batch, table_columns_number):
        encoded_tree = self.feature_encoder(batch.x.float(), table_columns_number)
        rep = self.bigg(encoded_tree, batch.edge_index, batch.batch)
        pred, va, iv = self.estimator(rep)
        return pred, va, iv
