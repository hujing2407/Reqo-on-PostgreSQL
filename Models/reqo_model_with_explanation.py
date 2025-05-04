import torch
from torch.nn import Linear, ModuleList, MaxPool2d, Dropout, Sequential, Sigmoid, Softplus
from torch_geometric.nn import TransformerConv, BatchNorm, GRUAggregation
from .DirGNNConv import DirGNNConv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        W = torch.stack([layer.weight for layer in self.column_layers], dim=0)
        node_column_new_enc = torch.relu(torch.bmm(node_column_enc, W.transpose(1, 2)))

        chunks = torch.split(node_column_new_enc.permute(1, 0, 2), tuple(table_columns_number), dim=1)
        pooled = [table.max(dim=1).values for table in chunks]
        node_table_enc = torch.cat(pooled, dim=1)

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

        g = torch.tensor(global_tree_index, device=device)
        starts = torch.cat([g.new_tensor([-1]), g[:-1]])
        counts = g - starts 
        tree_ids = torch.repeat_interleave(torch.arange(len(g), device=device), counts)
        global_idx = g[tree_ids]
        M = torch.cat([x, x[global_idx]], dim=1)
        return y, M, batch_global_labels, batch_local_labels

    def prepare_batches_for_subtrees(self, batch_subtree_index, subtree_index, subtree_labels):
        B = len(batch_subtree_index)
        labels_list = [subtree_labels[i].to(device) for i in batch_subtree_index]
        index_list = [subtree_index[i].to(device) for i in batch_subtree_index]
        all_labels, all_index = torch.cat(labels_list, 0), torch.cat(index_list, 0)

        label_counts = torch.tensor([l.size(0) for l in labels_list], device=device)
        tree_ids = torch.repeat_interleave(torch.arange(B, device=device), label_counts)

        out = all_labels.new_full((B,), -float('inf'))
        global_labels = torch.scatter_reduce(out, 0, tree_ids, all_labels, 'amax', include_self=True)
        local_labels = all_labels / global_labels[tree_ids]

        index_counts = torch.tensor([idx.size(0) for idx in index_list], device=device)
        sizes, ends = torch.tensor([idx.max().item() + 1 for idx in index_list], device=device), None
        ends = torch.cumsum(sizes, 0);
        starts = ends - sizes
        batched_index = all_index + torch.repeat_interleave(starts, index_counts)

        global_tree_index = (ends - 1).tolist()

        return batched_index, global_tree_index, global_labels, local_labels

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
        dims = [explanation_embedding_dim // (2 ** (i + 1)) for i in range(self.n_explainer_layers - 2)] + [1]
        for out_dim in dims:
            self.explainer_layers.append(Linear(explanation_embedding_dim, out_dim))
            explanation_embedding_dim = out_dim
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
        divs = [4] + [2] * (self.n_fcn_layers - 2)
        for div in divs:
            out_dim = estimation_embedding_dim // div
            self.fcn_layers.append(Linear(estimation_embedding_dim, out_dim))
            estimation_embedding_dim = out_dim

        dims = [estimation_embedding_dim // (2 ** i) for i in range(3)]
        for in_dim, out_dim in zip(dims, dims[1:] + [1]):
            self.fcn_layers_for_e.append(Linear(in_dim, out_dim))
            self.fcn_layers_for_v.append(Linear(in_dim, out_dim))

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
