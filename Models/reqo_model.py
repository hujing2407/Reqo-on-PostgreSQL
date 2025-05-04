import torch
from torch.nn import Linear, ModuleList, MaxPool2d, Dropout, Softplus, Sequential, Sigmoid
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

    def forward(self, node_features, edge_index, batch_index):
        x = node_features
        for i in range(self.n_conv_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            x = torch.relu(x)
        x = self.aggr_layer(x, batch_index)
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
    def __init__(self, encoder_params, estimator_params):
        super(Reqo, self).__init__()
        self.node_feature_dim = encoder_params["encoder_node_type_embedding_dim"] + 2 + encoder_params["encoder_table_num"] + encoder_params["encoder_table_num"] * encoder_params["encoder_column_embedding_dim"]
        self.embedding_dim = encoder_params["encoder_gnn_embedding_dim"]
        self.feature_encoder = Feature_encoder(encoder_params)
        self.bigg = BiGG(encoder_params, self.node_feature_dim)
        self.estimator = Estimator(estimator_params, self.embedding_dim)

    def forward(self, batch, table_columns_number):
        encoded_tree = self.feature_encoder(batch.x.float(), table_columns_number)
        rep = self.bigg(encoded_tree, batch.edge_index, batch.batch)
        pred, va, iv = self.estimator(rep)
        return pred, va, iv

