import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.init import xavier_uniform_, constant_
from torch.distributions.categorical import Categorical
from torch_geometric.nn import TransformerConv, AttentionalAggregation, MetaLayer
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
from torch_scatter import scatter_mean

class ActorCritic(nn.Module):
    def __init__(self, input_dim, embedding_dim, edge_dim, action_shape,
                 d_model, n_head, dim_feedforward, num_layers, dropout, 
                 activation, global_dim):
        super().__init__()
        
        # 기존 파라미터들
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._edge_dim = edge_dim
        self._action_shape = np.array(action_shape)
        self._action_dim = int(np.prod(self._action_shape))
        self._d_model = d_model
        self._n_head = n_head
        self._dim_feedforward = dim_feedforward
        self._num_layers = num_layers
        self._dropout = dropout
        self._activation = activation
        
        # Graph Transformer
        self._graph_transformer = GraphTransformer(
            input_dim=self._input_dim, 
            embedding_dim=self._embedding_dim, 
            edge_dim=self._edge_dim,
            num_layers=1, 
            d_model=self._d_model, 
            n_head=self._n_head,
            dim_feedforward=self._dim_feedforward, 
            dropout=self._dropout, 
            activation=self._activation
        )

        self.global_encoder = nn.Linear(global_dim, d_model)
        self.edge_encoder = nn.Linear(edge_dim, d_model)
        self.layers = nn.ModuleList([
            MetaLayer(
                edge_model=EdgeModel(d_model, d_model, d_model, dim_feedforward),
                node_model=NodeModel(d_model, d_model, d_model, dim_feedforward),
                global_model=GlobalModel(d_model, d_model, d_model, dim_feedforward)
            ) for _ in range(num_layers)
        ])
        self.actor_head = nn.Linear(d_model, self._action_dim)
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()
    
    def _reset_parameters(self):
        modules_to_init = [
        self.actor_head,              # nn.Linear
        self.critic_head,             # nn.Sequential
        self.global_encoder,          # nn.Linear
        self.edge_encoder,            # nn.Linear
        ]
        for module in modules_to_init:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=1.0)
                        nn.init.constant_(layer.bias, 0.0)
        # GraphTransformer 내부 Linear 계층 초기화
        if hasattr(self._graph_transformer, "_input_linear"):
            nn.init.orthogonal_(self._graph_transformer._input_linear.weight, gain=1.0)
            nn.init.constant_(self._graph_transformer._input_linear.bias, 0.0)
        if hasattr(self._graph_transformer, "_node_embedding_linear"):
            nn.init.orthogonal_(self._graph_transformer._node_embedding_linear.weight, gain=1.0)
            nn.init.constant_(self._graph_transformer._node_embedding_linear.bias, 0.0)
        if hasattr(self._graph_transformer, "_fusion_linear"):
            nn.init.orthogonal_(self._graph_transformer._fusion_linear.weight, gain=1.0)
            nn.init.constant_(self._graph_transformer._fusion_linear.bias, 0.0)
        # MetaLayer 내부 모듈(옵션: 각 레이어 별로 커스텀 초기화)
        for layer in self.layers:
            for submodule in [layer.edge_model, layer.node_model, layer.global_model]:
                for mod in submodule.modules():
                    if isinstance(mod, nn.Linear):
                        nn.init.orthogonal_(mod.weight, gain=1.0)
                        nn.init.constant_(mod.bias, 0.0)


    def forward(self, input, node_embedding, edge_attr, edge_index, ptr, batch, global_feat):
        # Graph Transformer 통과
        x = self._graph_transformer(
            input=input, 
            node_embedding=node_embedding,
            edge_attr=edge_attr, 
            edge_index=edge_index
        )                                           # [N, d_model]
        edge_attr = self.edge_encoder(edge_attr)    # [E, d_model]
        u = self.global_encoder(global_feat)        # [B, d_model]
        for layer in self.layers:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)

        # 3. Critic: u→MLP
        value = self.critic_head(u).squeeze(-1)     # [B]

        # Actor 정책 로짓
        policy_logit = self.actor_head(x)
        num_batch = int(ptr.shape[0]) - 1
        policy_logit_list = []
        
        for idx in range(num_batch):
            l = policy_logit[ptr[idx]: ptr[idx + 1], :]
            l = torch.reshape(l, shape=(-1, *self._action_shape))
            policy_logit_list.append(l)
        
        return policy_logit_list, value


class ActionDistribution:
    def __init__(self, logit):
        self._device = logit.device
        self._action_shape = np.array(logit.shape)[1:]
        logit = torch.flatten(logit)
        self._dist = None
        if not torch.all(torch.isinf(logit)):
            self._dist = Categorical(logits=logit)

    def sample(self):
        if self._dist is not None:
            idx = int(self._dist.sample())
            action = []
            for n in np.flip(self._action_shape):
                action.append(idx % n)
                idx = idx // n
            action.append(idx)
            action.reverse()
            action = tuple(action)
        else:
            action = None
        return action

    def entropy(self):
        entropy = self._dist.entropy() if self._dist is not None else torch.tensor(0.0, device=self._device)
        return entropy

    def log_prob(self, action):
        # action (tuple)
        action = np.array(action)
        if self._dist is not None:
            for i, n in enumerate(np.flip(self._action_shape)):
                action[:-1-i] *= n
            idx = torch.tensor(np.sum(action), dtype=torch.int, device=self._device)
            lp = self._dist.log_prob(idx)
        else:
            lp = None
        return lp


class GraphTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, d_model, n_head, edge_dim, dim_feedforward, dropout, activation="relu", device='cpu'):
        super(GraphTransformer, self).__init__()
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._d_model = d_model
        self._n_head = n_head
        self._edge_dim = edge_dim
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._device = device
        self._activation = activation
        self._input_linear = Linear(in_features=self._input_dim, out_features=self._d_model, bias=True, device=device)
        self._node_embedding_linear = Linear(in_features=self._embedding_dim, out_features=self._d_model, bias=True, device=device)
        self._fusion_linear = Linear(in_features=d_model * 3, out_features=d_model)
        self._layer_list = nn.ModuleList()
        self.final_norm = LayerNorm(d_model, eps=1e-5, device=device)
        for _ in range(self._num_layers):
            layer = GraphTransformerLayer(d_model=self._d_model, n_head=self._n_head,
                                          edge_dim=self._edge_dim,
                                          dim_feedforward=self._dim_feedforward, dropout=self._dropout,
                                          activation=self._activation, device=self._device)
            self._layer_list.append(layer)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self._input_linear.weight)
        xavier_uniform_(self._node_embedding_linear.weight)
        constant_(self._input_linear.bias, 0.)
        constant_(self._node_embedding_linear.bias, 0.)

    def forward(self, input, node_embedding, edge_attr, edge_index):
        input = self._input_linear(input)
        node_embedding = self._node_embedding_linear(node_embedding)
        x = node_embedding
        for layer in self._layer_list:
            combined_features = torch.cat([x, input, node_embedding], dim=-1)
            x = self._fusion_linear(combined_features)
            x = layer(x, edge_attr, edge_index)
        x = self.final_norm(x)
        return x


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, edge_dim, dim_feedforward, dropout, activation="relu", device='cpu'):
        super(GraphTransformerLayer, self).__init__()
        self._d_model = d_model
        self._n_head = n_head
        self._edge_dim = edge_dim
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._device = device
        self._activation = activation
        # Transformer convolution
        out_channel = d_model // n_head
        self._trans_conv = TransformerConv(in_channels=d_model, out_channels=out_channel, heads=n_head,
                                           concat=True, beta=False, dropout=dropout, edge_dim=edge_dim,
                                           bias=True, root_weight=True).to(device)
        # Feedforward neural network
        self.ffnn_linear1 = Linear(in_features=d_model, out_features=dim_feedforward, bias=True, device=device)
        self.ffnn_dropout = Dropout(dropout)
        self.ffnn_linear2 = Linear(in_features=dim_feedforward, out_features=d_model, bias=True, device=device)
        # Layer norm and dropout
        layer_norm_eps = 1e-5
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        # Activation
        self.activation = self._get_activation_fn(activation)
        # Reset parameters
        self._reset_parameters()

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")

    def _reset_parameters(self):
        xavier_uniform_(self.ffnn_linear1.weight)
        xavier_uniform_(self.ffnn_linear2.weight)
        constant_(self.ffnn_linear1.bias, 0.)
        constant_(self.ffnn_linear2.bias, 0.)
        self._trans_conv.reset_parameters()

    def forward(self, x, edge_attr, edge_index):
        # x2 = self._trans_conv(x=x, edge_index=edge_index.long(), edge_attr=edge_attr, return_attention_weights=None)
        # x = x + self.dropout1(x2)
        # x = self.norm1(x)
        # x2 = self.ffnn_linear2(self.ffnn_dropout(self.activation(self.ffnn_linear1(x))))
        # x = x + self.dropout2(x2)
        # x = self.norm2(x)

        x_norm = self.norm1(x)
        x2 = self._trans_conv(x=x_norm, edge_index=edge_index.long(), edge_attr=edge_attr, return_attention_weights=None)
        x = x + self.dropout1(x2)
        
        x_norm = self.norm2(x)
        x2 = self.ffnn_linear2(self.ffnn_dropout(self.activation(self.ffnn_linear1(x_norm))))
        x = x + self.dropout2(x2)

        return x


class EdgeModel(nn.Module):
    def __init__(self, edge_dim, node_dim, global_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [num_edges, node_dim], edge_attr: [num_edges, edge_dim], u: [num_graphs, global_dim], batch: [num_edges]
        u_per_edge = u[batch]
        h = torch.cat([src, dest, edge_attr, u_per_edge], dim=-1)
        return self.mlp(h)


class NodeModel(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        # Message aggregation: mean of incoming edge features
        row, col = edge_index
        agg = scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))
        u_per_node = u[batch]
        h = torch.cat([x, agg, u_per_node], dim=-1)
        return self.mlp(h)


class GlobalModel(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_dim + node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, global_dim)
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x/edge_attr: [num_nodes/edges, d_model], batch: [num_nodes], edge_batch: [num_edges]
        edge_batch = batch[edge_index[0]]
        x_mean = global_mean_pool(x, batch)
        edge_mean = global_mean_pool(edge_attr, edge_batch)
        h = torch.cat([u, x_mean, edge_mean], dim=-1)
        return self.mlp(h)
