import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.init import xavier_uniform_, constant_
from torch.distributions.categorical import Categorical
from torch_geometric.nn import TransformerConv, AttentionalAggregation
from torch_geometric.nn.pool import global_mean_pool


class ActorCritic(nn.Module):
    def __init__(self, input_dim, embedding_dim, edge_dim, action_shape,
                 d_model, n_head, dim_feedforward, num_layers, dropout, activation='gelu'):
        super(ActorCritic, self).__init__()
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
        self._graph_transformer = GraphTransformer(
            input_dim=self._input_dim, embedding_dim=self._embedding_dim, edge_dim=self._edge_dim,
            num_layers=self._num_layers, d_model=self._d_model, n_head=self._n_head,
            dim_feedforward=self._dim_feedforward, dropout=self._dropout, activation=self._activation
        )
        self._actor_linear = Linear(in_features=self._d_model, out_features=self._action_dim)
        self._critic_head = nn.Sequential(
                                Linear(in_features=self._d_model, out_features=self._d_model),
                                nn.ReLU(),
                                Linear(in_features=self._d_model, out_features=self._d_model),
                                nn.ReLU(),
                                Linear(in_features=self._d_model, out_features=1)
                            )
        gate_nn = nn.Sequential(nn.Linear(self._d_model, 1))
        self._attention_pool = AttentionalAggregation(gate_nn)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self._actor_linear.weight)
        nn.init.constant_(self._actor_linear.bias, 0.)
        # nn.init.xavier_uniform_(self._critic_linear.weight)
        # nn.init.constant_(self._critic_linear.bias, 0.)
        for layer in self._critic_head:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight)
                constant_(layer.bias, 0.)

    def forward(self, input, node_embedding, edge_attr, edge_index, ptr, batch):
        x = self._graph_transformer(input=input, node_embedding=node_embedding,
                                    edge_attr=edge_attr, edge_index=edge_index)
        value = self._attention_pool(x, batch)
        value = self._critic_linear(value)[:, 0]  # batch
        policy_logit = self._actor_linear(x)  # batch/node, action
        num_batch = int(ptr.shape[0]) - 1
        policy_logit_list = []
        for idx in range(num_batch):
            l = policy_logit[ptr[idx]: ptr[idx + 1], :]  # node, action
            l = torch.reshape(l, shape=(-1, *self._action_shape))  # node, action_dim
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
        self._layer_list = nn.ModuleList()
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
        x = self._node_embedding_linear(node_embedding)
        for layer in self._layer_list:
            x = x + input
            x = layer(x, edge_attr, edge_index)
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
        x2 = self._trans_conv(x=x, edge_index=edge_index.long(), edge_attr=edge_attr, return_attention_weights=None)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.ffnn_linear2(self.ffnn_dropout(self.activation(self.ffnn_linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
