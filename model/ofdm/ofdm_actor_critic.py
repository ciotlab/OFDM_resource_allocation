import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from model.actor_critic import ActorCritic


class OFDMActorCritic(nn.Module):
    def __init__(self, network_conf, model_conf):
        super(OFDMActorCritic, self).__init__()
        self.num_rb = network_conf['generator']['num_rb']
        self.num_beam = network_conf['generator']['num_beam']
        self.num_tx_power_level = network_conf['environment']['num_tx_power_level']
        self.num_power_attn_level = network_conf['graph']['num_power_attn_level']
        self.input_dim = self.num_rb * (self.num_tx_power_level + self.num_beam)
        self.embedding_dim = self.num_rb * self.num_beam * self.num_power_attn_level
        self.edge_dim = self.num_rb * self.num_beam * self.num_power_attn_level
        self.action_shape = (self.num_rb, self.num_tx_power_level, self.num_beam)
        self.actor_critic = ActorCritic(input_dim=self.input_dim, embedding_dim=self.embedding_dim,
                                        edge_dim=self.edge_dim, action_shape=self.action_shape, **model_conf)
        self.max_bs_power = network_conf['environment']['max_bs_power']

    def forward(self, data):
        # data: list of dictionary of graph, power_level, beam_idx, allocated
        # graph: PyG graph, power_level: (ue, rb), beam_idx: (ue, rb), allocated: (ue, rb)
        # Generate input
        device = next(self.parameters()).device
        keys = data[0].keys()
        data = {k: [d[k] for d in data] for k in keys}
        graph, power_level, beam_idx, allocated = data['graph'], data['power_level'], data['beam_idx'], data['allocated']
        allocated = torch.tensor(np.concatenate(allocated, axis=0)).to(device)  # batch/node, rb
        power_level = torch.tensor(np.concatenate(power_level, axis=0)).to(device)  # batch/node, rb
        power_level = F.one_hot(power_level.long(), self.num_tx_power_level).float()  # batch/node, rb, power_level
        power_level[torch.logical_not(allocated[:, :, None].expand(-1, -1, self.num_tx_power_level))] = 0.0
        beam_idx = torch.tensor(np.concatenate(beam_idx, axis=0)).to(device)  # batch/node, rb
        beam_idx = F.one_hot(beam_idx.long(), self.num_beam).float()  # batch/node, rb, beam
        beam_idx[torch.logical_not(allocated[:, :, None].expand(-1, -1, self.num_beam))] = 0.0
        input = torch.cat([power_level, beam_idx], dim=2).flatten(start_dim=-2, end_dim=-1)  # batch/node, input_dim

        global_feat = torch.tensor(np.full((len(graph), 1), self.max_bs_power, dtype=np.float32)).to(device)    # [batch_size, 1]  
        # Process graph
        g = Batch.from_data_list(graph)
        node_power_attn, edge_power_attn, edge_index, ptr, batch = (
            g.x.to(device), g.edge_attr.to(device), g.edge_index.to(device), g.ptr.to(device), g.batch.to(device))
        node_power_attn = F.one_hot(node_power_attn.long(), self.num_power_attn_level).float()  # batch/node, rb, beam, power_attn_level
        edge_power_attn = F.one_hot(edge_power_attn.long(), self.num_power_attn_level).float()  # batch/node, rb, beam, power_attn_level
        node_embedding = node_power_attn.flatten(start_dim=-3, end_dim=-1)  # batch/node, embedding_dim
        edge_attr = edge_power_attn.flatten(start_dim=-3, end_dim=-1)  # batch/edge, edge_dim
        policy_logit_list, value = self.actor_critic(input=input, node_embedding=node_embedding, edge_attr=edge_attr,
                                                     edge_index=edge_index, ptr=ptr, batch=batch, global_feat=global_feat)
        return policy_logit_list, value
