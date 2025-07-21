import numpy as np
import torch
from torch_geometric.data import Data
from network.graph_converter import GraphConverter


class OFDMGraphConverter(GraphConverter):
    def __init__(self, min_attn_db, max_attn_db, num_power_attn_level, prune_power_attn_thresh=None):
        super(OFDMGraphConverter, self).__init__()
        self.min_attn_db = min_attn_db
        self.max_attn_db = max_attn_db
        self.num_power_attn_level = num_power_attn_level
        self.prune_power_attn_thresh = prune_power_attn_thresh

    def convert(self, network):
        quantization_step = (self.max_attn_db - self.min_attn_db) / (self.num_power_attn_level - 2)
        ch = network['ch']  # ue, bs, rb, beam
        ch = 10.0 * np.log10(ch + 1e-200)  # dB scale
        ch_quant = np.floor((ch - self.min_attn_db) / quantization_step).astype(int) + 1
        ch_quant[ch_quant < 0] = 0
        ch_quant[ch_quant >= self.num_power_attn_level] = self.num_power_attn_level - 1  # ue, bs, rb, beam
        assoc = network['assoc']  # ue
        num_link, num_bs, num_rb, num_beam = ch.shape
        node_attr = []
        edge_index = []
        edge_attr = []
        for l in range(num_link):
            node_attr.append(ch_quant[l, assoc[l]])  # rb, beam
            for i in range(num_link):
                if i != l:
                    interf = ch[l, assoc[i]]  # rb, beam
                    interf_quant = ch_quant[l, assoc[i]]  # rb, beam
                    if self.prune_power_attn_thresh is not None and np.mean(interf) < self.prune_power_attn_thresh:
                        continue
                    edge_index.append(np.array((i, l)))
                    edge_attr.append(interf_quant)
        node_attr = torch.tensor(np.stack(node_attr, axis=0), dtype=torch.int32)
        edge_index = torch.tensor(np.stack(edge_index, axis=1), dtype=torch.long)
        edge_attr = torch.tensor(np.stack(edge_attr, axis=0), dtype=torch.int32)
        graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
        return graph