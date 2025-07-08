import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import dill
from network.network_generator import NetworkGenerator


class OFDMNetworkGenerator(NetworkGenerator):
    def __init__(self, data_dir, num_ue_range, num_rb=None, num_beam=None):
        super(OFDMNetworkGenerator).__init__()
        self.data_dir = Path(__file__).parents[0].resolve() / data_dir
        self.file_list = [os.path.join(self.data_dir, f)
                          for f in os.listdir(self.data_dir) if f.endswith('.pkl') and f != 'background.pkl']
        background_file = self.data_dir / 'background.pkl'
        with open(background_file, 'rb') as f:
            self.background = dill.load(f)
        with open(self.file_list[0], 'rb') as f:
            sample = dill.load(f)
        self.num_bs, self.num_rb, self.num_beam = sample['ch'].shape
        if num_rb is not None:
            if num_rb > self.num_rb:
                raise ValueError(f'num_rb in configuration must be equal to or smaller than {self.num_rb}')
            else:
                self.num_rb = num_rb
        if num_beam is not None and self.num_beam != num_beam:
            raise ValueError(f'num_beam in configuration must be {self.num_beam}')
        self.num_ue_range = num_ue_range
        self.rb_size = self.background['rb_size']  # number of subcarriers in one RB
        self.subcarrier_spacing = self.background['subcarrier_spacing']  # subcarrier spacing (Hz)

    def generate_network(self):
        num_ue = np.random.randint(low=self.num_ue_range[0], high=self.num_ue_range[1])
        num_ue = min(num_ue, len(self.file_list))
        file_list = np.random.choice(self.file_list, size=num_ue, replace=False)
        ch, pos = [], []
        for file in file_list:
            with open(file, 'rb') as f:
                data = dill.load(f)
            if np.sum(data['ch']) == 0.0:
                continue
            ch.append(data['ch'])
            pos.append(data['pos'])
        ch = np.stack(ch, axis=0)  # ue, bs, rb, beam
        ch = ch[:, :, :self.num_rb, :]  # Select first num_rb RBs
        pos = np.stack(pos, axis=0)  # ue, pos
        pow = np.sum(np.sum(ch, axis=-1), axis=-1)  # ue, bs
        assoc = np.argmax(pow, axis=1)  # ue
        return {'ch': ch, 'pos': pos, 'assoc': assoc, 'rb_size': self.rb_size, 'subcarrier_spacing': self.subcarrier_spacing}

    def plot(self, network):
        z_max = 2000
        ue_position = network['pos']
        ue_assoc = network['assoc']
        mesh_faces = self.background['mesh_faces']
        mesh_vertices = self.background['mesh_vertices']
        bs_position = self.background['bs_position']
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_zlim(0, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        x, y, z = mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2]
        ax.plot_trisurf(x, y, z, color='cyan', alpha=0.7, triangles=mesh_faces)
        for bs_pos in bs_position:
            x, y, z = bs_pos[0], bs_pos[1], bs_pos[2]
            ax.scatter(x, y, z, s=60, color='red', alpha=1.0, depthshade=False, edgecolors='red')
        for ue_pos, bs_idx in zip(ue_position, ue_assoc):
            ux, uy, uz = ue_pos[0], ue_pos[1], 5
            ax.scatter(ux, uy, uz, s=30, color='orange', alpha=1.0, depthshade=False, edgecolors='orange')
            bs_pos = bs_position[bs_idx]
            bx, by, bz = bs_pos[0], bs_pos[1], bs_pos[2]
            ax.plot([bx, ux], [by, uy], [bz, uz], color='black', linewidth=2)
        ax.view_init(elev=90, azim=-90)
        ax.set_proj_type('ortho')
        plt.show()

