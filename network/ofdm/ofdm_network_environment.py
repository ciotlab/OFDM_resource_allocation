import numpy as np
from itertools import product
import random
from network.network_environment import NetworkEnvironment


class OFDMNetworkEnvironment(NetworkEnvironment):
    def __init__(self, network, max_tx_power, num_tx_power_level, max_bs_power, noise_spectral_density=-174.0, alpha=1.0,
                 allow_reallocation=False):
        super(OFDMNetworkEnvironment, self).__init__()
        self.network = network
        ch = self.network['ch']  # ue, bs, rb, beam
        self.num_ue, self.num_bs, self.num_rb, self.num_beam = ch.shape
        self.assoc = self.network['assoc']  # ue
        self.ch = np.take_along_axis(arr=ch[:, np.newaxis, :, :, :],
                                     indices=self.assoc[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis],
                                     axis=2)[:, :, 0, :, :]  # target ue, source ue, rb, beam
        self.power_level_map = np.linspace(start=0.0, stop=max_tx_power, num=num_tx_power_level)
        self.num_tx_power_level = num_tx_power_level
        self.max_bs_power = max_bs_power
        rb_size = self.network['rb_size']  # number of subcarriers in one RB
        subcarrier_spacing = self.network['subcarrier_spacing']  # subcarrier spacing (Hz)
        noise_spectral_density = np.power(10.0, noise_spectral_density / 10.0) / 1000.0  # noise spectral density (W/Hz)
        self.noise_power = noise_spectral_density * subcarrier_spacing * rb_size  # noise power (W)
        self.alpha = alpha
        self.allow_reallocation = allow_reallocation
        # Resource allocation state
        self.power_level = None  # ndarray (ue, rb)
        self.beam_idx = None  # ndarray (ue, rb)
        self.allocated = None  # ndarray (ue, rb)
        # Network state
        self.bs_total_power = None  # ndarray (bs,)
        self.tx_power = None   # ndarray (ue, rb)
        self.rx_power = None  # ndarray (ue, rb)
        self.interference = None  # ndarray (ue, rb)
        self.power_mask = None  # ndarray (ue, rb, power), True if power is not allowed
        # Total score
        self.score = None
        # Initialization
        self.reset()

    def reset(self):
        power_level = np.full(shape=(self.num_ue, self.num_rb), fill_value=0, dtype=np.int32)
        beam_idx = np.full(shape=(self.num_ue, self.num_rb), fill_value=0, dtype=np.int32)
        allocated = np.full(shape=(self.num_ue, self.num_rb), fill_value=False, dtype=bool)
        resource_allocation = {'power_level': power_level, 'beam_idx': beam_idx, 'allocated': allocated,}
        self.set_resource_allocation(resource_allocation)

    def set_resource_allocation(self, resource_allocation):
        self.power_level = resource_allocation['power_level']
        self.beam_idx = resource_allocation['beam_idx']
        allocated = resource_allocation['allocated']
        self.allocated = allocated if allocated is not None\
            else np.full(shape=(self.num_ue, self.num_rb), fill_value=True, dtype=bool)
        self.compute_network_state_and_score()

    def get_resource_allocation(self):
        return {'power_level': self.power_level.copy(), 'beam_idx': self.beam_idx.copy(),
                'allocated': self.allocated.copy()}

    def compute_network_state(self):
        # Compute tx power
        self.tx_power = np.take_along_axis(arr=self.power_level_map[np.newaxis, np.newaxis, :],
                                           indices=self.power_level[:, :, np.newaxis], axis=-1)[:, :, 0]  # ue, rb
        self.tx_power[np.logical_not(self.allocated)] = 0.0
        # Compute total BS power
        ue_power = np.sum(self.tx_power, axis=-1)  # ue
        self.bs_total_power = np.zeros((self.num_ue, self.num_bs))
        np.put_along_axis(arr=self.bs_total_power, indices=self.assoc[:, np.newaxis], values=ue_power[:, np.newaxis], axis=-1)
        self.bs_total_power = np.sum(self.bs_total_power, axis=0)  # bs
        # Compute rx power and interference
        ch = np.take_along_axis(arr=self.ch, indices=self.beam_idx[np.newaxis, :, :, np.newaxis], axis=3)[:, :, :, 0]  # target ue, source ue, rb
        rxp = ch * self.tx_power[np.newaxis, :, :]  # target ue, source ue, rb
        self.rx_power = np.swapaxes(np.diagonal(rxp, axis1=0, axis2=1), axis1=0, axis2=1).copy()  # ue, rb
        self.interference = np.sum(rxp, axis=1) - self.rx_power  # ue, rb
        # Compute power mask
        bs_power_for_ue = np.take_along_axis(arr=self.bs_total_power[np.newaxis, :], indices=self.assoc[:, np.newaxis],
                                             axis=-1)[:, 0]  # ue
        ue_available_power = (self.max_bs_power - bs_power_for_ue)[:, np.newaxis] + self.tx_power  # ue, rb
        self.power_mask = ue_available_power[:, :, np.newaxis] < self.power_level_map  # ue, rb, power

    def compute_score(self):
        sinr = self.rx_power / (self.interference + self.noise_power)  # ue, rb
        spec_eff = np.mean(np.log2(1 + sinr), axis=1)  # ue
        se = spec_eff + 1E-20  # for numerical stability
        if self.alpha == 1.0:
            self.score = np.sum(np.log(se))
        else:
            self.score = np.sum(np.power(se, 1 - self.alpha) / (1 - self.alpha))

    def move(self, m):
        ue, rb, power_level, cur_beam_idx = m
        bs = self.assoc[ue]
        # Compute power difference
        prev_tx_power = self.tx_power[ue, rb]
        cur_tx_power = self.power_level_map[power_level]
        delta_tx_power = cur_tx_power - prev_tx_power
        prev_beam_idx = self.beam_idx[ue, rb]
        prev_ch = self.ch[:, ue, rb, prev_beam_idx]  # target ue
        prev_rxp = prev_ch * prev_tx_power  # target ue
        cur_ch = self.ch[:, ue, rb, cur_beam_idx]  # target ue
        cur_rxp = cur_ch * cur_tx_power  # target ue
        delta_rxp = cur_rxp - prev_rxp
        delta_rx_power = delta_rxp[ue]
        delta_rxp[ue] = 0.0
        delta_interference = delta_rxp  # target ue
        # Update network state
        self.power_level[ue, rb] = power_level
        self.beam_idx[ue, rb] = cur_beam_idx
        self.allocated[ue, rb] = True
        self.tx_power[ue, rb] = cur_tx_power
        self.bs_total_power[bs] += delta_tx_power
        self.rx_power[ue, rb] += delta_rx_power
        self.interference[:, rb] += delta_interference
        # Update power mask
        ue_set = (self.assoc == bs)
        ue_available_power = self.max_bs_power - self.bs_total_power[bs] + self.tx_power[ue_set, :]  # ue, rb
        self.power_mask[ue_set, :, :] = ue_available_power[:, :, np.newaxis] < self.power_level_map  # ue, rb, power
        self.compute_score()

    def get_move_mask(self):
        mask = self.power_mask.copy()  # ue, rb, power
        if not self.allow_reallocation:
            mask = np.logical_or(mask, self.allocated[:, :, np.newaxis])
        return mask  # ue, rb, power

    def get_all_available_moves(self):
        mask = self.get_move_mask()
        moves = list(zip(*np.where(np.logical_not(mask))))
        beam_indices = list(np.arange(self.num_beam))
        moves = list(product(moves, beam_indices))
        moves = [(i, j, k, l) for ((i, j, k), l) in moves]
        return moves

    def get_random_move(self):
        moves = self.get_all_available_moves()
        if len(moves) == 0:
            return None
        move = random.choice(moves)
        return move

    def is_finished(self):
        mask = self.get_move_mask()
        return np.all(mask)

    def is_feasible(self):
        feasible = self.bs_total_power <= self.max_bs_power
        feasible = np.all(feasible)
        return feasible
