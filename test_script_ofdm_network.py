import numpy as np
import ray
from pathlib import Path
import yaml
from network.ofdm.ofdm_network_generator import OFDMNetworkGenerator
from network.ofdm.ofdm_network_environment import OFDMNetworkEnvironment
from network.ofdm.ofdm_graph_converter import OFDMGraphConverter
from network.network_actor import get_network_actor


# # Test incremental move operation of OFDM network environment
# ng = OFDMNetworkGenerator(data_dir='myeongdong_arr_4_rb_16', num_ue_range=[50, 100], num_rb=12, num_beam=4)
# network = ng.generate_network()
# max_tx_power = 5
# num_tx_power_level = 17
# max_bs_power = 30
# env = OFDMNetworkEnvironment(network, max_tx_power, num_tx_power_level, max_bs_power, noise_spectral_density=-174.0,
#                              alpha=0.0, allow_reallocation=False)
# for step in range(10000):
#     m = env.get_random_move()
#     if m is None:
#         break
#     env.move(m)
#     if (step + 1) % 100 == 0:
#         prev_bs_total_power = env.bs_total_power
#         prev_tx_power = env.tx_power
#         prev_rx_power = env.rx_power
#         prev_interference = env.interference
#         prev_power_mask = env.power_mask
#         prev_score = env.score
#         env.compute_network_state_and_score()
#         print(f"bs_total_power_diff: {np.mean(np.square(env.bs_total_power - prev_bs_total_power)) / np.mean(np.square(env.bs_total_power))}")
#         print(f"tx_power_diff: {np.mean(np.square(env.tx_power - prev_tx_power)) / np.mean(np.square(env.tx_power))}")
#         print(f"rx_power_diff: {np.mean(np.square(env.rx_power - prev_rx_power)) / np.mean(np.square(env.rx_power))}")
#         print(f"interference_diff: {np.mean(np.square(env.interference - prev_interference)) / np.mean(np.square(env.interference))}")
#         print(f"score_diff: {np.square(env.score - prev_score) / np.square(env.score)}")
#         power_mask_diff = np.sum(env.power_mask != prev_power_mask)
#         print(f"power_mask_diff: {power_mask_diff}")
#         print(f"score: {env.score}")
#         print()


# # Graph converter test
# ng = OFDMNetworkGenerator(data_dir='myeongdong_arr_4_rb_16', num_ue_range=[50, 100], num_rb=12, num_beam=4)
# network = ng.generate_network()
# gc = OFDMGraphConverter(min_attn_db=-200, max_attn_db=-50, num_power_attn_level=10, prune_power_attn_thresh=-300)
# graph = gc.convert(network)
# pass


# # Network actor test
# with open(Path(__file__).parents[0] / 'config' / 'ofdm_ppo_config.yaml', 'r') as f:
#     conf = yaml.safe_load(f)
# network_conf = conf['network']
# na = get_network_actor(network_gen_cls=OFDMNetworkGenerator, network_env_cls=OFDMNetworkEnvironment,
#                        graph_converter_cls=OFDMGraphConverter, network_conf=network_conf, remote=False)
# na.generate_networks(num_networks=10)
# na.load_networks(directory_name='validation')
# networks = na.get_networks()
# network = na.get_network(id=3)
# graph_list = na.get_graph_list()
# graph = na.get_graph(id=3)
# env_info = na.get_env_info()


# Network actor remote test
with open(Path(__file__).parents[0] / 'config' / 'ofdm_ppo_config.yaml', 'r') as f:
    conf = yaml.safe_load(f)
network_conf = conf['network']
na = get_network_actor(network_gen_cls=OFDMNetworkGenerator, network_env_cls=OFDMNetworkEnvironment,
                       graph_converter_cls=OFDMGraphConverter, network_conf=network_conf, remote=True)
ray.get(na.generate_networks.remote(num_networks=10))
ray.get(na.load_networks.remote(directory_name='validation'))
networks = ray.get(na.get_networks.remote())
network = ray.get(na.get_network.remote(id=3))
graph_list = ray.get(na.get_graph_list.remote())
graph = ray.get(na.get_graph.remote(id=3))
env_info = ray.get(na.get_env_info.remote())
ray.kill(na)
pass
