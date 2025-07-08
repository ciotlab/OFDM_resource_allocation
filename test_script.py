import os
import numpy as np
import torch
import asyncio
import logging
import ray
from ray import serve, train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.callback import Callback
import random
import torch.nn as nn

from network.network_generator import NetworkGenerator
from network.network_environment import NetworkEnvironment
from simulator.inference_server import InferenceServer
from utility.graph import generate_graph_from_network
from model.ofdm_model import OFDMActorCritic
from model.actor_critic import ActionDistribution
from simulator.on_policy_simulator import OnPolicySimulator
from trainer.ppo_trainer import ppo_train_loop_per_worker

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # Network environment test
# ng = NetworkGenerator(data_dir='myeongdong_arr_4_rb_16', num_ue_range=[50, 100], num_rb=12)
# network = ng.generate_network()
# max_tx_power = 5
# num_tx_power_level = 17
# max_bs_power = 30
# env = NetworkEnvironment(network, max_tx_power, num_tx_power_level, max_bs_power, noise_spectral_density=-174.0,
#                          alpha=0.0, allow_reallocation=False)
# # Test incremental move operation
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
#
# # PyG test
# ng = NetworkGenerator(data_dir='myeongdong_arr_4_rb_16', num_ue_range=[50, 100], num_rb=12)
# network = ng.generate_network()
# graph = generate_graph_from_network(network, min_attn_db=-200, max_attn_db=-50, num_power_attn_level=10,
#                                     prune_power_attn_thresh=-300)

# Prepare graph, environment, model
num_rb = 12
num_networks = 10
num_tx_power_level = 17
num_power_attn_level = 10
device = 'cuda'

ng = NetworkGenerator(data_dir='myeongdong_arr_4_rb_16', num_ue_range=[50, 100], num_rb=num_rb)
num_beam = ng.num_beam
networks = ng.generate_networks(num_networks)
graph_list = []
env_list = []
for network in networks:
    graph = generate_graph_from_network(network, min_attn_db=-200, max_attn_db=-50, num_power_attn_level=num_power_attn_level,
                                        prune_power_attn_thresh=-300)
    graph_list.append(graph)
    env = NetworkEnvironment(network=network, max_tx_power=5, num_tx_power_level=num_tx_power_level, max_bs_power=30,
                             noise_spectral_density=-174.0, alpha=0.0, allow_reallocation=False)
    env_list.append(env)
ac_model_conf = {'num_rb': num_rb, 'num_tx_power_level': num_tx_power_level, 'num_beam': num_beam,
                 'num_power_attn_level': num_power_attn_level, 'd_model': 512, 'n_head': 8, 'dim_feedforward': 1024,
                 'num_layers': 8, 'dropout': 0.0, 'activation': 'gelu'}
ac_model = OFDMActorCritic(**ac_model_conf).to(device)

# # Direct inference test
# for env in env_list:
#     for step in range(10):
#         m = env.get_random_move()
#         env.move(m)
# data = []
# for env, graph in zip(env_list, graph_list):
#     d = env.get_resource_allocation()
#     mask = env.get_move_mask()
#     d['graph'] = graph
#     d['move_mask'] = mask
#     data.append(d)
# policy_logit_list, value = ac_model(data)

# On policy simulation test
serve_num_replicas = 1
serve_max_ongoing_requests = 20
if ray.is_initialized():
    ray.shutdown()
ray.init(logging_level=logging.ERROR)
ray.data.DataContext.get_current().enable_progress_bars = False
ray.data.DataContext.get_current().execution_options.verbose_progress = False
ray.data.DataContext.get_current().print_on_execution_start = False
ray.data.DataContext.get_current().use_ray_tqdm = False
logging.getLogger("ray.train").setLevel(logging.ERROR)
logging.getLogger("ray.air").setLevel(logging.ERROR)
logging.getLogger("ray.data").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
# Define ray actors
gamma = 0.99
lamb = 1.0
num_actor = 5
train_params = {'num_epochs_per_episode': 1, 'batch_size': 32, 'learning_rate': 0.0001, 'act_prob_ratio_exponent_clip': 1.0,
                'ppo_clip': 0.1, 'entropy_loss_weight': 0.1, 'value_loss_weight': 1.0, 'clip_max_norm': 0.1}
actors = [OnPolicySimulator.remote(i, gamma, lamb) for i in range(num_actor)]
# Store graph and checkpoint to Ray object store
graph_list_ref = ray.put(graph_list)
param_dicts = [{"params": [p for n, p in ac_model.named_parameters() if p.requires_grad]}]
optimizer = torch.optim.Adam(param_dicts, lr=train_params["learning_rate"])
check_point = {'model_state_dict': ac_model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict()}
check_point_ref = ray.put(check_point)
# Start serv
serve.start(detached=False, logging_config={"log_level": logging.WARNING})
serve_deployment = InferenceServer.options(num_replicas=serve_num_replicas, max_ongoing_requests=serve_max_ongoing_requests,
                                           ray_actor_options={"num_gpus": 1})
app = serve_deployment.bind(model_conf=ac_model_conf, check_point_ref=check_point_ref,
                            graph_list_ref=graph_list_ref)
model_server_handle = serve.run(app, logging_config={"log_level": logging.WARNING})

num_data = len(graph_list)
base = num_data // num_actor
remainder = num_data % num_actor
assignments = []
for actor_id in range(num_actor):
    count = base + (1 if actor_id < remainder else 0)
    assignments.extend([actor_id] * count)
results_ref = []
for sim_id, env in enumerate(env_list):
    actor_id = assignments[sim_id]
    results_ref.append(actors[actor_id].run.remote(sim_id, env, model_server_handle))
all_train_buf = []

progress = {}
while not ray.wait(results_ref, timeout=1)[0]:
    for actor in actors:
        progress.update(ray.get(actor.get_progress.remote()))
    if progress:
        print("Simulation Progress: ", end="")
        for sim_id in range(len(results_ref)):
            if sim_id in progress:
                print(f"[{sim_id}: {progress[sim_id]:3.0%}]", end="")
        print()

for buf in ray.get(results_ref):
    all_train_buf.extend(buf)

random.shuffle(all_train_buf)
for d in all_train_buf:
    state = d.pop('state')
    d['power_level'], d['beam_idx'], d['allocated'] = state['power_level'], state['beam_idx'], state['allocated']

print(f"Collected {len(all_train_buf)} samples from {len(results_ref)} simulations.")
serve.shutdown()
print("Simulation finished, Ray Serve shut down.")

# --- 2. Training Phase ---
print("\n--- 2. Starting Training Phase ---")
dataset = ray.data.from_items(all_train_buf)

config = {'train_params': train_params, 'model_conf': ac_model_conf, 'graph_list_ref': graph_list_ref,
          'check_point_ref': check_point_ref}

# class PrintResultCallback(Callback):
#     def on_result(self, result, **kwargs):
#         print(f"Epoch: {result['epoch']}/{result['num_epoch']}, Iter: {result['iter']}, "
#               f"Total Loss: {result['total_loss']:.3f}, Actor Loss: {result['actor_loss']:.3f}, "
#               f"Value Loss: {result['value_loss']:.3f}, Entropy: {result['entropy']:.3f}")


run_config = train.RunConfig(name="ModernSilentTrainExperiment", progress_reporter=None)
trainer = TorchTrainer(
    train_loop_per_worker=ppo_train_loop_per_worker,
    train_loop_config=config,
    scaling_config=ScalingConfig(use_gpu=True),
    datasets={"train": dataset},
    run_config=run_config
)
result = trainer.fit()

# final_metrics = result.metrics
# checkpoint_ref = final_metrics['checkpoint_ref']
# checkpoint = ray.get(checkpoint_ref)
# model_state_dict = checkpoint["model_state_dict"]
# optimizer_state_dict = checkpoint["optimizer_state_dict"]
#
# check_point = {'model_state_dict': model_state_dict,
#                'optimizer_state_dict': optimizer_state_dict}
# check_point_ref = ray.put(check_point)





