import torch
import ray
from pathlib import Path
import yaml
from network.ofdm.ofdm_network_generator import OFDMNetworkGenerator
from network.ofdm.ofdm_network_environment import OFDMNetworkEnvironment
from network.ofdm.ofdm_graph_converter import OFDMGraphConverter
from network.network_actor import get_network_actor
from model.ofdm.ofdm_actor_critic import OFDMActorCritic
from model.model_actor import get_model_actor


# # Model test
# with open(Path(__file__).parents[0] / 'config' / 'ofdm_ppo_config.yaml', 'r') as f:
#     conf = yaml.safe_load(f)
# network_conf = conf['network']
# network_actor = get_network_actor(network_gen_cls=OFDMNetworkGenerator, network_env_cls=OFDMNetworkEnvironment,
#                                   graph_converter_cls=OFDMGraphConverter, network_conf=network_conf, remote=False)
# network_actor.generate_networks(num_networks=10)
# network = network_actor.get_network(id = 0)
# graph = network_actor.get_graph(id = 0)
# env_info = network_actor.get_env_info()
# env = env_info['env_cls'](network=network, **env_info['env_conf'])
# model_conf = conf['model']
# model = OFDMActorCritic(network_conf=network_conf, model_conf=model_conf).to('cuda')
# resource_alloc = env.get_resource_allocation()
# data = {'graph': graph}
# data.update(resource_alloc)
# policy_logit_list, value = model([data])
# pass


# Test model actor
with open(Path(__file__).parents[0] / 'config' / 'ofdm_ppo_config.yaml', 'r') as f:
    conf = yaml.safe_load(f)
network_conf = conf['network']
network_actor = get_network_actor(network_gen_cls=OFDMNetworkGenerator, network_env_cls=OFDMNetworkEnvironment,
                                  graph_converter_cls=OFDMGraphConverter, network_conf=network_conf, remote=False)
model_cls = OFDMActorCritic
model_conf = conf['model']
optimizer_cls = torch.optim.Adam
optimizer_conf = {'lr': conf['train']['learning_rate']}
# # Non-remote model actor
# model_actor = get_model_actor(model_cls=model_cls, model_conf=model_conf, network_conf=network_conf,
#                               optimizer_cls=optimizer_cls, optimizer_conf=optimizer_conf,
#                               checkpoint_directory_name='ofdm', remote=False)
# model_info = model_actor.get_model_info()
# model = model_info['model_cls'](network_conf=model_info['network_conf'], model_conf=model_info['model_conf'])
# model_state_dict = model_actor.get_model_state_dict()
# model.load_state_dict(model_state_dict)
# optimizer_info = model_actor.get_optimizer_info()
# param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
# optimizer = optimizer_info['optimizer_cls'](param_dicts, **optimizer_info['optimizer_conf'])
# optimizer_state_dict = model_actor.get_optimizer_state_dict()
# optimizer.load_state_dict(optimizer_state_dict)
# model_actor.put_model_state_dict(model.state_dict())
# model_actor.put_optimizer_state_dict(optimizer.state_dict())
# model_actor.put_loss(1.0)
# model_actor.put_metric(2.0)
# model_actor.save_checkpoint('model_1')
# model_actor.load_checkpoint('model_1')
# Remote model actor
model_actor = get_model_actor(model_cls=model_cls, model_conf=model_conf, network_conf=network_conf,
                              optimizer_cls=optimizer_cls, optimizer_conf=optimizer_conf,
                              checkpoint_directory_name='ofdm', remote=True)
model_info = ray.get(model_actor.get_model_info.remote())
model = model_info['model_cls'](network_conf=model_info['network_conf'], model_conf=model_info['model_conf'])
model_state_dict = ray.get(model_actor.get_model_state_dict.remote())
model.load_state_dict(model_state_dict)
optimizer_info = ray.get(model_actor.get_optimizer_info.remote())
param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
optimizer = optimizer_info['optimizer_cls'](param_dicts, **optimizer_info['optimizer_conf'])
optimizer_state_dict = ray.get(model_actor.get_optimizer_state_dict.remote())
optimizer.load_state_dict(optimizer_state_dict)
ray.get(model_actor.put_model_state_dict.remote(model.state_dict()))
ray.get(model_actor.put_optimizer_state_dict.remote(optimizer.state_dict()))
ray.get(model_actor.put_loss.remote(1.0))
ray.get(model_actor.put_metric.remote(2.0))
ray.get(model_actor.save_checkpoint.remote('model_1'))
ray.get(model_actor.load_checkpoint.remote('model_1'))
pass


