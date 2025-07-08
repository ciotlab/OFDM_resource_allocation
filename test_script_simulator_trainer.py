import logging
from pathlib import Path
import yaml
import torch
import asyncio
import ray
from ray import serve, train
from network.ofdm.ofdm_network_generator import OFDMNetworkGenerator
from network.ofdm.ofdm_network_environment import OFDMNetworkEnvironment
from network.ofdm.ofdm_graph_converter import OFDMGraphConverter
from network.network_actor import get_network_actor
from model.ofdm.ofdm_actor_critic import OFDMActorCritic
from model.model_actor import get_model_actor
from simulator.on_policy_simulator import get_on_policy_simulator
from simulator.inference_server import create_inference_server
from trainer.ppo_trainer import ppo_train_loop_per_worker
from trainer.ofdm.ofdm_dataset import get_ofdm_dataset, get_ofdm_dataloader
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer


# Test simulator
if ray.is_initialized():
    ray.shutdown()
ray.init(logging_level=logging.WARNING)
ray.data.DataContext.get_current().enable_progress_bars = False
ray.data.DataContext.get_current().execution_options.verbose_progress = False
ray.data.DataContext.get_current().print_on_execution_start = False
ray.data.DataContext.get_current().use_ray_tqdm = False


with open(Path(__file__).parents[0] / 'config' / 'ofdm_ppo_config.yaml', 'r') as f:
    conf = yaml.safe_load(f)
network_conf = conf['network']
model_cls = OFDMActorCritic
model_conf = conf['model']
optimizer_cls = torch.optim.Adam
optimizer_conf = {'lr': conf['train']['learning_rate']}
network_actor = get_network_actor(network_gen_cls=OFDMNetworkGenerator, network_env_cls=OFDMNetworkEnvironment,
                                  graph_converter_cls=OFDMGraphConverter, network_conf=network_conf, remote=True)
ray.get(network_actor.generate_networks.remote(num_networks=10))
model_actor = get_model_actor(model_cls=model_cls, model_conf=model_conf, network_conf=network_conf,
                              optimizer_cls=optimizer_cls, optimizer_conf=optimizer_conf,
                              checkpoint_directory_name='ofdm', remote=True)

remote = False
sim = get_on_policy_simulator(actor_id=0, gamma=0.99, lamb=0.9, remote=remote)
if remote:
    serve.start(detached=False, logging_config={"log_level": logging.WARNING})
    inference_server = create_inference_server(max_batch_size=conf['run']['serve_max_batch_size'],
                                               batch_wait_timeout_s=conf['run']['serve_batch_wait_timeout_s'])
    serve_deployment = inference_server.options(num_replicas=conf['run']['serve_num_replicas'],
                                                max_ongoing_requests=conf['run']['serve_max_ongoing_requests'],
                                                ray_actor_options={"num_gpus": 1})
    app = serve_deployment.bind(network_actor=network_actor, model_actor=model_actor)
    model = serve.run(app, logging_config={"log_level": logging.WARNING})
    results_ref = [sim.run.remote(net_id=0, network_actor=network_actor, model=model)]
    simulation_actors = [sim]
    progress = {}
    while not ray.wait(results_ref, timeout=1)[0]:
        for actor in simulation_actors:
            progress.update(ray.get(actor.get_progress.remote()))
        if progress:
            print("Simulation Progress: ", end="")
            for sim_id in range(len(results_ref)):
                if sim_id in progress:
                    print(f"[{sim_id}: {progress[sim_id]:3.0%}]", end="")
            print()
    train_buf = ray.get(results_ref)[0]
    serve.shutdown()
    dataset = get_ofdm_dataset(train_buf)
    config = {'train_conf': conf['train'], 'network_actor': network_actor, 'model_actor': model_actor,
              'get_dataloader': get_ofdm_dataloader, 'parallel': True}
    trainer = TorchTrainer(
        train_loop_per_worker=ppo_train_loop_per_worker,
        train_loop_config=config,
        scaling_config=ScalingConfig(use_gpu=True),
        datasets={"train": dataset},
        run_config=train.RunConfig(name="ModernSilentTrainExperiment", progress_reporter=None)
    )
    trainer.fit()
else:
    model_info = ray.get(model_actor.get_model_info.remote())
    model = model_info['model_cls'](network_conf=model_info['network_conf'], model_conf=model_info['model_conf'])
    train_buf = asyncio.run(sim.run(net_id=0, network_actor=network_actor, model=model))
    dataset = get_ofdm_dataset(train_buf)
    config = {'train_conf': conf['train'], 'network_actor': network_actor, 'model_actor': model_actor,
              'get_dataloader': get_ofdm_dataloader, 'parallel': False, 'dataset': dataset}
    ppo_train_loop_per_worker(config)
pass


