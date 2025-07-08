import logging
from pathlib import Path
import yaml
import torch
import random
import ray
from ray import serve
import wandb
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


class MainOfdmPpoTrainer:
    def __init__(self, config_file, checkpoint_dir):
        # Read config file
        with open(Path(__file__).parents[0] / 'config' / config_file, 'r') as f:
            self.conf = yaml.safe_load(f)
        self.network_conf, self.model_conf, self.run_conf, self.simulation_conf, self.train_conf = (
            self.conf['network'], self.conf['model'], self.conf['run'], self.conf['simulation'], self.conf['train'])
        self.init_ray()
        self.network_actor = get_network_actor(network_gen_cls=OFDMNetworkGenerator, network_env_cls=OFDMNetworkEnvironment,
                                               graph_converter_cls=OFDMGraphConverter, network_conf=self.network_conf,
                                               remote=True)
        optimizer_conf = {'lr': self.train_conf['learning_rate']}
        self.model_actor = get_model_actor(model_cls=OFDMActorCritic, model_conf=self.model_conf, network_conf=self.network_conf,
                                           optimizer_cls=torch.optim.Adam, optimizer_conf=optimizer_conf,
                                           checkpoint_directory_name=checkpoint_dir, remote=True)
        self.simulators = []

    def init_ray(self):
        if ray.is_initialized():
            ray.shutdown()
        ray.init(logging_level=logging.WARNING)
        ray.data.DataContext.get_current().enable_progress_bars = False
        ray.data.DataContext.get_current().execution_options.verbose_progress = False
        ray.data.DataContext.get_current().print_on_execution_start = False
        ray.data.DataContext.get_current().use_ray_tqdm = False

    def run(self, use_wandb=False):
        if use_wandb:
            wandb.init(project='OFDM_PPO_resource_allocation', config=self.conf)
        for episode in range(self.run_conf['num_episodes']):
            print(f"\n--- 1. Starting Simulation Phase (episode: {episode}) ---")
            ray.get(self.network_actor.generate_networks.remote(self.run_conf['num_simulations_per_episode']))
            buffer = self.simulation()
            dataset = self.make_dataset(buffer)

            print(f"\n--- 2. Starting Training Phase (episode: {episode}) ---")
            self.train(dataset)

            print(f"\n--- 3. Starting Evaluation Phase (episode: {episode}) ---")
            ray.get(self.network_actor.load_networks.remote(directory_name='validation'))
            buffer = self.simulation()
            scores = [buf[-1]['score'] for buf in buffer]
            score = sum(scores) / len(scores)
            ray.get(self.model_actor.put_metric.remote({'score': score}))
            print(f"Score: {score:.4f}")
            ray.get(self.model_actor.save_checkpoint.remote(file_name=f'checkpoint_{episode}.pkl'))
            if use_wandb:
                self.wandb_log()
            print(f"----------- Episode {episode} has been finished -----------")

    def simulation(self):
        # Start serv
        serve.start(detached=True, logging_config={"log_level": logging.WARNING})
        inference_server = create_inference_server(max_batch_size=self.run_conf['serve_max_batch_size'],
                                                   batch_wait_timeout_s=self.run_conf['serve_batch_wait_timeout_s'])
        serve_deployment = inference_server.options(num_replicas=self.run_conf['serve_num_replicas'],
                                                    max_ongoing_requests=self.run_conf['serve_max_ongoing_requests'],
                                                    ray_actor_options={"num_gpus": 1})
        app = serve_deployment.bind(network_actor=self.network_actor, model_actor=self.model_actor)
        serve_handle = serve.run(app, logging_config={"log_level": logging.WARNING})
        # Reallocate simulators
        self.simulators = [get_on_policy_simulator(actor_id=id, **self.simulation_conf, remote=True)
                           for id in range(self.run_conf['num_simulator'])]
        # Schedule networks
        num_simulators = len(self.simulators)
        num_networks = ray.get(self.network_actor.get_num_networks.remote())
        base = num_networks // num_simulators
        remainder = num_networks % num_simulators
        assignments = []
        for actor_id in range(num_simulators):
            ray.get(self.simulators[actor_id].reset.remote())
            count = base + (1 if actor_id < remainder else 0)
            assignments.extend([actor_id] * count)
        results_ref = []
        for net_id in range(num_networks):
            actor_id = assignments[net_id]
            results_ref.append(self.simulators[actor_id].run.remote(net_id=net_id, network_actor=self.network_actor, model=serve_handle))
        progress = {}
        while not ray.wait(results_ref, timeout=1)[0]:
            for sim in self.simulators:
                progress.update(ray.get(sim.get_progress.remote()))
            if progress:
                print("Simulation Progress: ", end="")
                for sim_id in range(len(results_ref)):
                    if sim_id in progress:
                        print(f"[{sim_id}: {progress[sim_id]:3.0%}]", end="")
                print()
        buffer = []
        for buf in ray.get(results_ref):
            buffer.append(buf)
        serve.shutdown()
        print("Simulation finished, Ray Serve shut down.")
        return buffer

    @staticmethod
    def make_dataset(buffer):
        data = []
        for buf in buffer:
            data.extend(buf)
        random.shuffle(data)
        dataset = get_ofdm_dataset(data)
        return dataset

    def train(self, dataset):
        config = {'train_conf': self.train_conf, 'network_actor': self.network_actor, 'model_actor': self.model_actor,
                  'get_dataloader': get_ofdm_dataloader, 'parallel': True}
        trainer = TorchTrainer(
            train_loop_per_worker=ppo_train_loop_per_worker,
            train_loop_config=config,
            scaling_config=ScalingConfig(num_workers=self.run_conf['num_training_workers'], use_gpu=True),
            datasets={"train": dataset}
        )
        trainer.fit()
        print("Training finished.")

    def wandb_log(self):
        loss = ray.get(self.model_actor.get_loss.remote())
        metric = ray.get(self.model_actor.get_metric.remote())
        wandb_log = {'score': metric['score'], 'total_loss': loss['total_loss'],
                     'actor_loss': loss['actor_loss'], 'value_loss': loss['value_loss'],
                     'entropy': loss['entropy']}
        wandb.log(wandb_log)


if __name__ == "__main__":
    trainer = MainOfdmPpoTrainer(config_file='ofdm_ppo_config.yaml', checkpoint_dir='ofdm_ppo')
    trainer.run(use_wandb=True)





