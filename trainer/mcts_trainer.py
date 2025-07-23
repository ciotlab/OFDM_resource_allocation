import os
import copy
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import ray
from ray import train

from simulator.mcts_simulator import get_mcts_simulator
from network.ofdm.ofdm_network_environment import OFDMNetworkEnvironment
from model.actor_critic import ActionDistribution

def mcts_train_loop_per_worker(config):
    # load configs
    train_conf = config['train_conf']
    network_actor = config['network_actor']
    model_actor   = config['model_actor']
    parallel      = config['parallel']

    # prepare graph and model
    graph_list = ray.get(network_actor.get_graph_list.remote())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_info = ray.get(model_actor.get_model_info.remote())
    model = model_info['model_cls'](
        network_conf = model_info['network_conf'],
        model_conf   = model_info['model_conf']
    ).to(device)
    model_state = ray.get(model_actor.get_model_state_dict.remote())
    model.load_state_dict(model_state)
    model.train()

    # DDP if needed
    world_rank = 0
    world_size = 1
    if parallel:
        world_rank = train.get_context().get_world_rank()
        world_size = train.get_context().get_world_size()
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True
        )
    # prepare optimizer
    opt_info = ray.get(model_actor.get_optimizer_info.remote())
    params   = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = opt_info['optimizer_cls'](
        [{'params': params}],
        **opt_info['optimizer_conf']
    )
    opt_state = ray.get(model_actor.get_optimizer_state_dict.remote())
    optimizer.load_state_dict(opt_state)

    # prepare simulator
    sim = get_mcts_simulator.remote(
        actor_id    = world_rank,
        num_simulations = train_conf['num_simulations'],
        c_puct      = train_conf['c_puct'],
        max_num_steps = train_conf.get('max_num_steps', None)
    )

    # self-play + train loop
    for episode in range(train_conf['num_episodes']):
        # request one episode of self-play trajectories
        traj = ray.get(sim.run.remote(
            net_id         = world_rank,
            network_actor  = network_actor,
            model          = model_actor
        ))

        # batch update
        # each item: dict with 'state', 'pi', 'return'
        states = torch.stack([torch.tensor(t['state']).float().to(device)
                              for t in traj])
        pis    = torch.stack([torch.tensor(
                           [t['pi'].get(a, 0) for a in range(OFDMNetworkEnvironment().action_size)]
                           ).float().to(device)
                           for t in traj])
        returns = torch.tensor([t['return'] for t in traj])\
                     .float().to(device)

        logits, values = model({"state": states})
        values = values.view(-1)
        # value loss
        loss_v = F.mse_loss(values, returns)
        # policy loss
        logp = F.log_softmax(logits, dim=-1)
        loss_p = -(pis * logp).sum(dim=1).mean()
        loss = loss_v + loss_p

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_conf['clip_max_norm']
        )
        optimizer.step()

        # report & save
        if parallel:
            # gather losses
            losses = {'total_loss': loss.item()}
            all_losses = [{} for _ in range(world_size)]
            dist.all_gather_object(all_losses, losses)
            if world_rank == 0:
                avg_loss = sum(l['total_loss'] for l in all_losses) / world_size
                print(f"[Episode {episode+1}/{train_conf['num_episodes']}] "
                      f"Avg Loss: {avg_loss:.4f}")
        else:
            if episode % train_conf['log_interval'] == 0:
                print(f"[Episode {episode+1}/{train_conf['num_episodes']}] "
                      f"Loss: {loss.item():.4f}")

        # push updated model & optimizer
        if world_rank == 0:
            md = model.module.state_dict() if parallel else model.state_dict()
            ray.get(model_actor.put_model_state_dict.remote(md))
            od = optimizer.state_dict()
            # move tensors to cpu
            for st in od['state'].values():
                for k,v in st.items():
                    if isinstance(v, torch.Tensor):
                        st[k] = v.cpu()
            ray.get(model_actor.put_optimizer_state_dict.remote(od))

        # checkpoint
        if world_rank == 0 and (episode+1) % train_conf['checkpoint_interval'] == 0:
            os.makedirs(train_conf['checkpoint_dir'], exist_ok=True)
            path = os.path.join(train_conf['checkpoint_dir'],
                                f"mcts_checkpoint_{episode+1}.pt")
            torch.save(
                md if world_rank==0 else model.module.state_dict(),
                path
            )
