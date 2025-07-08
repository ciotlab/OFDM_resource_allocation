import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import ray
from ray import train
from model.actor_critic import ActionDistribution


def ppo_train_loop_per_worker(config):
    train_conf = config['train_conf']
    network_actor, model_actor = config['network_actor'], config['model_actor']
    # Prepare graph and model
    graph_list = ray.get(network_actor.get_graph_list.remote())
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = train.get_device()
    model_info = ray.get(model_actor.get_model_info.remote())
    model = model_info['model_cls'](network_conf=model_info['network_conf'],
                                    model_conf=model_info['model_conf']).to(device)
    model_state_dict = ray.get(model_actor.get_model_state_dict.remote())
    model.load_state_dict(model_state_dict)
    model.train()
    # Prepare optimizer
    optimizer_info = ray.get(model_actor.get_optimizer_info.remote())
    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = optimizer_info['optimizer_cls'](param_dicts, **optimizer_info['optimizer_conf'])
    optimizer_state_dict = ray.get(model_actor.get_optimizer_state_dict.remote())
    optimizer.load_state_dict(optimizer_state_dict)
    # Parallel or not
    world_rank = 0
    world_size = 1
    if config['parallel']:
        world_rank = train.get_context().get_world_rank()
        world_size = train.get_context().get_world_size()
        #model = train.torch.prepare_model(model)
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=True)
        train_dataset_shard = train.get_dataset_shard("train")
    else:
        train_dataset_shard = config['dataset']
    # Start training
    loss = {}
    get_dataloader = config['get_dataloader']
    for epoch in range(train_conf['num_epochs_per_episode']):
        dataloader = get_dataloader(train_dataset_shard, train_conf['batch_size'], graph_list)
        for iter, (state, policy_mask, action, old_action_log_prob, returns, value) in enumerate(dataloader):
            policy_logit, v = model(state)
            old_action_log_prob = torch.tensor(old_action_log_prob, device=device)
            returns = torch.tensor(returns, device=device)
            value = torch.tensor(value, device=device)
            advantage = returns - value
            action_log_prob = []
            entropy = []
            for a, pl, pm in zip(action, policy_logit, policy_mask):
                # Apply mask on policy
                pm = torch.tensor(pm, device=device)
                pl = torch.where(pm[:, :, :, None], -torch.inf, pl)  # ue, rb, power, beam
                # Get distribution and sample move
                distribution = ActionDistribution(pl)
                al = distribution.log_prob(a)
                action_log_prob.append(al)
                en = distribution.entropy()
                entropy.append(en)
            action_log_prob = torch.stack(action_log_prob)
            entropy = torch.stack(entropy)
            action_prob_ratio = torch.exp(torch.clamp(action_log_prob - old_action_log_prob,
                                                      max=train_conf['act_prob_ratio_exponent_clip']))
            clipped_action_prob_ratio = torch.where(advantage >= 0,
                                                    torch.minimum(action_prob_ratio,
                                                                  torch.tensor(1 + train_conf['ppo_clip'], device=device)),
                                                    torch.maximum(action_prob_ratio,
                                                                  torch.tensor(1 - train_conf['ppo_clip'], device=device)))
            actor_loss = -torch.mean(clipped_action_prob_ratio * advantage)
            entropy_loss = -torch.mean(entropy)
            value_loss = nn.MSELoss()(value, returns)
            total_loss = (actor_loss + train_conf['entropy_loss_weight'] * entropy_loss
                          + train_conf['value_loss_weight'] * value_loss)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_conf['clip_max_norm'])
            optimizer.step()
            # Loss report
            if iter % 10 == 0:
                loss = {'actor_loss': actor_loss.detach().item(), 'entropy': -entropy_loss.detach().item(),
                        'value_loss': value_loss.detach().item(), 'total_loss': total_loss.item()}
                if config['parallel']:
                    all_losses = [{} for _ in range(world_size)]
                    dist.all_gather_object(all_losses, loss)
                    loss = {}
                    for k in all_losses[0]:
                        v = [l[k] for l in all_losses]
                        loss[k] = sum(v) / len(v)
                if world_rank==0:
                    print(f"Epoch: {epoch + 1}/{train_conf['num_epochs_per_episode']}, Iter: {iter}, "
                          f"Total Loss: {loss['total_loss']:.3f}, Actor Loss: {loss['actor_loss']:.3f}, "
                          f"Value Loss: {loss['value_loss']:.3f}, Entropy: {loss['entropy']:.3f}")
    # Send model and optimizer state to model actor
    if world_rank == 0:
        model_state_dict = model.module.to('cpu').state_dict()
        ray.get(model_actor.put_model_state_dict.remote(model_state_dict))
        optimizer_state_dict = optimizer.state_dict()
        for state in optimizer_state_dict['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        ray.get(model_actor.put_optimizer_state_dict.remote(optimizer_state_dict))
        ray.get(model_actor.put_loss.remote(loss))
