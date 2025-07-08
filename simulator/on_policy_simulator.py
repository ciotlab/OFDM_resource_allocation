import numpy as np
import torch
from model.actor_critic import ActionDistribution
import ray


class OnPolicySimulator:
    def __init__(self, actor_id, gamma, lamb, max_num_steps=None, remote=True):
        self.actor_id = actor_id
        self.gamma = gamma  # Reward discount factor
        self.lamb = lamb  # for computing lambda returns (0: TD1, 1: MC)
        self.max_num_steps = max_num_steps
        self.remote = remote
        self.progress = {}

    def reset(self):
        self.progress = {}

    async def run(self, net_id, network_actor, model):
        network = ray.get(network_actor.get_network.remote(id=net_id))
        env_info = ray.get(network_actor.get_env_info.remote())
        env = env_info['env_cls'](network=network, **env_info['env_conf'])
        prev_score = env.score
        buffer = []
        step = 0
        while not env.is_finished():
            if self.max_num_steps is not None and step >= self.max_num_steps:
                break
            state = env.get_resource_allocation()
            if self.remote:
                data = {'net_id': net_id}  # net_id is used for finding matching graph
                data.update(state)
                result = await model.infer.remote(data)
                policy_logit, value = result['policy_logit'], result['value']
            else:
                graph = ray.get(network_actor.get_graph.remote(id=net_id))
                data = {'graph': graph}
                data.update(state)
                with torch.no_grad():
                    policy_logit, value = model([data])
                policy_logit, value = policy_logit[0].cpu(), value[0].cpu()
            # Apply mask on policy
            policy_mask = torch.tensor(env.get_move_mask(), device=policy_logit.device)  # ue, rb, power
            policy_logit = torch.where(policy_mask[:, :, :, None], -torch.inf, policy_logit)  # ue, rb, power, beam
            # Get distribution and sample move
            dist = ActionDistribution(policy_logit)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            env.move(action)
            # Get reward
            cur_score = env.score
            reward = cur_score - prev_score
            prev_score = cur_score
            # Record
            buf_data = {'net_id': net_id, 'state': state, 'action': action, 'policy_mask': policy_mask.cpu().numpy(),
                        'action_log_prob': action_log_prob.cpu().numpy(), 'value': value.item(),
                        'reward': reward.astype(np.float32).item(),
                        'score': cur_score.astype(np.float32).item()}
            buffer.append(buf_data)
            # Update progress
            step += 1
            m = policy_mask.cpu().numpy()
            progress = (np.sum(m) / m.size).item()
            if self.max_num_steps is not None:
                progress = max(progress, step / self.max_num_steps)
            self.progress[net_id] = progress
        # Calculate lambda returns
        reward = np.array([x['reward'] for x in buffer])
        value = np.array([x['value'] for x in buffer])
        num_step = reward.shape[0]
        ret = np.zeros((num_step,), dtype=np.float32)
        ret[num_step - 1] = reward[num_step - 1]
        for step in range(num_step - 2, -1, -1):
            ret[step] = ((1 - self.lamb) * self.gamma * value[step + 1] + reward[step]
                         + self.lamb * self.gamma * ret[step + 1])
        for i, r in enumerate(ret):
            buffer[i]['returns'] = r.item()
        return buffer

    def get_progress(self):
        return self.progress


def get_on_policy_simulator(actor_id, gamma, lamb, max_num_steps, remote=True):
    if remote:
        actor_cls = ray.remote(OnPolicySimulator)
        return actor_cls.remote(actor_id, gamma, lamb, max_num_steps, remote)
    else:
        return OnPolicySimulator(actor_id, gamma, lamb, max_num_steps, remote)


