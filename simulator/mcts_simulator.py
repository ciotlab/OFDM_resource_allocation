# simulator/mcts_simulator.py

import copy
import numpy as np
import torch
import ray
from collections import defaultdict
from model.actor_critic import ActionDistribution
from network.ofdm.ofdm_network_environment import OFDMNetworkEnvironment

class MCTSNode:
    """State-only MCTS node."""
    def __init__(self, state, prior=1.0):
        self.state = state              # 환경 관측값
        self.prior = prior              # prior probability P(s,a)
        self.visit_count = 0            # N(s,a)
        self.value_sum = 0.0            # W(s,a)
        self.children = {}              # action -> MCTSNode

    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

class MCTSSimulator:
    def __init__(self, actor_id, num_simulations, cpuct, gamma, lamb, remote=True):
        self.actor_id = actor_id
        self.num_simulations = num_simulations
        self.c_puct = cpuct
        self.gamma = gamma  # Reward discount factor
        self.lamb = lamb  # for computing lambda returns (0: TD1, 1: MC)
        self.remote = remote
        self.progress = {}

    def reset(self):
        self.progress.clear()

    async def run(self, net_id, network_actor, model):
        # 1) 환경 및 네트워크 준비
        network = ray.get(network_actor.get_network.remote(id=net_id))
        env_info = ray.get(network_actor.get_env_info.remote())
        env = env_info['env_cls'](network=network, **env_info['env_conf'])

        buffer = []
        step = 0
        prev_score = env.score

        # 2) self-play 에피소드
        while not env.is_finished():
            # 2.1) 루트 상태에서 prior 계산
            root_state = env.get_resource_allocation()
            data = {'net_id': net_id}
            data.update(root_state)
            result = await model.infer.remote(data)
            policy_logit, value = result['policy_logit'], result['value']
            policy_mask = torch.tensor(env.get_move_mask(), device=policy_logit.device)  # ue, rb, power
            policy_logit = torch.where(policy_mask[:, :, :, None], -torch.inf, policy_logit)  # ue, rb, power, beam
            dist = ActionDistribution(policy_logit)
            priors = dist.probs().cpu().numpy()
            root = MCTSNode(state=root_state)
            for a, p in enumerate(priors):
                root.children[a] = MCTSNode(state=None, prior=p.item())

            # 2.2) MCTS 시뮬레이션
            for _ in range(self.num_simulations):
                node = root
                path = []
                reward = 0.0

                # Selection & Expansion
                while True:
                    total_N = sum(child.visit_count for child in node.children.values())
                    best_score, best_a = -np.inf, None
                    for a, child in node.children.items():
                        u = self.c_puct * child.prior * np.sqrt(total_N) / (1 + child.visit_count)
                        score = child.q_value() + u
                        if score > best_score:
                            best_score, best_a = score, a

                    if best_a is None:
                        break
                    path.append((node, best_a))
                    child = node.children[best_a]

                    if child.state is None:
                        # Expansion: 경로 재현하여 새로운 상태 계산
                        sim_env = copy.deepcopy(env)
                        for parent, act in path:
                            action = np.unravel_index(act, policy_logit.shape)
                            sim_env.move(action)
                        child.state = sim_env.get_resource_allocation()

                        # leaf prior & value 계산
                        data = {'net_id': net_id}
                        data.update(child.state)
                        res = await model.infer.remote(data)
                        logits_leaf, v_leaf = res['policy_logit'], res['value']
                        policy_mask = torch.tensor(env.get_move_mask(), device=logits_leaf.device)  # ue, rb, power
                        logits_leaf = torch.where(policy_mask[:, :, :, None], -torch.inf, logits_leaf)  # ue, rb, power, beam
                        # Get distribution and sample move
                        leaf_dist = ActionDistribution(logits_leaf)
                        leaf_priors = leaf_dist.probs().cpu().numpy()
                        for a2, p2 in enumerate(leaf_priors):
                            if a2 not in child.children:
                                child.children[a2] = MCTSNode(state=None, prior=float(p2))
                        reward = float(v_leaf.item())
                        break
                    else:
                        # 이미 확장된 노드
                        node = child

                # Backup
                for parent, a in reversed(path):
                    cnode = parent.children[a]
                    cnode.visit_count += 1
                    cnode.value_sum += reward

            # 2.3) 정책 π 생성 및 행동 선택
            visit_counts = {a: child.visit_count for a, child in root.children.items()}
            total_visits = sum(visit_counts.values())
            if total_visits == 0:
                pi = {a: 1.0 / len(root.children) for a in root.children}
            else:
                pi = {a: cnt / total_visits for a, cnt in visit_counts.items()}

            actions, probs = zip(*pi.items())
            action_idx = torch.multinomial(torch.tensor(probs), 1).item()
            action = actions[action_idx]
            action = np.unravel_index(action, policy_logit.shape)
            # 2.4) 환경 적용 및 보상 계산
            state = env.get_resource_allocation()
            env.move(action)
            cur_score = env.score
            reward = float(cur_score - prev_score)
            prev_score = cur_score

            # 2.5) 트랜지션 기록
            buffer.append({
                'net_id': net_id,
                'state': state,
                'action': action,
                'pi': pi,
                'reward': reward,
                'score': cur_score
            })

            # 2.6) 진행도 업데이트
            step += 1
            mask = np.array(env.get_move_mask())
            prog = mask.sum() / mask.size
            self.progress[net_id] = prog

        # 3) λ-리턴 계산
        # buffer: [{'reward': r0, 'value': v0, ...}, {'reward': r1, 'value': v1, ...}, …]
        rewards = np.array([entry['reward'] for entry in buffer], dtype=np.float32)
        values  = np.array([entry['value']  for entry in buffer], dtype=np.float32)
        T = len(rewards)

        # 뒤에서부터 GAE 계산
        gae = 0.0
        returns = np.zeros_like(rewards)
        for t in reversed(range(T)):
            # δ_t = r_t + γ·V_{t+1} − V_t
            next_value = values[t+1] if t+1 < T else 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lamb * gae
            returns[t] = gae + values[t]   # GAE + V_t = λ-리턴

        # buffer에 삽입
        for i, entry in enumerate(buffer):
            entry['returns'] = float(returns[i])

        return buffer

    def get_progress(self):
        return self.progress

def get_mcts_simulator(actor_id, num_simulations, cpuct, gamma, lamb, remote=True):
    """Ray 원격 액터 또는 로컬 인스턴스 반환."""
    if remote:
        return ray.remote(MCTSSimulator).remote(actor_id, num_simulations, cpuct, gamma, lamb, remote)
    return MCTSSimulator(actor_id, num_simulations, cpuct, gamma, lamb, remote)
