import numpy as np
import random
import time
import copy
from pathlib import Path
import yaml

from network.ofdm.ofdm_network_generator import OFDMNetworkGenerator
from network.ofdm.ofdm_network_environment import OFDMNetworkEnvironment


class TabuSearch:
    def __init__(self, gen_cls, gen_conf, env_cls, env_conf,
                 tabu_move_selection_prob, tabu_max_iterations, tabu_tenure):
        self.gen_cls = gen_cls
        self.gen_conf = gen_conf
        self.env_cls = env_cls
        self.env_conf = env_conf
        self.tabu_move_selection_prob = tabu_move_selection_prob
        self.tabu_max_iterations = tabu_max_iterations
        self.tabu_tenure = tabu_tenure
        self.loaded_networks = []
        self.num_loaded_networks = 0

    def load_saved_networks(self, directory_name):
        self.loaded_networks = self.gen_cls.load_networks(directory_name)
        self.num_loaded_networks = len(self.loaded_networks)

    def solve_all_loaded_networks(self):
        score_list = []
        elapsed_time_list = []
        for idx in range(self.num_loaded_networks):
            score, elapsed_time, _ = self.solve_loaded_network(idx)
            score_list.append(score)
            elapsed_time_list.append(elapsed_time)
        avg_score = sum(score_list) / len(score_list)
        avg_elapsed_time = sum(elapsed_time_list) / len(elapsed_time_list)
        return avg_score, avg_elapsed_time

    def solve_loaded_network(self, index):
        score, elapsed_time, resource_allocation = self.solve(self.loaded_networks[index])
        return score, elapsed_time, resource_allocation

    def solve(self, network):
        cur_env = env_cls(network, **env_conf)
        start = time.perf_counter()
        tabu_list = {}
        best_resource_allocation, best_score = None, -np.inf
        for it in range(self.tabu_max_iterations):
            all_moves = cur_env.get_all_available_moves()
            if len(all_moves) == 0:
                break
            n_moves = max(int(len(all_moves) * self.tabu_move_selection_prob), 1)
            cand_moves = random.sample(all_moves, n_moves)
            selected_move, selected_score = None, -np.inf
            for cand_move in cand_moves:
                cand_env = copy.deepcopy(cur_env)
                cand_env.move(cand_move)
                cand_score = cand_env.score
                if cand_score > selected_score and ((cand_move not in tabu_list) or cand_score > best_score):
                    selected_move, selected_score = cand_move, cand_score
                if cand_score > best_score:
                    best_resource_allocation, best_score = cand_env.get_resource_allocation(), cand_score
            if selected_move is None:
                break
            cur_env.move(selected_move)
            tabu_list[selected_move] = self.tabu_tenure
            tabu_list = {move:(tenure - 1) for move, tenure in tabu_list.items() if tenure > 1}
            print(f"Iteration: {it}, Score: {best_score}")
        end = time.perf_counter()
        elapsed_time = end - start
        return best_score, elapsed_time, best_resource_allocation


if __name__ == '__main__':
    # Network parameters
    config_file = 'ofdm_ppo_config.yaml'
    with open(Path(__file__).parents[0] / 'config' / config_file, 'r') as f:
        conf = yaml.safe_load(f)
    gen_conf = conf['network']['generator']
    env_conf = conf['network']['environment']
    gen_cls = OFDMNetworkGenerator
    env_cls = OFDMNetworkEnvironment
    # Tabu search parameters
    tabu_move_selection_prob = 0.001
    tabu_max_iterations = 1000
    tabu_tenure = 10
    ts = TabuSearch(gen_cls, gen_conf, env_cls, env_conf, tabu_move_selection_prob, tabu_max_iterations, tabu_tenure)
    ts.load_saved_networks('validation')
    #score, elapsed_time, _ = ts.solve_loaded_network(0)
    score, elapsed_time = ts.solve_all_loaded_networks()
    print(f'Score: {score}, Elapsed time: {elapsed_time}')


