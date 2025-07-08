import ray
import numpy as np


def get_ofdm_dataset(buffer):
    for d in buffer:
        state = d.pop('state')
        d['power_level'], d['beam_idx'], d['allocated'] = state['power_level'], state['beam_idx'], state['allocated']
    dataset = ray.data.from_items(buffer)
    return dataset


def get_ofdm_dataloader(dataset, batch_size, graph_list):
    state, policy_mask, action, action_log_prob, returns, value  = [], [], [], [], [], []
    for d in dataset.iter_rows():
        graph = graph_list[d['net_id']]
        s = {'graph': graph, 'power_level': d['power_level'], 'beam_idx': d['beam_idx'], 'allocated': d['allocated']}
        state.append(s)
        policy_mask.append(d['policy_mask'])
        action.append(d['action'])
        action_log_prob.append(d['action_log_prob'])
        returns.append(d['returns'])
        value.append(d['value'])
        if len(state) == batch_size:
            action_log_prob, returns, value = np.stack(action_log_prob), np.stack(returns), np.stack(value)
            yield state, policy_mask, action, action_log_prob, returns, value
            state, policy_mask, action, action_log_prob, returns, value  = [], [], [], [], [], []
    if state:
        action_log_prob, returns, value = np.stack(action_log_prob), np.stack(returns), np.stack(value)
        yield state, policy_mask, action, action_log_prob, returns, value
