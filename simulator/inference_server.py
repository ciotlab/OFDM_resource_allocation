import torch
import ray
from ray import serve


def create_inference_server(max_batch_size, batch_wait_timeout_s):
    @serve.deployment
    class InferenceServer:
        def __init__(self, network_actor, model_actor):
            self.graph_list = ray.get(network_actor.get_graph_list.remote())
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_info = ray.get(model_actor.get_model_info.remote())
            self.model = model_info['model_cls'](network_conf=model_info['network_conf'],
                                                 model_conf=model_info['model_conf'])
            model_state_dict = ray.get(model_actor.get_model_state_dict.remote())
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()

        @serve.batch(max_batch_size=max_batch_size, batch_wait_timeout_s=batch_wait_timeout_s)
        async def infer(self, data):
            for d in data:
                net_id = d.pop('net_id')
                graph = self.graph_list[net_id]
                d['graph'] = graph
            with torch.no_grad():
                policy_logit_list, value = self.model(data)
            out = [{'policy_logit': policy_logit.cpu(), 'value': v.cpu()} for (policy_logit, v) in zip(policy_logit_list, value)]
            return out
    return InferenceServer
