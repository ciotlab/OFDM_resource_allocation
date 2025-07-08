import ray


class NetworkActor:
    def __init__(self, network_gen_cls, network_env_cls, graph_converter_cls, network_conf):
        self.network_env_cls = network_env_cls
        self.network_conf = network_conf
        self.network_generator = network_gen_cls(**self.network_conf['generator'])
        self.graph_converter = graph_converter_cls(**self.network_conf['graph'])
        self.network_list = []
        self.graph_list = []

    def generate_networks(self, num_networks):
        self.network_list = self.network_generator.generate_networks(num_networks)
        self.generate_graph()

    def put_networks(self, network_list):
        self.network_list = network_list
        self.generate_graph()

    def load_networks(self, directory_name):
        self.network_list = self.network_generator.load_networks(directory_name=directory_name)
        self.generate_graph()

    def generate_graph(self):
        self.graph_list = []
        for network in self.network_list:
            graph = self.graph_converter.convert(network=network)
            self.graph_list.append(graph)

    def get_num_networks(self):
        return len(self.network_list)

    def get_network(self, id):
        return self.network_list[id]

    def get_networks(self):
        return self.network_list

    def get_graph(self, id):
        return self.graph_list[id]

    def get_graph_list(self):
        return self.graph_list

    def get_env_info(self):
        info = {'env_cls': self.network_env_cls, 'env_conf': self.network_conf['environment']}
        return info


def get_network_actor(network_gen_cls, network_env_cls, graph_converter_cls, network_conf, remote=True):
    if remote:
        actor_cls = ray.remote(NetworkActor)
        return actor_cls.remote(network_gen_cls, network_env_cls, graph_converter_cls, network_conf)
    else:
        return NetworkActor(network_gen_cls, network_env_cls, graph_converter_cls, network_conf)
