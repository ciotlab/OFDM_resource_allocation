import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
import dill


class NetworkGenerator(ABC):
    def __init__(self):
        pass

    def generate_networks(self, num_networks):
        networks = []
        for _ in range(num_networks):
            networks.append(self.generate_network())
        return networks

    @abstractmethod
    def generate_network(self):
        pass

    def generate_and_save_networks(self, num_networks, directory_name):
        networks = self.generate_networks(num_networks)
        self.save_networks(networks=networks, directory_name=directory_name)
        return networks

    @classmethod
    def save_networks(cls, networks, directory_name):
        module = sys.modules[cls.__module__]
        directory = Path(module.__file__).parents[0].resolve() / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        for count, network in enumerate(networks):
            file_name = directory / (str(count) + '.pkl')
            with open(file_name, "wb") as f:
                dill.dump(network, f)

    @classmethod
    def load_networks(cls, directory_name):
        module = sys.modules[cls.__module__]
        directory = Path(module.__file__).parents[0].resolve() / directory_name
        file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]
        file_list.sort()
        networks = []
        for file in file_list:
            with open(file, 'rb') as f:
                networks.append(dill.load(f))
        return networks

    @abstractmethod
    def plot(self, network):
        pass



if __name__ == '__main__':
    ng = NetworkGenerator(data_dir='myeongdong_arr_4_rb_16', num_ue_range=[50, 100], num_rb=12)
    ng.generate_and_save_networks(num_networks=100, directory_name='validation')
    ng.generate_and_save_networks(num_networks=100, directory_name='test')
    loaded_networks = ng.load_networks('validation')
    # ng.plot(loaded_networks[0])