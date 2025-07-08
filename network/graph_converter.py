from abc import ABC, abstractmethod


class GraphConverter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def convert(self, network):
        pass
