from abc import ABC, abstractmethod


class NetworkEnvironment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_resource_allocation(self, resource_allocation):
        pass

    @abstractmethod
    def get_resource_allocation(self):
        pass

    def compute_network_state_and_score(self):
        self.compute_network_state()
        self.compute_score()

    @abstractmethod
    def compute_network_state(self):
        pass

    @abstractmethod
    def compute_score(self):
        pass

    @abstractmethod
    def move(self, m):
        pass

    @abstractmethod
    def get_move_mask(self):
        pass

    @abstractmethod
    def get_all_available_moves(self):
        pass

    @abstractmethod
    def get_random_move(self):
        pass

    @abstractmethod
    def is_finished(self):
        pass

    @abstractmethod
    def is_feasible(self):
        pass
