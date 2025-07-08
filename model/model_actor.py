import dill
import ray
from pathlib import Path


class CheckPoint:
    def __init__(self, model_state_dict, optimizer_state_dict, loss=None, metric=None):
        self.model_state_dict = model_state_dict
        self.optimizer_state_dict = optimizer_state_dict
        self.loss = loss
        self.metric = metric

    def save(self, directory_name, file_name):
        directory_path = Path(__file__).parents[1].resolve() / 'checkpoint' / directory_name
        directory_path.mkdir(parents=True, exist_ok=True)
        file_path = directory_path / file_name
        checkpoint = {'model_state_dict': self.model_state_dict, 'optimizer_state_dict': self.optimizer_state_dict,
                      'loss': self.loss, 'metric': self.metric}
        with open(file_path, "wb") as f:
            dill.dump(checkpoint, f)

    def load(self, directory_name, file_name):
        file_path = Path(__file__).parents[1].resolve() / 'checkpoint' / directory_name / file_name
        with open(file_path, "rb") as f:
            checkpoint = dill.load(f)
        self.model_state_dict = checkpoint['model_state_dict']
        self.optimizer_state_dict = checkpoint['optimizer_state_dict']
        self.loss = checkpoint['loss']
        self.metric = checkpoint['metric']


class ModelActor:
    def __init__(self, model_cls, network_conf, model_conf, optimizer_cls, optimizer_conf, checkpoint_directory_name):
        self.checkpoint_directory_name = checkpoint_directory_name
        self.model_cls = model_cls
        self.network_conf = network_conf
        self.model_conf = model_conf
        self.optimizer_cls = optimizer_cls
        self.optimizer_conf = optimizer_conf
        self.cur_checkpoint = None
        self.init_cur_checkpoint()

    def init_cur_checkpoint(self):
        model = self.model_cls(self.network_conf, self.model_conf)
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
        optimizer = self.optimizer_cls(param_dicts, **self.optimizer_conf)
        self.cur_checkpoint = CheckPoint(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict())

    def save_checkpoint(self, file_name):
        self.cur_checkpoint.save(directory_name=self.checkpoint_directory_name, file_name=file_name)

    def load_checkpoint(self, file_name):
        self.cur_checkpoint.load(directory_name=self.checkpoint_directory_name, file_name=file_name)

    def get_model_state_dict(self):
        return self.cur_checkpoint.model_state_dict

    def put_model_state_dict(self, model_state_dict):
        self.cur_checkpoint.model_state_dict = model_state_dict

    def get_optimizer_state_dict(self):
        return self.cur_checkpoint.optimizer_state_dict

    def put_optimizer_state_dict(self, optimizer_state_dict):
        self.cur_checkpoint.optimizer_state_dict = optimizer_state_dict

    def get_loss(self):
        return self.cur_checkpoint.loss

    def put_loss(self, loss):
        self.cur_checkpoint.loss = loss

    def get_metric(self):
        return self.cur_checkpoint.metric

    def put_metric(self, metric):
        self.cur_checkpoint.metric = metric

    def get_model_info(self):
        info = {'model_cls': self.model_cls, 'network_conf': self.network_conf, 'model_conf': self.model_conf}
        return info

    def get_optimizer_info(self):
        info = {'optimizer_cls': self.optimizer_cls, 'optimizer_conf': self.optimizer_conf}
        return info


def get_model_actor(model_cls, network_conf, model_conf, optimizer_cls, optimizer_conf, checkpoint_directory_name, remote=True):
    if remote:
        actor_cls = ray.remote(ModelActor)
        return actor_cls.remote(model_cls, network_conf, model_conf, optimizer_cls, optimizer_conf, checkpoint_directory_name)
    else:
        return ModelActor(model_cls, network_conf, model_conf, optimizer_cls, optimizer_conf, checkpoint_directory_name)

