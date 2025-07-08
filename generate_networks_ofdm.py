import yaml
from pathlib import Path
from network.ofdm.ofdm_network_generator import OFDMNetworkGenerator


# Parameters
config_file = 'ofdm_ppo_config.yaml'
with open(Path(__file__).parents[0] / 'config' / config_file, 'r') as f:
    conf = yaml.safe_load(f)['network']['generator']
data_dir = conf['data_dir']
num_ue_range = conf['num_ue_range']
num_rb = conf['num_rb']
num_beam = conf['num_beam']

# Number of networks
num_networks_validation = 100
num_networks_test = 100

# Generate networks
ng = OFDMNetworkGenerator(data_dir=data_dir, num_ue_range=num_ue_range, num_rb=num_rb, num_beam=num_beam)
ng.generate_and_save_networks(num_networks=num_networks_validation, directory_name='validation')
ng.generate_and_save_networks(num_networks=num_networks_test, directory_name='test')

# # Load and plot test
# loaded_networks = ng.load_networks('validation')
# ng.plot(loaded_networks[0])

