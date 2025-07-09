import numpy as np
import ray
from pathlib import Path
import yaml
from network.ofdm.ofdm_network_generator import OFDMNetworkGenerator
from network.ofdm.ofdm_network_environment import OFDMNetworkEnvironment
from network.ofdm.ofdm_graph_converter import OFDMGraphConverter
from network.network_actor import get_network_actor
import logging


# # Test incremental move operation of OFDM network environment
# ng = OFDMNetworkGenerator(data_dir='myeongdong_arr_4_rb_16', num_ue_range=[50, 100], num_rb=12, num_beam=4)
# network = ng.generate_network()
# max_tx_power = 5
# num_tx_power_level = 17
# max_bs_power = 30
# env = OFDMNetworkEnvironment(network, max_tx_power, num_tx_power_level, max_bs_power, noise_spectral_density=-174.0,
#                              alpha=0.0, allow_reallocation=False)
# for step in range(10000):
#     m = env.get_random_move()
#     if m is None:
#         break
#     env.move(m)
#     if (step + 1) % 100 == 0:
#         prev_bs_total_power = env.bs_total_power
#         prev_tx_power = env.tx_power
#         prev_rx_power = env.rx_power
#         prev_interference = env.interference
#         prev_power_mask = env.power_mask
#         prev_score = env.score
#         env.compute_network_state_and_score()
#         print(f"bs_total_power_diff: {np.mean(np.square(env.bs_total_power - prev_bs_total_power)) / np.mean(np.square(env.bs_total_power))}")
#         print(f"tx_power_diff: {np.mean(np.square(env.tx_power - prev_tx_power)) / np.mean(np.square(env.tx_power))}")
#         print(f"rx_power_diff: {np.mean(np.square(env.rx_power - prev_rx_power)) / np.mean(np.square(env.rx_power))}")
#         print(f"interference_diff: {np.mean(np.square(env.interference - prev_interference)) / np.mean(np.square(env.interference))}")
#         print(f"score_diff: {np.square(env.score - prev_score) / np.square(env.score)}")
#         power_mask_diff = np.sum(env.power_mask != prev_power_mask)
#         print(f"power_mask_diff: {power_mask_diff}")
#         print(f"score: {env.score}")
#         print()


# # Graph converter test
# ng = OFDMNetworkGenerator(data_dir='myeongdong_arr_4_rb_16', num_ue_range=[50, 100], num_rb=12, num_beam=4)
# network = ng.generate_network()
# gc = OFDMGraphConverter(min_attn_db=-200, max_attn_db=-50, num_power_attn_level=10, prune_power_attn_thresh=-300)
# graph = gc.convert(network)
# pass


# # Network actor test
# with open(Path(__file__).parents[0] / 'config' / 'ofdm_ppo_config.yaml', 'r') as f:
#     conf = yaml.safe_load(f)
# network_conf = conf['network']
# na = get_network_actor(network_gen_cls=OFDMNetworkGenerator, network_env_cls=OFDMNetworkEnvironment,
#                        graph_converter_cls=OFDMGraphConverter, network_conf=network_conf, remote=False)
# na.generate_networks(num_networks=10)
# na.load_networks(directory_name='validation')
# networks = na.get_networks()
# network = na.get_network(id=3)
# graph_list = na.get_graph_list()
# graph = na.get_graph(id=3)
# env_info = na.get_env_info()


# Network actor remote test
# with open(Path(__file__).parents[0] / 'config' / 'ofdm_ppo_config.yaml', 'r') as f:
#     conf = yaml.safe_load(f)
# network_conf = conf['network']
# na = get_network_actor(network_gen_cls=OFDMNetworkGenerator, network_env_cls=OFDMNetworkEnvironment,
#                        graph_converter_cls=OFDMGraphConverter, network_conf=network_conf, remote=True)
# ray.get(na.generate_networks.remote(num_networks=40))
# ray.get(na.load_networks.remote(directory_name='validation'))
# networks = ray.get(na.get_networks.remote())
# network = ray.get(na.get_network.remote(id=3))
# graph_list = ray.get(na.get_graph_list.remote())
# graph = ray.get(na.get_graph.remote(id=3))
# env_info = ray.get(na.get_env_info.remote())
# ray.kill(na)
# pass
def main():
    """
    Validation 및 Test 데이터셋을 생성하고 저장하는 메인 함수
    """
    # --- 1. Ray 초기화 ---
    # 스크립트를 여러 번 실행해도 문제없도록, 기존 Ray 인스턴스를 종료하고 새로 시작합니다.
    if ray.is_initialized():
        ray.shutdown()
    ray.init(logging_level=logging.WARNING)
    print("Ray가 성공적으로 초기화되었습니다.")

    # --- 2. 설정 파일 로드 ---
    # config 파일에서 네트워크 생성에 필요한 설정을 가져옵니다.
    try:
        with open(Path(__file__).parents[0] / 'config' / 'ofdm_ppo_config.yaml', 'r') as f:
            conf = yaml.safe_load(f)
        network_conf = conf['network']
        print("설정 파일(ofdm_ppo_config.yaml)을 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print("오류: config/ofdm_ppo_config.yaml 파일을 찾을 수 없습니다.")
        return

    # --- 3. 네트워크 액터(Network Actor) 생성 ---
    # 데이터셋 생성, 변환, 관리 등 모든 작업을 처리할 원격 액터를 생성합니다.
    # remote=True로 설정하여 별도의 프로세스에서 동작하도록 합니다.
    print("네트워크 액터를 생성 중입니다...")
    network_actor = get_network_actor(
        network_gen_cls=OFDMNetworkGenerator,
        network_env_cls=OFDMNetworkEnvironment,
        graph_converter_cls=OFDMGraphConverter,
        network_conf=network_conf,
        remote=True
    )
    print("네트워크 액터가 성공적으로 생성되었습니다.")

    # --- 4. 데이터셋 생성 및 저장 ---
    num_validation_networks = 40  # 생성할 검증용 데이터셋 개수
    num_test_networks = 40       # 생성할 테스트용 데이터셋 개수

    # Validation 데이터셋 생성
    print(f"\n--- 검증용(validation) 데이터셋 {num_validation_networks}개 생성 시작 ---")
    validation_task = network_actor.generate_and_save_networks.remote(
        num_networks=num_validation_networks,
        directory_name='validation'
    )
    ray.get(validation_task) # 작업이 끝날 때까지 기다립니다.
    print(f"검증용 데이터셋 생성이 완료되었습니다. 'validation' 폴더를 확인하세요.")

    # Test 데이터셋 생성
    print(f"\n--- 테스트용(test) 데이터셋 {num_test_networks}개 생성 시작 ---")
    test_task = network_actor.generate_and_save_networks.remote(
        num_networks=num_test_networks,
        directory_name='test'
    )
    ray.get(test_task) # 작업이 끝날 때까지 기다립니다.
    print(f"테스트용 데이터셋 생성이 완료되었습니다. 'test' 폴더를 확인하세요.")

    # --- 5. 리소스 정리 ---
    # 사용이 끝난 액터를 종료하고 Ray 인스턴스를 깔끔하게 정리합니다.
    print("\n작업 완료. Ray 리소스를 정리합니다.")
    ray.kill(network_actor)
    ray.shutdown()
    print("모든 작업이 성공적으로 종료되었습니다.")


if __name__ == '__main__':
    main()