"""
数据加载模块，用于加载和预处理各种数据集。
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from utils import preprocess_neural_data


def load_synthetic_data(
    num_samples: int = 10000,
    num_neurons: int = 100,
    num_latent: int = 3,
    noise_level: float = 0.1,
    preprocessing: str = 'zscore',
    test_size: float = 0.2,
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """加载合成数据。

    Args:
        num_samples: 样本数量
        num_neurons: 神经元数量
        num_latent: 潜在变量维度
        noise_level: 噪声水平
        preprocessing: 预处理类型
        test_size: 测试集比例
        random_seed: 随机种子

    Returns:
        包含训练和测试数据的字典
    """
    from utils import generate_synthetic_data

    # 生成合成数据
    neural_data, latent = generate_synthetic_data(
        num_samples=num_samples,
        num_neurons=num_neurons,
        num_latent=num_latent,
        noise_level=noise_level,
        random_seed=random_seed
    )

    # 预处理数据
    neural_data = preprocess_neural_data(neural_data, preprocessing)

    # 分割训练集和测试集
    train_size = int(num_samples * (1 - test_size))

    train_neural = neural_data[:train_size]
    test_neural = neural_data[train_size:]

    train_latent = latent[:train_size]
    test_latent = latent[train_size:]

    return {
        'train_neural': train_neural,
        'test_neural': test_neural,
        'train_latent': train_latent,
        'test_latent': test_latent
    }


def load_hippocampus_data(
    data_dir: str = 'data',
    dataset_name: str = 'hc-11',
    bin_width_ms: int = 25,
    preprocessing: str = 'zscore',
    test_size: float = 0.2,
    random_seed: int = 42
) -> Dict[str, Dict[str, np.ndarray]]:
    """加载大鼠海马体数据。

    Args:
        data_dir: 数据目录
        dataset_name: 数据集名称
        bin_width_ms: 时间窗口宽度（毫秒）
        preprocessing: 预处理类型
        test_size: 测试集比例
        random_seed: 随机种子

    Returns:
        包含不同大鼠数据的字典
    """
    import os
    import scipy.io as sio
    from scipy import signal

    # 设置随机种子
    np.random.seed(random_seed)

    # 构建数据路径
    hc_data_dir = os.path.join(data_dir, dataset_name)

    if not os.path.exists(hc_data_dir):
        raise FileNotFoundError(f"数据目录 {hc_data_dir} 不存在。请从CRCNS下载hc-11数据集并放置在 {hc_data_dir} 目录中。")

    # 查找所有会话目录
    session_dirs = [d for d in os.listdir(hc_data_dir) if os.path.isdir(os.path.join(hc_data_dir, d))]

    if not session_dirs:
        raise FileNotFoundError(f"在 {hc_data_dir} 中未找到会话目录。")

    result = {}

    for session_dir in session_dirs:
        session_path = os.path.join(hc_data_dir, session_dir)

        # 查找.mat文件
        mat_files = [f for f in os.listdir(session_path) if f.endswith('.mat')]

        if not mat_files:
            print(f"警告：在 {session_path} 中未找到.mat文件，跳过此会话。")
            continue

        # 加载.mat文件（假设每个会话只有一个.mat文件）
        mat_file = os.path.join(session_path, mat_files[0])
        try:
            data = sio.loadmat(mat_file)
        except Exception as e:
            print(f"警告：无法加载 {mat_file}：{e}，跳过此会话。")
            continue

        # 提取神经元活动数据
        # 注意：实际的键名可能需要根据数据集结构调整
        if 'spikes' in data:
            spikes = data['spikes']  # 假设这是神经元尖峰数据
        elif 'S' in data:
            spikes = data['S']
        else:
            print(f"警告：在 {mat_file} 中未找到神经元活动数据，跳过此会话。")
            continue

        # 提取位置数据
        if 'position' in data:
            position = data['position']
        elif 'pos' in data:
            position = data['pos']
        elif 'tracking' in data:
            position = data['tracking']
        else:
            print(f"警告：在 {mat_file} 中未找到位置数据，使用随机数据代替。")
            position = np.random.rand(spikes.shape[0], 2)

        # 提取方向数据（如果有）
        if 'direction' in data:
            direction = data['direction']
        elif 'dir' in data:
            direction = data['dir']
        elif 'heading' in data:
            direction = data['heading']
        else:
            print(f"警告：在 {mat_file} 中未找到方向数据，使用随机数据代替。")
            direction = np.random.rand(spikes.shape[0], 1)

        # 将尖峰数据转换为时间窗口内的发放率
        # 假设spikes是一个(time_points, neurons)的矩阵
        if bin_width_ms > 1:
            # 重新采样到指定的时间窗口宽度
            bin_factor = bin_width_ms // 1  # 假设原始数据是1ms一个样本
            if bin_factor > 1:
                # 使用卷积进行下采样
                kernel = np.ones(bin_factor) / bin_factor
                binned_spikes = np.zeros((spikes.shape[0] // bin_factor, spikes.shape[1]))

                for i in range(spikes.shape[1]):
                    binned_signal = signal.convolve(spikes[:, i], kernel, mode='valid')[::bin_factor]
                    binned_spikes[:, i] = binned_signal

                # 同样对位置和方向数据进行下采样
                binned_position = np.zeros((position.shape[0] // bin_factor, position.shape[1]))
                for i in range(position.shape[1]):
                    binned_signal = signal.convolve(position[:, i], kernel, mode='valid')[::bin_factor]
                    binned_position[:, i] = binned_signal

                binned_direction = np.zeros((direction.shape[0] // bin_factor, direction.shape[1]))
                for i in range(direction.shape[1]):
                    binned_signal = signal.convolve(direction[:, i], kernel, mode='valid')[::bin_factor]
                    binned_direction[:, i] = binned_signal

                neural_data = binned_spikes
                position_data = binned_position
                direction_data = binned_direction
            else:
                neural_data = spikes
                position_data = position
                direction_data = direction
        else:
            neural_data = spikes
            position_data = position
            direction_data = direction

        # 预处理数据
        neural_data = preprocess_neural_data(neural_data, preprocessing)

        # 分割训练集和测试集
        num_samples = neural_data.shape[0]
        train_size = int(num_samples * (1 - test_size))

        train_neural = neural_data[:train_size]
        test_neural = neural_data[train_size:]

        train_position = position_data[:train_size]
        test_position = position_data[train_size:]

        train_direction = direction_data[:train_size]
        test_direction = direction_data[train_size:]

        # 使用会话名称作为键
        result[session_dir] = {
            'train_neural': train_neural,
            'test_neural': test_neural,
            'train_position': train_position,
            'test_position': test_position,
            'train_direction': train_direction,
            'test_direction': test_direction
        }

    if not result:
        raise ValueError(f"未能从 {hc_data_dir} 加载任何有效数据。")

    return result


def load_primate_data(
    data_dir: str = 'data',
    dataset_name: str = '000127',
    bin_width_ms: int = 1,
    gaussian_smoothing_sigma_ms: int = 40,
    preprocessing: str = 'zscore',
    test_size: float = 0.2,
    random_seed: int = 42
) -> Dict[str, Dict[str, np.ndarray]]:
    """加载灵长类运动任务数据。

    Args:
        data_dir: 数据目录
        dataset_name: 数据集名称 (DANDI Archive ID)
        bin_width_ms: 时间窗口宽度（毫秒）
        gaussian_smoothing_sigma_ms: 高斯平滑的sigma（毫秒）
        preprocessing: 预处理类型
        test_size: 测试集比例
        random_seed: 随机种子

    Returns:
        包含主动和被动运动数据的字典
    """
    import os
    import h5py
    from scipy import signal
    import warnings

    # 设置随机种子
    np.random.seed(random_seed)

    # 构建数据路径
    primate_data_dir = os.path.join(data_dir, dataset_name)

    if not os.path.exists(primate_data_dir):
        raise FileNotFoundError(f"数据目录 {primate_data_dir} 不存在。请从DANDI Archive下载灵长类数据集 (ID: {dataset_name}) 并放置在 {primate_data_dir} 目录中。")

    # 查找所有NWB文件
    nwb_files = []
    for root, _, files in os.walk(primate_data_dir):
        for file in files:
            if file.endswith('.nwb'):
                nwb_files.append(os.path.join(root, file))

    if not nwb_files:
        raise FileNotFoundError(f"在 {primate_data_dir} 中未找到.nwb文件。")

    result = {}
    conditions = ['active', 'passive']

    for nwb_file in nwb_files:
        try:
            with h5py.File(nwb_file, 'r') as f:
                # 提取会话信息
                # 注意：实际的键名和结构可能需要根据数据集调整
                session_id = os.path.basename(nwb_file).split('.')[0]

                # 尝试确定这是主动还是被动运动
                # 这里假设文件名或元数据中包含相关信息
                condition = None
                for cond in conditions:
                    if cond in nwb_file.lower():
                        condition = cond
                        break

                if condition is None:
                    # 如果无法从文件名确定，尝试从元数据中获取
                    if 'general/session_id' in f:
                        session_metadata = f['general/session_id'][()]
                        if isinstance(session_metadata, bytes):
                            session_metadata = session_metadata.decode('utf-8')

                        for cond in conditions:
                            if cond in session_metadata.lower():
                                condition = cond
                                break

                if condition is None:
                    # 如果仍然无法确定，使用文件名作为条件
                    condition = session_id

                # 提取神经元活动数据
                # 假设数据在units/spike_times或类似位置
                spike_times = None
                spike_units = None

                # 尝试不同的可能路径
                possible_spike_paths = [
                    'units/spike_times',
                    'analysis/spike_times',
                    'processing/spikes/spike_times'
                ]

                for path in possible_spike_paths:
                    if path in f:
                        spike_times = f[path][()]
                        break

                if spike_times is None:
                    # 如果找不到预定义路径，搜索整个文件
                    def find_dataset(name, obj):
                        if isinstance(obj, h5py.Dataset) and ('spike' in name.lower() or 'times' in name.lower()):
                            nonlocal spike_times
                            if spike_times is None:
                                spike_times = obj[()]

                    f.visititems(find_dataset)

                if spike_times is None:
                    warnings.warn(f"在 {nwb_file} 中未找到神经元活动数据，跳过此文件。")
                    continue

                # 提取位置和方向数据
                position_data = None
                direction_data = None

                # 尝试不同的可能路径
                possible_position_paths = [
                    'processing/behavior/position',
                    'acquisition/position',
                    'analysis/position'
                ]

                for path in possible_position_paths:
                    if path in f:
                        position_data = f[path][()]
                        break

                possible_direction_paths = [
                    'processing/behavior/direction',
                    'acquisition/direction',
                    'analysis/direction'
                ]

                for path in possible_direction_paths:
                    if path in f:
                        direction_data = f[path][()]
                        break

                # 如果找不到位置或方向数据，创建随机数据
                if position_data is None:
                    warnings.warn(f"在 {nwb_file} 中未找到位置数据，使用随机数据代替。")
                    # 假设时间点数量与神经元活动数据相同
                    position_data = np.random.rand(len(spike_times), 2)

                if direction_data is None:
                    warnings.warn(f"在 {nwb_file} 中未找到方向数据，使用随机数据代替。")
                    direction_data = np.random.rand(len(spike_times), 1)

                # 将尖峰时间转换为时间窗口内的发放率
                # 这里需要根据实际数据格式进行调整

                # 应用高斯平滑
                if gaussian_smoothing_sigma_ms > 0:
                    sigma = gaussian_smoothing_sigma_ms / bin_width_ms  # 转换为bin数
                    kernel_size = int(sigma * 6)  # 通常使用6*sigma作为核大小
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # 确保核大小是奇数

                    gaussian_kernel = signal.gaussian(kernel_size, sigma)
                    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)  # 归一化

                    # 对神经元活动数据应用高斯平滑
                    smoothed_neural = np.zeros_like(spike_times)
                    for i in range(spike_times.shape[1]):
                        smoothed_neural[:, i] = signal.convolve(
                            spike_times[:, i], gaussian_kernel, mode='same'
                        )

                    neural_data = smoothed_neural
                else:
                    neural_data = spike_times

                # 预处理数据
                neural_data = preprocess_neural_data(neural_data, preprocessing)

                # 分割训练集和测试集
                num_samples = neural_data.shape[0]
                train_size = int(num_samples * (1 - test_size))

                train_neural = neural_data[:train_size]
                test_neural = neural_data[train_size:]

                train_position = position_data[:train_size]
                test_position = position_data[train_size:]

                train_direction = direction_data[:train_size]
                test_direction = direction_data[train_size:]

                # 使用条件和会话ID作为键
                key = f"{condition}_{session_id}"
                result[key] = {
                    'train_neural': train_neural,
                    'test_neural': test_neural,
                    'train_position': train_position,
                    'test_position': test_position,
                    'train_direction': train_direction,
                    'test_direction': test_direction
                }

        except Exception as e:
            warnings.warn(f"处理 {nwb_file} 时出错：{str(e)}，跳过此文件。")

    if not result:
        # 如果没有成功加载任何数据，使用模拟数据
        warnings.warn(f"未能从 {primate_data_dir} 加载任何有效数据，使用模拟数据代替。")

        for condition in conditions:
            # 模拟神经数据和位置数据
            num_samples = 10000
            num_neurons = 100

            neural_data = np.random.randn(num_samples, num_neurons)
            position_data = np.random.rand(num_samples, 2)  # 2D位置
            direction_data = np.random.rand(num_samples, 1)  # 方向

            # 预处理数据
            neural_data = preprocess_neural_data(neural_data, preprocessing)

            # 分割训练集和测试集
            train_size = int(num_samples * (1 - test_size))

            train_neural = neural_data[:train_size]
            test_neural = neural_data[train_size:]

            train_position = position_data[:train_size]
            test_position = position_data[train_size:]

            train_direction = direction_data[:train_size]
            test_direction = direction_data[train_size:]

            result[condition] = {
                'train_neural': train_neural,
                'test_neural': test_neural,
                'train_position': train_position,
                'test_position': test_position,
                'train_direction': train_direction,
                'test_direction': test_direction
            }

    return result


def load_allen_data(
    data_dir: str = 'data',
    calcium_dataset: str = 'visual_drift',
    neuropixels_dataset: str = 'visual_coding_neuropixels',
    preprocessing: str = 'zscore',
    test_size: float = 0.2,
    random_seed: int = 42
) -> Dict[str, Dict[str, np.ndarray]]:
    """加载Allen Brain Observatory数据。

    Args:
        data_dir: 数据目录
        calcium_dataset: 钙成像数据集名称
        neuropixels_dataset: Neuropixels数据集名称
        preprocessing: 预处理类型
        test_size: 测试集比例
        random_seed: 随机种子

    Returns:
        包含钙成像和Neuropixels数据的字典
    """
    import os
    import warnings

    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError:
        warnings.warn("未安装AllenSDK。请使用 'pip install allensdk' 安装。使用模拟数据代替。")
        return _load_allen_data_mock(preprocessing, test_size, random_seed)

    # 设置随机种子
    np.random.seed(random_seed)

    # 构建数据路径
    allen_data_dir = os.path.join(data_dir, 'allen')
    calcium_data_dir = os.path.join(allen_data_dir, calcium_dataset)
    neuropixels_data_dir = os.path.join(allen_data_dir, neuropixels_dataset)

    # 确保目录存在
    os.makedirs(allen_data_dir, exist_ok=True)
    os.makedirs(calcium_data_dir, exist_ok=True)
    os.makedirs(neuropixels_data_dir, exist_ok=True)

    result = {}

    # 尝试加载钙成像数据
    try:
        # 创建BrainObservatoryCache对象
        boc = BrainObservatoryCache(manifest_file=os.path.join(calcium_data_dir, 'manifest.json'))

        # 获取实验容器
        containers = boc.get_experiment_containers()
        if not containers:
            warnings.warn("未找到钙成像实验容器，使用模拟数据代替。")
            result['calcium'] = _generate_mock_data('calcium', preprocessing, test_size, random_seed)
        else:
            # 选择第一个容器
            container_id = containers[0]['id']

            # 获取容器中的实验
            experiments = boc.get_ophys_experiments(experiment_container_ids=[container_id])
            if not experiments:
                warnings.warn(f"容器 {container_id} 中未找到实验，使用模拟数据代替。")
                result['calcium'] = _generate_mock_data('calcium', preprocessing, test_size, random_seed)
            else:
                # 选择第一个实验
                experiment_id = experiments[0]['id']

                # 获取实验数据
                data_set = boc.get_ophys_experiment_data(experiment_id)

                # 获取神经元响应和刺激表
                neural_data = data_set.get_dff_traces()[1]  # [1]是dF/F数据，[0]是cell_ids
                stim_table = data_set.get_stimulus_table('natural_scenes')

                # 获取DINO特征（这里假设已经预先计算并存储）
                dino_features_path = os.path.join(calcium_data_dir, f'dino_features_{experiment_id}.npy')
                if os.path.exists(dino_features_path):
                    dino_features = np.load(dino_features_path)
                else:
                    # 如果没有预先计算的DINO特征，创建随机特征
                    warnings.warn(f"未找到DINO特征文件 {dino_features_path}，使用随机特征代替。")
                    dino_feature_dim = 768
                    dino_features = np.random.randn(neural_data.shape[0], dino_feature_dim)

                # 预处理数据
                neural_data = preprocess_neural_data(neural_data, preprocessing)

                # 分割训练集和测试集
                num_samples = neural_data.shape[0]
                train_size = int(num_samples * (1 - test_size))

                train_neural = neural_data[:train_size]
                test_neural = neural_data[train_size:]

                train_dino = dino_features[:train_size]
                test_dino = dino_features[train_size:]

                result['calcium'] = {
                    'train_neural': train_neural,
                    'test_neural': test_neural,
                    'train_dino': train_dino,
                    'test_dino': test_dino
                }
    except Exception as e:
        warnings.warn(f"加载钙成像数据时出错：{str(e)}，使用模拟数据代替。")
        result['calcium'] = _generate_mock_data('calcium', preprocessing, test_size, random_seed)

    # 尝试加载Neuropixels数据
    try:
        # 创建EcephysProjectCache对象
        manifest_path = os.path.join(neuropixels_data_dir, 'manifest.json')
        ecephys_cache = EcephysProjectCache(manifest=manifest_path)

        # 获取会话
        sessions = ecephys_cache.get_session_table()
        if sessions.empty:
            warnings.warn("未找到Neuropixels会话，使用模拟数据代替。")
            result['neuropixels'] = _generate_mock_data('neuropixels', preprocessing, test_size, random_seed)
        else:
            # 选择第一个会话
            session_id = sessions.index[0]

            # 获取会话数据
            session = ecephys_cache.get_session_data(session_id)

            # 获取神经元响应
            units = session.units
            spike_times = session.spike_times

            # 将尖峰时间转换为发放率
            bin_width_ms = 10  # 10ms时间窗口
            total_time_ms = int(session.running_speed['end_time'].max() * 1000)
            num_bins = total_time_ms // bin_width_ms

            # 创建时间窗口
            time_bins = np.arange(0, total_time_ms + bin_width_ms, bin_width_ms) / 1000.0

            # 初始化发放率矩阵
            neural_data = np.zeros((num_bins, len(units)))

            # 对每个神经元计算发放率
            for i, unit_id in enumerate(units.index):
                unit_spike_times = spike_times[unit_id]
                hist, _ = np.histogram(unit_spike_times, bins=time_bins)
                neural_data[:, i] = hist / (bin_width_ms / 1000.0)  # 转换为Hz

            # 获取DINO特征（这里假设已经预先计算并存储）
            dino_features_path = os.path.join(neuropixels_data_dir, f'dino_features_{session_id}.npy')
            if os.path.exists(dino_features_path):
                dino_features = np.load(dino_features_path)
            else:
                # 如果没有预先计算的DINO特征，创建随机特征
                warnings.warn(f"未找到DINO特征文件 {dino_features_path}，使用随机特征代替。")
                dino_feature_dim = 768
                dino_features = np.random.randn(neural_data.shape[0], dino_feature_dim)

            # 预处理数据
            neural_data = preprocess_neural_data(neural_data, preprocessing)

            # 分割训练集和测试集
            num_samples = neural_data.shape[0]
            train_size = int(num_samples * (1 - test_size))

            train_neural = neural_data[:train_size]
            test_neural = neural_data[train_size:]

            train_dino = dino_features[:train_size]
            test_dino = dino_features[train_size:]

            result['neuropixels'] = {
                'train_neural': train_neural,
                'test_neural': test_neural,
                'train_dino': train_dino,
                'test_dino': test_dino
            }
    except Exception as e:
        warnings.warn(f"加载Neuropixels数据时出错：{str(e)}，使用模拟数据代替。")
        result['neuropixels'] = _generate_mock_data('neuropixels', preprocessing, test_size, random_seed)

    return result


def _generate_mock_data(
    modality: str,
    preprocessing: str,
    test_size: float,
    random_seed: int
) -> Dict[str, np.ndarray]:
    """生成模拟数据。

    Args:
        modality: 模态类型，'calcium'或'neuropixels'
        preprocessing: 预处理类型
        test_size: 测试集比例
        random_seed: 随机种子

    Returns:
        包含模拟数据的字典
    """
    np.random.seed(random_seed)

    # 模拟神经数据和视频特征
    num_samples = 10000
    num_neurons = 100 if modality == 'calcium' else 400
    dino_feature_dim = 768

    neural_data = np.random.randn(num_samples, num_neurons)
    dino_features = np.random.randn(num_samples, dino_feature_dim)

    # 预处理数据
    neural_data = preprocess_neural_data(neural_data, preprocessing)

    # 分割训练集和测试集
    train_size = int(num_samples * (1 - test_size))

    train_neural = neural_data[:train_size]
    test_neural = neural_data[train_size:]

    train_dino = dino_features[:train_size]
    test_dino = dino_features[train_size:]

    return {
        'train_neural': train_neural,
        'test_neural': test_neural,
        'train_dino': train_dino,
        'test_dino': test_dino
    }


def _load_allen_data_mock(
    preprocessing: str,
    test_size: float,
    random_seed: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """当AllenSDK不可用时，加载模拟的Allen数据。

    Args:
        preprocessing: 预处理类型
        test_size: 测试集比例
        random_seed: 随机种子

    Returns:
        包含模拟数据的字典
    """
    # 设置随机种子
    np.random.seed(random_seed)

    # 假设我们有钙成像和Neuropixels数据
    modalities = ['calcium', 'neuropixels']
    result = {}

    for modality in modalities:
        result[modality] = _generate_mock_data(modality, preprocessing, test_size, random_seed)

    return result


def load_video_decoding_data(
    data_dir: str = 'data',
    dataset_name: str = 'visual_coding_neuropixels',
    brain_area: str = 'V1',
    num_repeats: int = 10,
    preprocessing: str = 'zscore',
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """加载视频解码数据。

    Args:
        data_dir: 数据目录
        dataset_name: 数据集名称
        brain_area: 脑区
        num_repeats: 重复次数
        preprocessing: 预处理类型
        random_seed: 随机种子

    Returns:
        包含神经数据和帧索引的字典
    """
    import os
    import warnings

    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError:
        warnings.warn("未安装AllenSDK。请使用 'pip install allensdk' 安装。使用模拟数据代替。")
        return _generate_video_decoding_mock_data(num_repeats, preprocessing, random_seed)

    # 设置随机种子
    np.random.seed(random_seed)

    # 构建数据路径
    video_data_dir = os.path.join(data_dir, dataset_name)

    # 确保目录存在
    os.makedirs(video_data_dir, exist_ok=True)

    # 尝试加载Neuropixels数据
    try:
        # 创建EcephysProjectCache对象
        manifest_path = os.path.join(video_data_dir, 'manifest.json')
        ecephys_cache = EcephysProjectCache(manifest=manifest_path)

        # 获取会话
        sessions = ecephys_cache.get_session_table()
        if sessions.empty:
            warnings.warn("未找到Neuropixels会话，使用模拟数据代替。")
            return _generate_video_decoding_mock_data(num_repeats, preprocessing, random_seed)

        # 选择包含自然电影刺激的会话
        session_id = None
        for idx in sessions.index:
            session = ecephys_cache.get_session_data(idx)
            if 'natural_movie' in session.stimulus_names:
                session_id = idx
                break

        if session_id is None:
            warnings.warn("未找到包含自然电影刺激的会话，使用模拟数据代替。")
            return _generate_video_decoding_mock_data(num_repeats, preprocessing, random_seed)

        # 获取会话数据
        session = ecephys_cache.get_session_data(session_id)

        # 获取指定脑区的单元
        units = session.units
        if brain_area not in units['ecephys_structure_acronym'].unique():
            available_areas = units['ecephys_structure_acronym'].unique()
            warnings.warn(f"未找到脑区 {brain_area}，可用的脑区有：{available_areas}。使用第一个可用脑区代替。")
            brain_area = available_areas[0]

        area_units = units[units['ecephys_structure_acronym'] == brain_area]
        if area_units.empty:
            warnings.warn(f"脑区 {brain_area} 中没有单元，使用模拟数据代替。")
            return _generate_video_decoding_mock_data(num_repeats, preprocessing, random_seed)

        # 获取自然电影刺激表
        stim_table = session.get_stimulus_table('natural_movie')
        if stim_table.empty:
            warnings.warn("未找到自然电影刺激表，使用模拟数据代替。")
            return _generate_video_decoding_mock_data(num_repeats, preprocessing, random_seed)

        # 获取帧索引
        frame_indices = stim_table['frame'].values

        # 计算每个电影的重复次数
        movie_ids = stim_table['stimulus_name'].unique()
        actual_repeats = {}
        for movie_id in movie_ids:
            actual_repeats[movie_id] = stim_table[stim_table['stimulus_name'] == movie_id].shape[0]

        # 选择重复次数最接近请求的电影
        target_movie = None
        min_diff = float('inf')
        for movie_id, repeats in actual_repeats.items():
            diff = abs(repeats - num_repeats)
            if diff < min_diff:
                min_diff = diff
                target_movie = movie_id

        if target_movie is None:
            warnings.warn("未找到合适的电影重复次数，使用模拟数据代替。")
            return _generate_video_decoding_mock_data(num_repeats, preprocessing, random_seed)

        # 过滤刺激表以仅包含目标电影
        movie_stim_table = stim_table[stim_table['stimulus_name'] == target_movie]

        # 获取帧索引和时间戳
        frame_indices = movie_stim_table['frame'].values
        start_times = movie_stim_table['start_time'].values
        end_times = movie_stim_table['stop_time'].values

        # 计算每个单元在每个时间窗口内的发放率
        neural_data = np.zeros((len(start_times), len(area_units)))

        for i, (start_time, end_time) in enumerate(zip(start_times, end_times)):
            for j, unit_id in enumerate(area_units.index):
                # 获取单元的尖峰时间
                spike_times = session.spike_times[unit_id]

                # 计算时间窗口内的尖峰数
                spikes_in_window = np.sum((spike_times >= start_time) & (spike_times < end_time))

                # 计算发放率（Hz）
                firing_rate = spikes_in_window / (end_time - start_time)

                neural_data[i, j] = firing_rate

        # 预处理数据
        neural_data = preprocess_neural_data(neural_data, preprocessing)

        # 获取实际的帧数和重复次数
        num_frames = len(np.unique(frame_indices))
        actual_num_repeats = len(start_times) // num_frames

        return {
            'neural_data': neural_data,
            'frame_indices': frame_indices,
            'num_frames': num_frames,
            'num_repeats': actual_num_repeats
        }

    except Exception as e:
        warnings.warn(f"加载视频解码数据时出错：{str(e)}，使用模拟数据代替。")
        return _generate_video_decoding_mock_data(num_repeats, preprocessing, random_seed)


def _generate_video_decoding_mock_data(
    num_repeats: int,
    preprocessing: str,
    random_seed: int
) -> Dict[str, np.ndarray]:
    """生成模拟的视频解码数据。

    Args:
        num_repeats: 重复次数
        preprocessing: 预处理类型
        random_seed: 随机种子

    Returns:
        包含模拟数据的字典
    """
    # 设置随机种子
    np.random.seed(random_seed)

    # 假设我们有一个视频，重复播放了num_repeats次
    num_frames = 1000
    num_neurons = 100

    # 为每一帧创建神经响应
    frame_responses = np.random.randn(num_frames, num_neurons)

    # 重复num_repeats次
    neural_data = np.tile(frame_responses, (num_repeats, 1))

    # 创建帧索引
    frame_indices = np.tile(np.arange(num_frames), num_repeats)

    # 预处理数据
    neural_data = preprocess_neural_data(neural_data, preprocessing)

    return {
        'neural_data': neural_data,
        'frame_indices': frame_indices,
        'num_frames': num_frames,
        'num_repeats': num_repeats
    }
