import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import rel_entr
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# 用DTW计算两个轨迹的相似度
def cal_dtw_similarity(traj1, traj2):
    distance, _ = fastdtw(traj1, traj2, dist=euclidean)
    return distance

def cal_mse_euclidean_similarity(traj1, traj2):
    scaled_traj2 = Scaled_length(traj1, traj2)
    return np.mean((traj1 - scaled_traj2) * (traj1 - scaled_traj2))

def calculate_kl_1d(distribution1, distribution2):
    kl_divergence = np.sum(rel_entr(distribution1, distribution2))
    return kl_divergence

def cal_kl_cost(data1, data2):
    L = data1.shape[0]
    kl_cost = 0.0
    for i in range(L):
        kl_cost += calculate_kl_1d(data1[i], data2[i])
    return kl_cost / L

def _FFT(trajectory: list):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    multivariate_function = np.vstack((x, y, z))
    
    # 进行多元FFT变换
    fft_ori = np.fft.fftn(multivariate_function)

    return fft_ori

def FFT(trajectory: list):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    multivariate_function = np.vstack((x, y, z))
    
    # 进行多元FFT变换
    fft_ori = np.fft.fftn(multivariate_function)
    # print(fft_ori)
    # fft_result = np.fft.fftshift(fft_ori)
    # print(fft_result)
    # 计算频谱
    power_spectrum = np.abs(fft_ori)
    # print(power_spectrum)

    return power_spectrum

def Scaled_length(reference_distri, compare_distri):
    # 原始数组的行数和列数
    original_rows, original_cols = compare_distri.shape

    # 插值为3x10数组
    desired_rows, desired_cols = reference_distri.shape

    # 定义插值函数
    interp_func = RegularGridInterpolator((np.arange(original_rows), np.arange(original_cols)), compare_distri)

    # 生成插值节点
    new_rows_indices = np.linspace(0, original_rows - 1, desired_rows)
    new_cols_indices = np.linspace(0, original_cols - 1, desired_cols)
    grid_x, grid_y = np.meshgrid(new_rows_indices, new_cols_indices, indexing='ij')
    new_indices = np.stack((grid_x, grid_y), axis=-1)

    # 在新数组大小上进行插值
    new_array = interp_func(new_indices)

    return new_array

def Normalization(arr):
    normalized_array = np.array(arr) / np.array(arr).sum(axis=1)[:, np.newaxis]
    return normalized_array

def cal_mse_cost(data1, data2):
    return np.mean(np.square(data1 - data2))

def calculate_kl_contour_cost(init, comp):
    # init: N * 3, comp: N * 3
    FFT_init = FFT(init)
    FFT_comp = FFT(comp)
    scaled_comp = Scaled_length(FFT_init, FFT_comp)
    return cal_kl_cost(Normalization(FFT_init), Normalization(scaled_comp))
    # return cal_mse_cost(Normalization(FFT_init), Normalization(scaled_comp))
    # return cal_mse_cost(FFT_init, scaled_comp)

def calculate_js_contour_cost(init, comp):
    # init: N * 3, comp: N * 3
    FFT_init = FFT(init)
    FFT_comp = FFT(comp)
    scaled_comp = Scaled_length(FFT_init, FFT_comp)
    js_div = 0.5 * cal_kl_cost(Normalization(FFT_init), (Normalization(FFT_init) + Normalization(scaled_comp)) / 2) + 0.5 * cal_kl_cost(Normalization(scaled_comp), (Normalization(FFT_init) + Normalization(scaled_comp)) / 2)
    return js_div

def calculate_nmse_contour_cost(init, comp):
    # init: N * 3, comp: N * 3
    FFT_init = FFT(init)
    FFT_comp = FFT(comp)
    scaled_comp = Scaled_length(FFT_init, FFT_comp)
    return cal_mse_cost(Normalization(FFT_init), Normalization(scaled_comp))
    # return cal_mse_cost(FFT_init, scaled_comp)

def calculate_mse_contour_cost(init, comp):
    # init: N * 3, comp: N * 3
    FFT_init = FFT(init)
    FFT_comp = FFT(comp)
    scaled_comp = Scaled_length(FFT_init, FFT_comp)
    v = FFT_init - scaled_comp

    return np.mean(v * v)

def cal_ex_item(init, comp):
    FFT_init = FFT(init)
    FFT_comp = FFT(comp)
    # scaled_comp = Scaled_length(FFT_init, FFT_comp)
    # print(FFT_init.shape)

    item1 = np.sum(FFT_init * FFT_comp)
    item2 = np.abs(np.sum(_FFT(init) * np.conj(_FFT(comp))))

    return 2*(item1 - item2)

def calculate_omse_contour_cost(init, comp):
    # init: N * 3, comp: N * 3
    FFT_init = _FFT(init)
    FFT_comp = _FFT(comp)
    scaled_comp = Scaled_length(FFT_init, FFT_comp)
    v = FFT_init - scaled_comp
    vv = np.conj(v)
    return np.abs(np.mean(v * vv))