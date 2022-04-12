import h5py
import sys
import numpy as np
import math
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def get_search_freq(N_total_samples, search_freq, Fs):
    t_start = 0
    t_end = N_total_samples / Fs
    # 开始和结束时间点
    start_sample = math.floor(t_start * Fs)
    end_sample = math.ceil(t_end * Fs)
    # 选取在扫描频率之间的点
    x_fr = Fs / end_sample * np.arange(0, math.floor(end_sample / 2))  
    _, freq_sels = ismember(search_freq, x_fr)
    freq_sels = freq_sels.reshape((freq_sels.shape[0], 1))
    # 扫描频点的个数
    N_freqs = len(freq_sels)
    # 扫描的频点
    frequencies = x_fr[freq_sels]

    return start_sample, end_sample, freq_sels, N_freqs, frequencies


def developCSM(mic_signal, search_freq, Fs):

    # 定义采样点数和开始结束时间
    N_total_samples = mic_signal.shape[0]
    # 麦克风阵列数
    N_mic = mic_signal.shape[1]

    start_sample, end_sample,freq_sels, N_freqs, _ = get_search_freq(N_total_samples, search_freq, Fs)

    # 初始化互谱矩阵 CSM
    CSM = np.zeros((N_mic, N_mic, N_freqs), dtype=complex)  # npU
    # 对采集到的时域数据进行傅里叶变换
    mic_signal_fft = np.sqrt(2) * fft(mic_signal[start_sample : end_sample + 1, :], axis=0) / (end_sample -  start_sample)
    # 生成互谱矩阵 CSM
    for K in range(0, N_freqs):
        # 计算第 K 个频率下的互谱矩阵
        CSM[:, :, K] = mic_signal_fft[freq_sels[K], :].T * mic_signal_fft[freq_sels[K], :].conj()


    return CSM



def steerVector2(plane_distance, frequencies, scan_limits, grid_resolution, mic_positions, c, mic_centre):

    # ------ 计算导向矢量
    # 麦克风个数和扫描频点个数
    N_mic = mic_positions.shape[1]
    N_freqs = frequencies.size

    # 定义扫描平面
    x = np.arange(scan_limits[0], scan_limits[1] + grid_resolution, grid_resolution)
    x = x.reshape(1, x.size)
    y = np.arange(scan_limits[2], scan_limits[3] + grid_resolution, grid_resolution)
    y = y.reshape(1, y.size)
    z = plane_distance
    N_X = x.size
    N_Y = y.size

    # 沿着Y轴扩大N_X倍
    X = np.tile(x, (N_X, 1))
    # 沿着X轴扩大N_Y倍
    Y = np.tile(y.T, (1, N_Y))

    # 初始化转向矢量
    g = np.zeros((N_X, N_Y, N_mic, N_freqs),  dtype=complex)
    w = np.zeros((N_X, N_Y, N_mic, N_freqs), dtype=complex)

    # 计算扫描平面到麦克风阵列中心的距离
    r_scan_to_mic_centre = np.sqrt((X - mic_centre[0]) ** 2 + (Y - mic_centre[1]) ** 2 + (z - mic_centre[2]) ** 2)

    # 初始化变量
    r_scan_to_mic = np.zeros((N_X, N_Y, N_mic))

    # 计算转向矢量
    for K in range(N_freqs):
        # 角频率 w
        omega = 2 * np.pi * frequencies[K]
        for m in range(N_mic):
        # 计算扫描平面到第 m 个麦克风的距离
           r_scan_to_mic[:, :, m] = np.sqrt((X - mic_positions[0, m]) ** 2 + (Y - mic_positions[1, m]) ** 2 + z ** 2)
           w[:, :, m, K] = (r_scan_to_mic[:,:,m] / r_scan_to_mic_centre) * np.exp(-1j * omega * (r_scan_to_mic[:, :, m] - r_scan_to_mic_centre) / c)
           g[:,:,m, K] = 1 / ((r_scan_to_mic[:,:,m] / r_scan_to_mic_centre)) * np.exp(-1j * omega * (r_scan_to_mic[:,:,m] - r_scan_to_mic_centre) / c)


    return g, w



def Fast_DAS(g, w, frequencies):

    # DAS 算法
    # 参数初始化
    N_freqs = frequencies.size
    N_mic = w.shape[2]

    # 计算波束形成的声功率图
    for K in range (N_freqs):
        #  频率 K 对应的转向矢量
        wk = w[:, :, :, K]
        gk = g[:, :, :, K]
        if int(wk.size / N_mic) != wk.size / N_mic:
            print("Input Type can't be reshaped!")
            break
        w_reshape = wk.reshape(int(wk.size / N_mic), N_mic, order='F')
        wk_reshape = w_reshape.T

        g_reshape = gk.reshape(int(wk.size / N_mic), N_mic, order='F')
        A = (np.abs(np.dot(w_reshape.conj(), g_reshape.T)) ** 2) / (N_mic ** 2)        
  
    return A, wk_reshape, N_mic


def get_DAS_result(wk_reshape, CSM, frequencies, N_mic, scan_limits, scan_resolution):
    # 定义扫描平面
    X = np.arange(scan_limits[0], scan_limits[1] + scan_resolution, scan_resolution)
    X = X.reshape(1, X.size)
    Y = np.arange(scan_limits[2], scan_limits[3] + scan_resolution, scan_resolution)
    Y = Y.reshape(1, Y.size)
    N_X = X.size
    N_Y = Y.size

    # 变量初始化
    B = np.zeros((1, N_X * N_Y))
    N_freqs = frequencies.size
    for K in range (N_freqs):
         # 频率 K 下的波束成像图
        B_freqK = np.sum(wk_reshape.conj() * (np.dot(CSM[:, :, K], wk_reshape)), axis=0) / (N_mic ** 2)
        # 累加各个频率成分
        B = B + B_freqK
        
    B = np.real(B.T)
    DAS_result = B.reshape(N_X, N_Y, order='F')

    return DAS_result



def ismember(a, b):
    # tf = np.in1d(a,b) # for newer versions of numpy
    tf = np.array([i in b for i in a])
    u = np.unique(np.array(a)[tf])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a, tf)])
    return tf, index


def pyContourf(output):
    BF_dr = 6
    x, y = np.where(output == np.max(output))
    maxSPL = np.ceil(np.max(output))
    #print("maxSPL", maxSPL)
    plt.contourf(output, cmap = plt.cm.hot, levels=np.arange(maxSPL - BF_dr, maxSPL + 1, 1))
    plt.colorbar()
    plt.scatter(y, x, marker='x')
    # plt.show()
    plt.savefig('test.png')
    plt.close()

def pyContourf_two(output, label, save_dir, file_name):

    fig = plt.figure(figsize=(41, 20))
    ax = fig.add_subplot(1, 2, 1)
    ax.contourf(output, cmap = plt.cm.hot)
    ax = fig.add_subplot(1, 2, 2)
    ax.contourf(label, cmap = plt.cm.hot)
    # plt.colorbar()
    # plt.scatter(x, y, marker='x')

    plt.savefig(save_dir + '/' + file_name + '.png')
    plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(model, optimizer, opt, epoch, save_file):
    #print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def get_microphone_info(micro_array_path):
    mic_array = h5py.File(micro_array_path, 'r')['array']
    mic_array = np.array(mic_array)

    mic_x_axis = mic_array[0]
    mic_x_axis = mic_x_axis.reshape(mic_x_axis.shape[0], 1)
    
    mic_y_axis = mic_array[1]
    mic_y_axis = mic_y_axis.reshape(mic_y_axis.shape[0], 1)                   
    mic_z_axis = np.zeros((56, 1))
    mic_pos = np.concatenate((mic_x_axis, mic_y_axis, mic_z_axis), axis=1)
    # 阵列中心的坐标
    mic_centre = mic_pos.mean(axis=0)

    return mic_pos, mic_centre


def get_magnitude(test_raw_sound_data, simulation_single_sound_source_data):
    
    test_raw_sound_data = np.transpose(test_raw_sound_data, (1, 0))
    simulation_single_sound_source_data = np.transpose(simulation_single_sound_source_data, (1, 0))
    rand_row = np.random.randint(test_raw_sound_data.shape[0])

    rand_row_raw_sound_data = test_raw_sound_data[rand_row]
    rand_simulation_single_sound_source_data = simulation_single_sound_source_data[rand_row]

    magnitude = sum(abs(rand_simulation_single_sound_source_data)) / sum(abs(rand_row_raw_sound_data))

    return magnitude


def data_preprocess(raw_sound_data):

    # 采样频率
    fs = 51200
    # 扫描频率范围
    scan_freq = [2000]

    CSM = developCSM(raw_sound_data, scan_freq, fs)

    return CSM


def generate_A_and_wk_reshape(micro_array_path, save_A_and_wk_reshape_path, N_total_samples):
    c = 343
    # 采样频率
    fs = 51200
    # 麦克风和声源之间的距离
    z_dist = 3.0
    # 扫描频点
    # 扫描区域限定范围和扫描网格分辨率
    scan_x = [-2, 2]
    scan_y = [-2, 2]
    scan_resolution = 0.1
    # 扫描频率范围
    scan_freq = [2000]

    mic_pos, mic_centre = get_microphone_info(micro_array_path)
    _, _, _, _, frequencies = get_search_freq(N_total_samples, scan_freq, fs)
    g, w = steerVector2(z_dist, frequencies, scan_x + scan_y, scan_resolution, mic_pos.T, c, mic_centre)
    A, wk_reshape, _ =  Fast_DAS(g, w, frequencies)

    print("Saving the A in " + save_A_and_wk_reshape_path)
    np.savetxt(save_A_and_wk_reshape_path + 'A.csv', A, delimiter = ',')
    
    print("Saving the wk_reshape in " + save_A_and_wk_reshape_path)
    np.savetxt(save_A_and_wk_reshape_path + 'wk_reshape.csv', wk_reshape, delimiter = ',')
 


#log the terminal message in the txt
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    test_raw_data = np.random.random(size=(1024, 56))
    simulation_single_sound_source_data = np.random.random(size=(1024, 56))
    get_magnitude(test_raw_data, simulation_single_sound_source_data)





