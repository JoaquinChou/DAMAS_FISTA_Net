import argparse
import os
import sys
sys.path.append('../../')
from utils.utils import developCSM, Fast_DAS, pyContourf_two, steerVector2, get_microphone_info, get_search_freq, get_DAS_result
import numpy as np
import h5py
import scipy.io as scio



def array_signal_trans_DAS(args):
    c = 343
    # 采样频率
    fs = 51200
    # 麦克风和声源之间的距离
    z_dist = 2.5
    # 扫描频率范围
    scan_freq = [2000]
    # 扫描区域限定范围和扫描网格分辨率
    scan_x = [-2, 2]
    scan_y = [-2, 2]
    scan_resolution = 0.1


    mic_pos, mic_centre = get_microphone_info(args.micro_array_path)

    args.save_DAS_results_path += 'One/'
    if not os.path.exists(args.save_DAS_results_path):
        os.makedirs(args.save_DAS_results_path)

    for file in os.listdir(args.data_path):
        if file.split('.')[-1] == 'txt':
            continue
        f = h5py.File(args.data_path + file, 'r')
        raw_sound_data = np.array(f['time_data'].value)
        # f = scio.loadmat(args.data_path + file)
        # raw_sound_data = np.array(f['one_data'])
        # raw_sound_data = np.transpose(raw_sound_data, (1, 0))
        
        N_total_samples = raw_sound_data.shape[0]
        N_mic = raw_sound_data.shape[1]

        _, _, _, _, frequencies = get_search_freq(N_total_samples, scan_freq, fs)
        CSM = developCSM(raw_sound_data, scan_freq, fs)
        g, w = steerVector2(z_dist, frequencies, scan_x + scan_y, scan_resolution, mic_pos.T, c, mic_centre)
        _, wk_reshape, _ = Fast_DAS(g, w, frequencies)
        DAS_result = get_DAS_result(wk_reshape, CSM, frequencies, N_mic, scan_x + scan_y, scan_resolution)

        print("Saving the " + file.split('.')[0] + "_DAS result!")
        scio.savemat(args.save_DAS_results_path + file.split('.')[0] + '.mat', {'DAS_result':DAS_result})


# only for one image testing
def check_DAS_result(args):

    sample_name = args.DAS_result_path.split('/')[-1].split('.')[0]
    print(sample_name)
    # just for test the DAS_result
    DAS_result = scio.loadmat(args.DAS_result_path)['DAS_result']

    pyContourf_two(DAS_result, DAS_result, './', sample_name)











        






def main():
    		
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='D:/Ftp_Server/zgx/data/Acoustic-NewData/data/')
    # parser.add_argument('--data_path', type=str, default='D:/Ftp_Server/zgx/data/Acoustic-NewData/generalization_one_data/')
    parser.add_argument('--save_DAS_results_path', type=str, default='./DAS_results/')
    # parser.add_argument('--save_DAS_results_path', type=str, default='./generalization_data_DAS_results/')
    parser.add_argument('--DAS_result_path', type=str, default='../../data/generalization/one_data_DAS_result.mat')
    parser.add_argument('--micro_array_path', type=str, default='D:/Ftp_Server/zgx/codes/Fast_DAS_2/MicArray/56_spiral_array.mat')
    args = parser.parse_args()

    # check_DAS_result(args)
    array_signal_trans_DAS(args)

if __name__ == '__main__':
    	
	main()