import argparse
import os
import sys
sys.path.append('../../')
from utils.utils import Fast_DAS, pyContourf_two, steerVector2, get_microphone_info, get_search_freq, get_DAS_result, wgn, data_preprocess
from utils.config import Config
import numpy as np
import h5py
import scipy.io as scio




def array_signal_trans_DAS(args):
    con = Config(args.config).getConfig()['base']

    mic_pos, mic_centre = get_microphone_info(args.micro_array_path)

    args.save_DAS_results_path += 'One/'
    if not os.path.exists(args.save_DAS_results_path):
        os.makedirs(args.save_DAS_results_path)

    for file in os.listdir(args.data_path):
        if file.split('.')[-1] == 'txt':
            continue
        f = h5py.File(args.data_path + file, 'r')
        raw_sound_data = np.array(f['time_data'].value)
        if args.add_noise:
            raw_sound_data += wgn(raw_sound_data, args.dB_value)
        # f = scio.loadmat(args.data_path + file)
        # raw_sound_data = np.array(f['one_data'])
        # raw_sound_data = np.transpose(raw_sound_data, (1, 0))
        
        N_mic = raw_sound_data.shape[1]

        _, _, _, _, frequencies = get_search_freq(con['N_total_samples'], con['scan_low_freq'], con['scan_high_freq'], con['fs'])
        CSM = data_preprocess(raw_sound_data, args.config)
        g, w = steerVector2(con['z_dist'], frequencies, con['scan_x'] + con['scan_y'], con['scan_resolution'], mic_pos.T, con['c'], mic_centre)
        _, wk_reshape, _ = Fast_DAS(g, w, frequencies)
        DAS_result = get_DAS_result(wk_reshape, CSM, frequencies, N_mic, con['scan_x'] + con['scan_y'], con['scan_resolution'] )

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
    parser.add_argument('--save_DAS_results_path', type=str, default='./DAS_results_-10dB/')
    # parser.add_argument('--save_DAS_results_path', type=str, default='./generalization_data_DAS_results/')
    parser.add_argument('--DAS_result_path', type=str, default='../../data/generalization/one_data_DAS_result.mat')
    parser.add_argument('--micro_array_path', type=str, default='D:/Ftp_Server/zgx/codes/Fast_DAS_2/MicArray/56_spiral_array.mat')

    parser.add_argument('--config', default='../../utils/config.yml', type=str, help='config file path')
    parser.add_argument('--add_noise', action='store_true', help='whether add gaussian noise to test')
    parser.add_argument('--dB_value', default=0, type=float, help='gaussian noise value')
  
    args = parser.parse_args()

    # check_DAS_result(args)
    array_signal_trans_DAS(args)

if __name__ == '__main__':
    	
	main()