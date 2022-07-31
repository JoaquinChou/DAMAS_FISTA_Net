from networks.damas_fista_net import DAMAS_FISTANet
from utils.utils import pyContourf_two, Logger
from datasets.generalization_dataset import SoundDataset
import numpy as np
import torch
import argparse
import os
import sys
import h5py
import time
from utils.utils import get_search_freq
from utils.config import Config


def parse_option():
    parser = argparse.ArgumentParser('argument for testing')

    parser.add_argument('--test_dir', 
                        help='The directory used to evaluate the models',
                        default='./data/One_test.txt', type=str)
    parser.add_argument('--results_dir', 
                        help='The directory used to save the save image',
                        default='./img_results/', type=str)
    parser.add_argument('--output_dir', 
                        help='The directory used to save the save image',
                        default='./output_results/', type=str)
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--LayNo', 
                        default=5, 
                        type=int,
                        help='iteration nums')
    parser.add_argument('--show_DAS_result',
                        action='store_true',
                        help='whether show DAS result')
    parser.add_argument('--simulation_single_sound_source_data', 
                        help='The simulation data is randomlt chosen from the simulation dataset',
                        default='D:/Ftp_Server/zgx/data/Acoustic-NewData/data/One_687.h5', type=str)
    parser.add_argument('--micro_array_path', default='./data/56_spiral_array.mat', type=str, help='micro array path')
    parser.add_argument('--wk_reshape_path', default='./data/wk_reshape.npy', type=str, help='wk_reshape path')
    parser.add_argument('--A_path', default='./data/A.npy', type=str, help='A path')
    parser.add_argument('--ATA_path', default='./data/ATA.npy', type=str, help='ATA path')
    parser.add_argument('--L_path', default='./data/ATA_eigenvalues.npy', type=str, help='L path')
    parser.add_argument('--two_source', action='store_true', help='whether use two source')
    parser.add_argument('--config', default='./utils/config.yml', type=str, help='config file path')
    parser.add_argument('--add_noise', action='store_true', help='whether add gaussian noise to test')
    parser.add_argument('--dB_value', default=0, type=float, help='gaussian noise value')
  

    args = parser.parse_args()

    args.results_dir += args.ckpt.split('/')[-2] + '/' + args.ckpt.split('/')[-1].split('.')[0] + '/'
    if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)

    args.output_dir += args.ckpt.split('/')[-2] + '/' + args.ckpt.split('/')[-1].split('.')[0] + '/'
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    return args

# 加载声源数据
def set_loader(args):
    test_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.test_dir, args.simulation_single_sound_source_data, args.wk_reshape_path, args.A_path, args.ATA_path, args.L_path, args.config),
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    return test_dataloader


def set_model(args):
    # 加载模型


    if not args.show_DAS_result:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        if torch.cuda.is_available():
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict

            model = DAMAS_FISTANet(args.LayNo)
            model = model.cuda()
            model.load_state_dict(state_dict)

    else:
            model = DAMAS_FISTANet(args.LayNo, args.show_DAS_result)
            model = model.cuda()

    return model



def test(test_dataloader, model, args):

    model.eval()

    # 预热
    cnt=1
    with torch.no_grad():
        for idx, (CSM, wk_reshape, A, ATA, L, sample_name) in enumerate(test_dataloader):
            if cnt==100:
                break
            cnt+=1
            
            sample_name = sample_name[0]
            CSM = CSM.cuda(non_blocking=True)
            wk_reshape = wk_reshape.cuda(non_blocking=True)
            A = A.cuda(non_blocking=True)
            ATA = ATA.cuda(non_blocking=True)
            L = L.cuda(non_blocking=True)

            output = None
            # forward
            for K in range(len(args.frequencies)):
                CSM_K = CSM[:, :, :, K]
                wk_reshape_K = wk_reshape[:, :, :, K]
                A_K = A[:, :, :, K]
                ATA_K = ATA[:, :, :, K]
                L_K = L[:, K]
                L_K = torch.unsqueeze(L_K, 1).to(torch.float64)
                L_K = torch.unsqueeze(L_K, 2).to(torch.float64)
                # forward
                if output is None:
                    output = model(CSM_K, wk_reshape_K, A_K, ATA_K, L_K)
                else:
                    output += model(CSM_K, wk_reshape_K, A_K, ATA_K, L_K)




    time_list = list()
    with torch.no_grad():
        for idx, (CSM, wk_reshape, A, ATA, L, sample_name) in enumerate(test_dataloader):
            start_time = time.time()
            
            sample_name = sample_name[0]
            CSM = CSM.cuda(non_blocking=True)
            wk_reshape = wk_reshape.cuda(non_blocking=True)
            A = A.cuda(non_blocking=True)
            ATA = ATA.cuda(non_blocking=True)
            L = L.cuda(non_blocking=True)

            output = None
            torch.cuda.synchronize()
            start_time = time.time()

            # forward
            for K in range(len(args.frequencies)):
                CSM_K = CSM[:, :, :, K]
                wk_reshape_K = wk_reshape[:, :, :, K]
                A_K = A[:, :, :, K]
                ATA_K = ATA[:, :, :, K]
                L_K = L[:, K]
                L_K = torch.unsqueeze(L_K, 1).to(torch.float64)
                L_K = torch.unsqueeze(L_K, 2).to(torch.float64)
                # forward
                if output is None:
                    output = model(CSM_K, wk_reshape_K, A_K, ATA_K, L_K)
                else:
                    output += model(CSM_K, wk_reshape_K, A_K, ATA_K, L_K)

            # 同步GPU时间
            torch.cuda.synchronize()
            end_time = time.time()
            now_time = end_time - start_time
           
            np_output = output.cpu().numpy()
            np_output = np_output.reshape(41, 41, order='F')

            f = h5py.File(args.output_dir + sample_name + '.h5','w')
            f['damas_fista_net_output'] = np_output
            f.close()

            pyContourf_two(np_output, np_output, args.results_dir, sample_name)

            time_list.append(now_time)
            print(str(idx + 1) + "\ttime={}\t".format(now_time))
        
        print("time_for_mean={}".format(np.mean(time_list)))



def main():
    args = parse_option()
    sys.stdout = Logger(args.results_dir + "log.txt")

    con = Config(args.config).getConfig()['base']
    _, _, _, _, args.frequencies = get_search_freq(con['N_total_samples'], con['scan_low_freq'], con['scan_high_freq'], con['fs'])

    model = set_model(args)
    print(model)
    # build data loader
    test_dataloader = set_loader(args)
    test(test_dataloader, model, args)

if __name__ == '__main__':
    
    main()