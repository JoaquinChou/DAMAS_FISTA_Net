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
import pandas

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
        SoundDataset(args.test_dir, args.simulation_single_sound_source_data),
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    return test_dataloader


def set_model(args, wk_reshape, A):
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

            model = DAMAS_FISTANet(args.LayNo, wk_reshape, A)
            model = model.cuda()
            model.load_state_dict(state_dict)

    else:
            model = DAMAS_FISTANet(args.LayNo, wk_reshape, A, args.show_DAS_result)
            model = model.cuda()

    return model



def test(test_dataloader, model, args):

    model.eval()
    with torch.no_grad():
        for idx, (CSM, sample_name) in enumerate(test_dataloader):
            start_time = time.time()
            
            sample_name = sample_name[0]
            CSM = CSM.cuda()
            output = model(CSM)
            end_time = time.time()

            print("time=", end_time - start_time)
            np_output = output.cpu().numpy()

            print("##MAX_output", torch.max(output))

            np_output = np_output.reshape(41, 41, order='F')

            f = h5py.File(args.output_dir + sample_name + '.h5','w')
            f['damas_fista_net_output'] = np_output
            f.close()

            pyContourf_two(np_output, np_output, args.results_dir, sample_name)

def main():
    args = parse_option()
    sys.stdout = Logger(args.results_dir + "log.txt")

    # 读取存储的w和A
    A = pandas.read_csv('./data/A.csv', header=None)
    temp_A = np.array(A)
    A = torch.from_numpy(temp_A.astype(float))
    wk_reshape = np.genfromtxt('./data/wk_reshape.csv', dtype=complex, delimiter=',') 
    temp_w = np.array(wk_reshape)
    wk_reshape = torch.from_numpy(temp_w.astype(complex))

    wk_reshape = wk_reshape.cuda(non_blocking=True)
    A = A.cuda(non_blocking=True)


    model = set_model(args, wk_reshape, A)
    # build data loader
    test_dataloader = set_loader(args)
    test(test_dataloader, model, args)

if __name__ == '__main__':
    
    main()