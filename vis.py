from networks.damas_fista_net import DAMAS_FISTANet
from utils.utils import pyContourf_two, Logger
from datasets.dataset import SoundDataset
import numpy as np
import torch
import argparse
import os
import sys
import h5py
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
    parser.add_argument('--label_dir', 
                        help='The directory used to evaluate the models',
                        default='D:/Ftp_Server/zgx/data/Acoustic-NewData/DAMAS_FISTA_Net_Label/', type=str)
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
                        
    parser.add_argument('--LayNo', 
                        default=5, 
                        type=int,
                        help='iteration nums')
    parser.add_argument('--show_DAS_result',
                        action='store_true',
                        help='whether show DAS result')

    parser.add_argument('--horizonal_distance', default=2.5, type=float, help='horizonal distance between microphone array and sound source')
    parser.add_argument('--micro_array_path', default='./data/56_spiral_array.mat', type=str, help='micro array path')
  

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
        SoundDataset(args.test_dir, args.label_dir, args.horizonal_distance, args.micro_array_path),
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    return test_dataloader


def set_model(args, wk_reshape, A):

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



def test(test_dataloader, model, args):

    model.eval()
    with torch.no_grad():
        for idx, (CSM, label, sample_name) in enumerate(test_dataloader):
            sample_name = sample_name[0]
            CSM = CSM.cuda()
            label = label.cuda()
            output = model(CSM)
           
            np_output = output.cpu().numpy()
            np_label = label.cpu().numpy()


            print("##MAX_output", torch.max(output))
            print("####MAX_label", torch.max(label))

            np_label = np_label.reshape(41, 41, order='F')
            np_output = np_output.reshape(41, 41, order='F')

            f = h5py.File(args.output_dir + sample_name + '.h5','w')
            f['damas_fista_net_output'] = np_output
            f.close()

            pyContourf_two(np_output, np_label, args.results_dir, sample_name)

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