from networks.damas_fista_net import DAMAS_FISTANet
from utils.utils import pyContourf_two, Logger
from datasets.dataset import SoundDataset
import numpy as np
import torch
import argparse
import os
import sys
import h5py
from utils.config import Config
import math
import time
from utils.utils import get_search_freq, neighbor_2_zero, find_match_source

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

    parser.add_argument('--micro_array_path', default='./data/56_spiral_array.mat', type=str, help='micro array path')
    parser.add_argument('--wk_reshape_path', default='./data/wk_reshape.npy', type=str, help='wk_reshape path')
    parser.add_argument('--A_path', default='./data/A.npy', type=str, help='A path')
    parser.add_argument('--ATA_path', default='./data/ATA.npy', type=str, help='ATA path')
    parser.add_argument('--L_path', default='./data/ATA_eigenvalues.npy', type=str, help='L path')
    parser.add_argument('--more_source', action='store_true', help='whether use more source')
    parser.add_argument('--source_num', default=1, type=int, help='source_num')
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
def set_loader(args, config):
    test_dataloader = torch.utils.data.DataLoader(
        
        SoundDataset(args.test_dir, args.label_dir, config['z_dist'], args.micro_array_path, args.wk_reshape_path, args.A_path, args.ATA_path, args.L_path, None, args.config, add_noise=args.add_noise, dB_value=args.dB_value, more_source=args.more_source),
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    return test_dataloader


def set_model(args):

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


# def test(test_dataloader, model, args, config):
    
#     model.eval()
#     location_bias_list = list()
#     Power_bias_list = list()
#     time_list = list()
#     scanning_area_X = np.arange(config['scan_x'][0], config['scan_x'][1] + config['scan_resolution'], config['scan_resolution'])
#     scanning_area_Y = np.arange(config['scan_y'][0], config['scan_y'][1] + config['scan_resolution'], config['scan_resolution'])

   


#     with torch.no_grad():
#         for idx, (CSM, wk_reshape, A, ATA, L, label, sample_name) in enumerate(test_dataloader):
#             sample_name = sample_name[0]
#             CSM = CSM.cuda(non_blocking=True)
#             wk_reshape = wk_reshape.cuda(non_blocking=True)
#             A = A.cuda(non_blocking=True)
#             ATA = ATA.cuda(non_blocking=True)
#             L = L.cuda(non_blocking=True)

#             output = None
#             torch.cuda.synchronize()
#             start_time = time.time()

#             # forward
#             for K in range(len(args.frequencies)):
#                 CSM_K = CSM[:, :, :, K]
#                 wk_reshape_K = wk_reshape[:, :, :, K]
#                 A_K = A[:, :, :, K]
#                 ATA_K = ATA[:, :, :, K]
#                 L_K = L[:, K]
#                 L_K = torch.unsqueeze(L_K, 1).to(torch.float64)
#                 L_K = torch.unsqueeze(L_K, 2).to(torch.float64)
#                 # forward
#                 if output is None:
#                     output = model(CSM_K, wk_reshape_K, A_K, ATA_K, L_K)
#                 else:
#                     output += model(CSM_K, wk_reshape_K, A_K, ATA_K, L_K)

#             # 同步GPU时间
#             torch.cuda.synchronize()
#             end_time = time.time()
#             now_time = end_time - start_time
           

#             np_gt = label.cpu().numpy()
#             np_output = output.cpu().numpy()

#             np_label = np_gt.reshape(41, 41, order='F')
#             np_out = np_output.reshape(41, 41, order='F')
#             f = h5py.File(args.output_dir + sample_name + '.h5','w')
#             f['damas_fista_net_output'] = np_out
#             f.close()
#             pyContourf_two(np_out, np_label, args.results_dir, sample_name)

#             np_gt = np.squeeze(np_gt, 0)
#             max_gt = max(np_gt)[0]
#             np_gt = np.where(np_gt == max(np_gt))
#             gt_x_pos = math.ceil((np_gt[0][0] + 1) / len(scanning_area_X))
#             gt_y_pos = np.mod((np_gt[0][0] + 1), len(scanning_area_Y))
#             gt_x = scanning_area_X[gt_x_pos - 1]
#             gt_y = scanning_area_Y[gt_y_pos - 1]


#             np_output = np.squeeze(np_output, 0)
#             max_output = max(np_output)[0]
#             np_output = np.where(np_output == max(np_output))
#             output_x_pos = math.ceil((np_output[0][0] + 1) / len(scanning_area_X))
#             output_y_pos = np.mod((np_output[0][0] + 1), len(scanning_area_Y))
#             output_x = scanning_area_X[output_x_pos - 1]
#             output_y = scanning_area_Y[output_y_pos - 1]

#             Power_output = max_output
#             Power_label = max_gt 
#             Power_bias = np.abs(Power_output - Power_label)
           
#             location_bias = np.sqrt(((output_x - gt_x)**2 + (output_y - gt_y)**2))
#             location_bias_list.append(location_bias)
#             Power_bias_list.append(Power_bias)

#             time_list.append(now_time)
           

#             print(str(idx + 1) + "___label_x={}\t label_y={}\t Power_label={}".format(
#                                                         gt_x, gt_y, Power_label))

#             print(str(idx + 1) + "___output_x={}\t output_y={}\t Power_output={}\t time={}".format(
#                                                         output_x, output_y, Power_output, now_time))

#             print(str(idx + 1) + "___location_bias={}\t Power_bias={}".format(
#                                                         location_bias, Power_bias))
        
#         print("mean_location_bias_in_val={}\t mean_Power_bias_in_val={}\t time_for_mean={}".format(np.mean(location_bias_list), np.mean(Power_bias_list), np.mean(time_list)))


def test_more_source(test_dataloader, model, args, config):
    """test"""

    model.eval()
    location_bias_list = list()
    Power_bias_list = list()
    time_list = list()
    scanning_area_X = np.arange(config['scan_x'][0], config['scan_x'][1] + config['scan_resolution'], config['scan_resolution'])
    scanning_area_Y = np.arange(config['scan_y'][0], config['scan_y'][1] + config['scan_resolution'], config['scan_resolution'])

    # 预热
    cnt=1
    with torch.no_grad():
        for idx, (CSM, wk_reshape, A, ATA, L, _, _) in enumerate(test_dataloader):
            if cnt==100:
                break
            cnt+=1
        
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



   
    with torch.no_grad():
        for idx, (CSM, wk_reshape, A, ATA, L, label, sample_name) in enumerate(test_dataloader):
    
            sample_name = sample_name[0]
            CSM = CSM.cuda(non_blocking=True)
            wk_reshape = wk_reshape.cuda(non_blocking=True)
            A = A.cuda(non_blocking=True)
            ATA = ATA.cuda(non_blocking=True)
            L = L.cuda(non_blocking=True)

            # forward
            output = None
            torch.cuda.synchronize()
            start_time = time.time()
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

            torch.cuda.synchronize()
            end_time = time.time()
            now_time = end_time - start_time
            time_list.append(now_time)

            np_gt = label.cpu().numpy()
            np_output = output.cpu().numpy()    

            np_label = np_gt.reshape(41, 41, order='F')
            np_out = np_output.reshape(41, 41, order='F')
            f = h5py.File(args.output_dir + sample_name + '.h5','w')
            f['damas_fista_net_output'] = np_out
            f.close()
            pyContourf_two(np_out, np_label, args.results_dir, sample_name)


            np_gt = np.squeeze(np.squeeze(np_gt, 0), 1)
            np_output = np.squeeze(np.squeeze(np_output, 0), 1)
            print("source num is->", args.source_num)

            # use the output_mat and gt_mat to calculate the location error
            output_mat = np.zeros((args.source_num, 3))
            gt_mat = np.zeros((args.source_num, 3))

            for i in range(args.source_num):
                max_gt = max(np_gt)
                print("max_gt->", max_gt)
                np_gt_index = np.where(np_gt == max_gt)[0][0]
                print("np_gt_index->", np_gt_index)

                gt_y_pos = math.ceil((np_gt_index + 1) / len(scanning_area_X))
                gt_x_pos = (np_gt_index + 1) - (gt_y_pos - 1) * len(scanning_area_X)
                gt_x = scanning_area_X[gt_x_pos - 1]
                gt_y = scanning_area_Y[gt_y_pos - 1]

                max_output = max(np_output)
                print("max_output->", max_output)
                np_output_index = np.where(np_output == max_output)[0][0]
                print("np_output_index->", np_output_index)
                
                output_y_pos = math.ceil((np_output_index + 1) / len(scanning_area_X))
                output_x_pos = (np_output_index + 1) - (output_y_pos - 1) * len(scanning_area_X)
                output_x = scanning_area_X[output_x_pos - 1]
                output_y = scanning_area_Y[output_y_pos - 1]

                output_mat[i][0] = output_x
                output_mat[i][1] = output_y
                output_mat[i][2] = max_output

                gt_mat[i][0] = gt_x
                gt_mat[i][1] = gt_y
                gt_mat[i][2] = max_gt
    
                # 置零操作
                gt_matrix = np_gt.reshape(41, 41, order='F')
                out_matrix = np_output.reshape(41, 41, order='F')

                gt_matrix = neighbor_2_zero(gt_matrix, gt_x_pos-1, gt_y_pos-1)
                out_matrix = neighbor_2_zero(out_matrix, output_x_pos-1, output_y_pos-1)

                np_gt = gt_matrix.reshape(gt_matrix.size, order='F')
                np_output = out_matrix.reshape(out_matrix.size, order='F')


             
            temp_gt_mat = gt_mat
            for i in range(args.source_num):
                gt_mat = temp_gt_mat
                min_index, location_bias, Power_bias, temp_gt_mat = find_match_source(output_mat[i], temp_gt_mat)
                location_bias_list.append(location_bias)
                Power_bias_list.append(Power_bias)

                print(sample_name + "__" + str(idx + 1) + "___source_num_" + str(i) +  "___label_x={}\t label_y={}\t Power_label={}".format(
                                                            gt_mat[min_index][0], gt_mat[min_index][1], gt_mat[min_index][2]))

                print(sample_name + "__" + str(idx + 1) + "___source_num_" + str(i) + "___output_x={}\t output_y={}\t Power_output={}\t time={}".format(
                                                            output_mat[i][0], output_mat[i][1], output_mat[i][2], now_time))

                print(sample_name + "__" + str(idx + 1) + "___source_num_" + str(i) +  "___location_bias={}\t Power_bias={}".format(
                                                        location_bias, Power_bias))

        print("mean_location_bias_in_val={}\t mean_Power_bias_in_val={}\t time_for_mean={}".format(np.mean(location_bias_list), np.mean(Power_bias_list), np.mean(time_list)))


      

def main():
    args = parse_option()
    sys.stdout = Logger(args.results_dir + "log.txt")
    con = Config(args.config).getConfig()['base']
    _, _, _, _, args.frequencies = get_search_freq(con['N_total_samples'], con['scan_low_freq'], con['scan_high_freq'], con['fs'])

    model = set_model(args)
    print(model)
    # build data loader
    test_dataloader = set_loader(args, con)
    # test(test_dataloader, model, args, con)
    test_more_source(test_dataloader, model, args, con)

if __name__ == '__main__':
    
    main()