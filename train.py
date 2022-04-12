import argparse
import sys
import torch
import torch.backends.cudnn as cudnn
import pandas
import numpy as np
import os
import time
import math
from datasets.dataset import SoundDataset
from utils.utils import AverageMeter, save_model, Logger
from networks.damas_fista_net import DAMAS_FISTANet

def parse_option():
    parser = argparse.ArgumentParser(description='DAMAS_FISTANet for sound source in pytorch')

    parser.add_argument('--print_freq',
                            type=int,
                            default=1,
                            help='print frequency')
    parser.add_argument('--save_freq',
                            type=int,
                            default=10,
                            help='save frequency')
    parser.add_argument('--train_dir', 
                        help='The directory used to train the models',
                        default='./data/One_train.txt', type=str)
    parser.add_argument('--test_dir', 
                        help='The directory used to evaluate the models',
                        default='./data/One_test.txt', type=str)
    parser.add_argument('--label_dir', 
                        help='The directory used to evaluate the models',
                        default='D:/Ftp_Server/zgx/data/Acoustic-NewData/DAMAS_FISTA_Net_Label/', type=str)
    parser.add_argument('--save_folder', dest='save_folder',
                        help='The directory used to save the models',
                        default='./save_models/',
                        type=str)
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--val_epochs', default=20, type=int, metavar='N',
                        help='number of val epochs to run')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for dataloader')
    parser.add_argument('--learning_rate', '--learning-rate', default=0.01, type=float,
                            metavar='LR', help='initial learning rate')
    parser.add_argument('--MultiStepLR',
                            action='store_true',
                            help='using MultiStepLR')    
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)')
    parser.add_argument('--LayNo', 
                        default=5, 
                        type=int,
                        help='iteration nums')
    parser.add_argument('--scanning_area', default=[-2, 2], type=list, help='the size of imaging area')
    parser.add_argument('--scanning_resolution', default=0.1, type=float, help='the resolution of imaging area')
    parser.add_argument('--horizonal_distance', default=2.5, type=float, help='horizonal distance between microphone array and sound source')
    parser.add_argument('--micro_array_path', default='./data/56_spiral_array.mat', type=str, help='micro array path')
  


    args = parser.parse_args()
    record_time = time.localtime(time.time())

    return record_time, args


def set_loader(args):
    train_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.train_dir, args.label_dir, args.horizonal_distance, args.micro_array_path),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.test_dir, args.label_dir, args.horizonal_distance, args.micro_array_path),
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    return train_dataloader, test_dataloader



def set_model(args, wk_reshape, A):
    # 加载模型
    model = DAMAS_FISTANet(args.LayNo, wk_reshape, A)
    model.cuda()
    cudnn.benchmark = True

    return model


def set_optimizer(args, model):
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    return optimizer


def adjust_learning_rate(args, optimizer):
    # 定义学习率策略
    if args.MultiStepLR:
        args.learning_rate *= 0.95
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
            print('lr=', param_group['lr']) 
    



def train(train_dataloader, model, optimizer, epoch, args):
 
    model.train()

    losses = AverageMeter()

    for idx, (CSM, label, _) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            CSM = CSM.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        bsz = label.shape[0]

        # compute_loss
        output = model(CSM)

        loss = torch.sum((output - label) ** 2)
        
        losses.update(loss.item(), bsz)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if (idx + 1) % args.print_freq == 0:
           print('Train: [{0}][{1}/{2}]\t'
                  'loss={loss.val:.8f} \t'
                  'mean={loss.avg:.8f}\t'.format(
                      epoch,
                      idx + 1,
                      len(train_dataloader),
                      loss=losses
                      ))
        sys.stdout.flush()






def test(test_dataloader, model, args):
    """test"""

    model.eval()
    location_bias_list = list()
    Power_bias_list = list()
    time_list = list()
    scanning_area_X = np.arange(args.scanning_area[0], args.scanning_area[1] + args.scanning_resolution, args.scanning_resolution)
    scanning_area_Y = scanning_area_X

    with torch.no_grad():
        for idx, (CSM, label, _) in enumerate(test_dataloader):

            CSM = CSM.cuda(non_blocking=True)
            start_time = time.time()
            # forward
            output = model(CSM)

            end_time = time.time()

            np_gt = label.cpu().numpy()
            np_gt = np.squeeze(np_gt, 0)
            max_gt = max(np_gt)[0]
            np_gt = np.where(np_gt == max(np_gt))
            gt_x_pos = math.ceil((np_gt[0][0] + 1) / len(scanning_area_X))
            gt_y_pos = np.mod((np_gt[0][0] + 1), len(scanning_area_Y))
            gt_x = scanning_area_X[gt_x_pos - 1]
            gt_y = scanning_area_Y[gt_y_pos - 1]

            np_output = output.cpu().numpy()
            np_output = np.squeeze(np_output, 0)
            max_output = max(np_output)[0]
            np_output = np.where(np_output == max(np_output))
            output_x_pos = math.ceil((np_output[0][0] + 1) / len(scanning_area_X))
            output_y_pos = np.mod((np_output[0][0] + 1), len(scanning_area_Y))
            output_x = scanning_area_X[output_x_pos - 1]
            output_y = scanning_area_Y[output_y_pos - 1]


            Power_output = max_output
            Power_label = max_gt 
            Power_bias = np.abs(Power_output - Power_label)
           
            now_time = end_time - start_time
            location_bias = np.sqrt(((output_x - gt_x)**2 + (output_y - gt_y)**2))
            location_bias_list.append(location_bias)
            Power_bias_list.append(Power_bias)
            time_list.append(now_time)
           

            print(str(idx + 1) + "___label_x={}\t label_y={}\t Power_label={}".format(
                                                        gt_x, gt_y, Power_label))

            print(str(idx + 1) + "___output_x={}\t output_y={}\t Power_output={}\t time={}".format(
                                                        output_x, output_y, Power_output, now_time))

            print(str(idx + 1) + "___location_bias={}\t Power_bias={}".format(
                                                        location_bias, Power_bias))
        
        print("mean_location_bias_in_val={}\t mean_Power_bias_in_val={}\t time_for_mean={}".format(np.mean(location_bias_list), np.mean(Power_bias_list), np.mean(time_list)))









def main():
    record_time, args = parse_option()
    args.save_folder = args.save_folder + '{}/'.format(time.strftime('%m-%d-%H-%M', record_time))
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

    sys.stdout = Logger(args.save_folder + "log.txt")


    # read the A.csv and wk_reshape.csv
    A = pandas.read_csv('./data/A.csv', header=None)
    temp_A = np.array(A)
    A = torch.from_numpy(temp_A.astype(float))
    wk_reshape = np.genfromtxt('./data/wk_reshape.csv', dtype=complex, delimiter=',') 
    temp_w = np.array(wk_reshape)
    wk_reshape = torch.from_numpy(temp_w.astype(complex))

    wk_reshape = wk_reshape.cuda(non_blocking=True)
    A = A.cuda(non_blocking=True)

    # build data loader
    train_dataloader, test_dataloader = set_loader(args)

    # build model and criterion
    model = set_model(args, wk_reshape, A)

    # build optimizer
    optimizer = set_optimizer(args, model)

    print('===========================================')
    print('DAMAS_FISTA-Net...')
    print('===> Start Epoch {} End Epoch {}'.format(args.start_epoch, args.epochs))

    # training routine
    for epoch in range(1, args.epochs + 1):
        if args.MultiStepLR and epoch != 1: 
            adjust_learning_rate(args, optimizer)
      
        # train for one epoch
        time1 = time.time()
        train(train_dataloader, model, optimizer, epoch, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_folder, 'ckpt_epoch_{epoch}.pt'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

        # evaluation
        if epoch % args.val_epochs == 0:
            test(test_dataloader, model, args)

    # save the last model
    save_file = os.path.join(args.save_folder, 'last.pt')
    save_model(model, optimizer, args, args.epochs, save_file)

if __name__ == '__main__':
    main()
