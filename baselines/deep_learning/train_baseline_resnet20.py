import argparse
import sys
import torch
import numpy as np
import os
import time
from datasets.resnet20_dataset import SoundDataset
import sys
sys.path.append('../../')
from utils.utils import save_model, Logger, AverageMeter
from networks.resnet20 import resnet20
import math

def parse_option():
    parser = argparse.ArgumentParser(description='baselines resnet20 for sound source in pytorch')

    parser.add_argument('--print_freq',
                            type=int,
                            default=5,
                            help='print frequency')
    parser.add_argument('--save_freq',
                            type=int,
                            default=20,
                            help='save frequency')
    parser.add_argument('--train_dir', 
                        help='The directory used to train the models',
                        default='../../data/One_train.txt', type=str)
    parser.add_argument('--test_dir', 
                        help='The directory used to evaluate the models',
                        default='../../data/One_test.txt', type=str)
    parser.add_argument('--label_dir', 
                        help='The directory used to train the models',
                        default='D:/Ftp_Server/zgx/data/Acoustic-NewData/data/One.txt', type=str)
    parser.add_argument('--DAS_results_dir', 
                        help='The directory used to train the models',
                        default='./DAS_results/One/', type=str)
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
    parser.add_argument('--MultiStepLR', action='store_true', help='using MultiStepLR')    
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)')
    parser.add_argument('--horizonal_distance', default=2.5, type=float, help='horizonal distance between microphone array and sound source')
    parser.add_argument('--micro_array_path', default='D:/Ftp_Server/zgx/codes/Fast_DAS_2/MicArray/56_spiral_array.mat', type=str, help='micro array path')
  

    args = parser.parse_args()
    record_time = time.localtime(time.time())

    return record_time, args


# 加载声源数据
def set_loader(args):
    train_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.train_dir, args.label_dir, args.DAS_results_dir, args.micro_array_path, args.horizonal_distance),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.test_dir, args.label_dir, args.DAS_results_dir, args.micro_array_path, args.horizonal_distance),
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    return train_dataloader, test_dataloader



def set_model():
    # 加载模型
    model = resnet20()
    model.cuda()

    return model


def set_optimizer(args, model):
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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

    for idx, (DAS_result, gt) in enumerate(train_dataloader):
        bsz = DAS_result.shape[0]
        if torch.cuda.is_available():
            DAS_result = DAS_result.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)

        output = model(DAS_result)

        # compute_loss
        loss = torch.mean((output- gt)** 2)
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






def test(test_dataloader, model):
    """test"""

    model.eval()
    location_bias_list = list()
    Power_bias_list = list()
    time_list = list()

    with torch.no_grad():
        for idx, (DAS_result, gt) in enumerate(test_dataloader):
            start_time = time.time()

            DAS_result = DAS_result.cuda(non_blocking=True)
            output = model(DAS_result)

            
            end_time = time.time()
            now_time = end_time - start_time
            np_output = output.cpu().numpy()
            np_output = np.squeeze(np_output, 0)
            gt = np.squeeze(gt.numpy(), 0)
            location_bias = np.sqrt(((np_output[0] - gt[0])**2 + (np_output[1] - gt[1])**2))

            # transform to SPL
            # SPL_output = 20 * math.log(np_output[2] / 2e-5, 10)
            # SPL_label = 20 * math.log(gt[2] / 2e-5, 10)
            # SPL_bias = np.abs(SPL_label - SPL_output)
            np_output[2] = np_output[2] ** 2
            gt[2] = gt[2] ** 2
            Power_bias = np.abs(np_output[2] - gt[2])

            time_list.append(now_time)
            location_bias_list.append(location_bias)
            Power_bias_list.append(Power_bias)


            print(str(idx + 1) + "___label_x={}\t label_y={}\t Power_label={}".format(
                                                        gt[0], gt[1], gt[2]))

            print(str(idx + 1) + "___output_x={}\t output_y={}\t Power_output={}\t time={}".format(
                                                        np_output[0], np_output[1], np_output[2], now_time))

            print(str(idx + 1) + "___location_bias={}\t Power_bias={}".format(
                                                        location_bias, Power_bias))
        
        print("mean_location_bias_in_val={}\t mean_Power_bias_in_val={}\t time_for_mean={}".format(np.mean(location_bias_list), np.mean(Power_bias_list), np.mean(time_list[1:])))







def main():
    record_time, args = parse_option()
    args.save_folder = args.save_folder + '{}/'.format(time.strftime('%m-%d-%H-%M', record_time))
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

    sys.stdout = Logger(args.save_folder + "log.txt")

    # build data loader
    train_dataloader, test_dataloader = set_loader(args)

    # build model and criterion
    model = set_model()

    # build optimizer
    optimizer = set_optimizer(args, model)

    print('===========================================')
    print('ResNet20...')
    print('===> Start Epoch {} End Epoch {}'.format(args.start_epoch, args.epochs))

    # training routine
    for epoch in range(1, args.epochs + 1):
        if args.MultiStepLR and (epoch != 1):
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
            test(test_dataloader, model)

    # save the last model
    save_file = os.path.join(args.save_folder, 'last.pt')
    save_model(model, optimizer, args, args.epochs, save_file)







if __name__ == '__main__':
    main()
