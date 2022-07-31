from networks.resnet20 import resnet20
from networks.Alxnet import AlexNet
import torch
import argparse
import sys
from datasets.resnet20_dataset import SoundDataset as resnet_SoundDataset
from datasets.Alxnet_dataset import SoundDataset as AlxNet_SoundDataset
from train_baseline_resnet20 import test as resnet_test
from train_baseline_Alxnet import test as AlxNet_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='baselines resnet20 for sound source in pytorch')
    parser.add_argument('--test_dir', 
                        help='The directory used to evaluate the models',
                        default='../../data/generalization/generalization_One.txt', type=str)
                        # default='../../data/One_test.txt', type=str)
    parser.add_argument('--label_dir', 
                        help='The directory used to train the models',
                        # default='D:/Ftp_Server/zgx/data/Acoustic-NewData/generalization_one_data/One.txt', type=str)
                        default='', type=str)
    parser.add_argument('--DAS_results_dir', 
                        help='The directory which is saving DAS result',
                        default='', type=str)
                        # default='./DAS_results/One/', type=str)
    parser.add_argument('--horizonal_distance', default=2.5, type=float, help='horizonal distance between microphone array and sound source')
    parser.add_argument('--micro_array_path', default='D:/Ftp_Server/zgx/codes/Fast_DAS_2/MicArray/56_spiral_array.mat', type=str, help='micro array path')
    parser.add_argument('--scanning_area', default=[-2, 2], type=list, help='the size of imaging area')
    parser.add_argument('--scanning_resolution', default=0.1, type=float, help='the resolution of imaging area')
    parser.add_argument('--ckpt', default='', type=str, help='test model path')
    parser.add_argument('--mode', default=None, choices=['resnet20', 'AlxNet'], help='resnet20 or AlxNet')
    

    args = parser.parse_args()
  
    if args.mode == 'resnet20':
        test_dataloader = torch.utils.data.DataLoader(
            resnet_SoundDataset(args.test_dir, args.label_dir, args.DAS_results_dir, args.micro_array_path, args.horizonal_distance),
            batch_size=1, shuffle=True,
            num_workers=0, pin_memory=True)

        model = resnet20()
    
    elif args.mode == 'AlxNet':
        test_dataloader = torch.utils.data.DataLoader(
            AlxNet_SoundDataset(args.test_dir, args.label_dir, args.DAS_results_dir),
            batch_size=1, shuffle=True,
            num_workers=0, pin_memory=True)
        
        model = AlexNet()

    else:
        print("Input mode type error!")
        sys.exit(0)


    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        model.load_state_dict(state_dict)



if args.mode == 'resnet20':

    resnet_test(test_dataloader, model)
    
elif args.mode == 'AlxNet':

    AlxNet_test(test_dataloader, model, args)
