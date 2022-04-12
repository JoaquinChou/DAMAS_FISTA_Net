import warnings
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append('../../')
from utils.utils import get_microphone_info
warnings.filterwarnings("ignore")

class SoundDataset(Dataset):
    def __init__(self, data_dir, label_dir, DAS_results_dir, micro_array_path, horizonal_distance):
        super(SoundDataset, self).__init__()
        self.data_dir = data_dir
        self.data_name = []

        self.label_dir = label_dir
        self.gt = []

        self.DAS_results_dir = DAS_results_dir
        self.micro_array_path = micro_array_path
        self.horizonal_distance = horizonal_distance

        with open(self.data_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.data_name.append(line)

        with open(self.label_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.gt.append(line)

        _, self.mic_centre = get_microphone_info(self.micro_array_path)


        
    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):

        sample_name = self.data_name[index].split('/')[-1].split('.')[0]
        DAS_result = scio.loadmat(self.DAS_results_dir + sample_name + '.mat')['DAS_result']
        DAS_result = np.float32(DAS_result)
        DAS_result = DAS_result[np.newaxis, :, :]

        gt_index = int(sample_name.split('_')[-1]) - 1

        x = np.float32(self.gt[gt_index].split('_')[2])
        y = np.float32(self.gt[gt_index].split('_')[3])
        p = np.float32(self.gt[gt_index].split('_')[-1].replace('.h5', ''))

        microphone_center_to_source_distance = np.linalg.norm(np.array([x, y, self.horizonal_distance]) - self.mic_centre)
        p = p / microphone_center_to_source_distance

        return DAS_result, np.float32([x, y, p])
    


if __name__ == '__main__':
    data_dir = '../../../data/One_train.txt'
    label_dir = 'D:/Ftp_Server/zgx/data/Acoustic-NewData/data/One.txt'
    DAS_results_dir = '../DAS_results/One/'
    micro_array_path = 'D:/Ftp_Server/zgx/codes/Fast_DAS_2/MicArray/56_spiral_array.mat'
    horizonal_distance = 2.5
    s = SoundDataset(data_dir, label_dir, DAS_results_dir, micro_array_path, horizonal_distance)
    s.__getitem__(7)