import warnings
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
import glob
warnings.filterwarnings("ignore")

class SoundDataset(Dataset):
    def __init__(self, data_dir, label_dir, DAS_results_dir):
        super(SoundDataset, self).__init__()
        self.data_dir = data_dir
        self.data_name = []

        self.label_dir = label_dir
        self.gt = []

        self.DAS_results_dir = DAS_results_dir

        with open(self.data_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.data_name.append(line)


        
    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):

        sample_name = self.data_name[index].split('/')[-1].split('.')[0]
        # print(sample_name)
        DAS_result = scio.loadmat(self.DAS_results_dir + sample_name + '.mat')['DAS_result']
        DAS_result = np.float32(DAS_result)
        DAS_result = DAS_result[np.newaxis, :, :]

        gt_name = glob.glob(self.label_dir + sample_name + '_label_*.mat')[0]
        gt = scio.loadmat(gt_name)['vector_label']
        gt = np.where(gt == 1)[0][0]       

        return DAS_result, gt
    


if __name__ == '__main__':
    data_dir = '../../../data/One_train.txt'
    label_dir = 'D:/Ftp_Server/zgx/data/Acoustic-NewData/VectorLabel/'
    DAS_results_dir = '../DAS_results/One/'
    s = SoundDataset(data_dir, label_dir, DAS_results_dir)
    s.__getitem__(10)