import warnings
import h5py
import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append('../')
from utils.utils import data_preprocess, get_magnitude
# ignore warnings
import scipy.io as scio
warnings.filterwarnings("ignore")


class SoundDataset(Dataset):
    def __init__(self, data_dir, simulation_single_sound_source_data, wk_reshape_path, A_path, ATA_path, L_path, yml_path):
        super(SoundDataset, self).__init__()
        self.data_name = []
        self.data_dir = data_dir
        self.simulation_data = h5py.File(simulation_single_sound_source_data, 'r')
        self.simulation_data = np.array(self.simulation_data['time_data'].value)
        self.wk_reshape = np.load(wk_reshape_path) 
        self.A = np.load(A_path) 
        self.ATA = np.load(ATA_path) 
        self.L = np.load(L_path) 
        self.L = np.float32(self.L) 
        self.yml_path=yml_path

        with open(self.data_dir, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    continue
                line = line.strip('\n')
                self.data_name.append(line)

        
    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):

        # print(self.data_name[index])

        sample_name = self.data_name[index].split('/')[-1].split('.')[-2]

        if self.data_name[index].split('/')[-1].split('.')[-1] == 'h5':
            f = h5py.File(self.data_name[index], 'r')
            raw_sound_data = np.array(f['time_data'].value)

            # magnitude = get_magnitude(raw_sound_data, self.simulation_data)
            # print(magnitude)
            magnitude = 1
            raw_sound_data = raw_sound_data * magnitude

        elif self.data_name[index].split('/')[-1].split('.')[-1] == 'mat':
            # one_source--key is ['one_data']
            # two_source--key is ['two_data']
            # f = scio.loadmat(self.data_name[index])['one_data']
            f = scio.loadmat(self.data_name[index])['two_data']
            # f = scio.loadmat(self.data_name[index])['four_data']
            raw_sound_data = np.transpose(f, (1, 0))
            magnitude = get_magnitude(raw_sound_data, self.simulation_data)
            # print(magnitude)
            raw_sound_data = raw_sound_data * magnitude
            raw_sound_data = np.float32(raw_sound_data)

        CSM = data_preprocess(raw_sound_data, self.yml_path)
     
        return CSM, self.wk_reshape, self.A, self.ATA, self.L, sample_name

if __name__ == '__main__':
    data_dir = '../data/generalization/generalization.txt'
    simulation_single_sound_source_data = 'D:/Ftp_Server/zgx/data/Acoustic-NewData/data/One_687.h5'
    s = SoundDataset(data_dir, simulation_single_sound_source_data)
    s.__getitem__(1)