import warnings
import h5py
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append('../')
from utils.utils import data_preprocess, get_microphone_info, wgn
import glob
# ignore warnings
warnings.filterwarnings("ignore")


class SoundDataset(Dataset):
    def __init__(self, data_dir, label_dir, horizonal_distance, micro_array_path, wk_reshape_path, A_path, ATA_path, L_path, frequencies, yml_path, train=False, add_noise=False, dB_value=0, more_source=False):
        super(SoundDataset, self).__init__()
        self.data_name = []
        self.label_dir = label_dir
        self.data_dir = data_dir
        self.horizonal_distance = horizonal_distance
        self.micro_array_path = micro_array_path
        self.wk_reshape = np.load(wk_reshape_path) 
        self.A = np.load(A_path) 
        self.ATA = np.load(ATA_path) 
        self.L = np.load(L_path) 
        self.L = np.float32(self.L) 
        self.frequencies = frequencies
        self.train = train
        self.add_noise=add_noise
        self.dB_value=dB_value
        self.more_source=more_source
        self.yml_path=yml_path

        with open(self.data_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.data_name.append(line)

        _, self.mic_centre = get_microphone_info(self.micro_array_path)

        
    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):
        sample_name = self.data_name[index].split('/')[-1].split('.')[-2]
        f = h5py.File(self.data_name[index], 'r')
        raw_sound_data = np.array(f['time_data'].value)
        if self.add_noise:
            raw_sound_data += wgn(raw_sound_data, self.dB_value)

        gt_name = glob.glob(self.label_dir + sample_name + '_label_*.mat')[0]
        label = scio.loadmat(gt_name)['damas_fista_net_label']
        label = label.reshape(label.size, 1, order='F')

        x = np.float32(gt_name.split('_')[-3])
        y = np.float32(gt_name.split('_')[-2])
        freq_label = np.float32(gt_name.split('_')[-1].split('.mat')[0])


        CSM = data_preprocess(raw_sound_data, self.yml_path)

        if not self.more_source:
            
            microphone_center_to_source_distance = np.linalg.norm(np.array([x, y, self.horizonal_distance]) - self.mic_centre)
            label = (label / microphone_center_to_source_distance)**2
        
        if self.train:
            K = np.where(self.frequencies == freq_label)[0][0]
            CSM_K = CSM[:, :, K]
            wk_reshape_K = self.wk_reshape[:, :, K]
            A_K = self.A[:, :, K]
            ATA_K = self.ATA[:, :, K]
            L_K = np.array([[self.L[K]]])
            
            return CSM_K, wk_reshape_K, A_K, ATA_K, L_K, K, label, sample_name
                
        else:
            return CSM, self.wk_reshape, self.A, self.ATA, self.L, label, sample_name




if __name__ == '__main__':
    data_dir = '../data/One_train.txt'
    label_dir = 'D:/Ftp_Server/zgx/data/Acoustic-NewData/DAMAS_FISTA_Net_Label/'
    s = SoundDataset(data_dir, label_dir, 2.5, '../data/56_spiral_array.mat')
    s.__getitem__(2)