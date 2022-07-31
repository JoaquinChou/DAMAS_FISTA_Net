from utils import generate_A_and_wk_reshape_and_ATA_and_ATA_Eigenvalues
micro_array_path = '../data/56_spiral_array.mat'
save_A_and_wk_reshape_path = '../data/'
config_path = './config.yml'

if __name__ == '__main__':
    generate_A_and_wk_reshape_and_ATA_and_ATA_Eigenvalues(micro_array_path, save_A_and_wk_reshape_path, config_path)