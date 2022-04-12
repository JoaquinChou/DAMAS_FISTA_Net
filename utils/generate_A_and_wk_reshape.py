from utils import generate_A_and_wk_reshape

micro_array_path = '../data/56_spiral_array.mat'
save_A_and_wk_reshape_path = '../data/'
# in our experiment, sample num is 1024
N_total_samples = 1024

if __name__ == '__main__':
    generate_A_and_wk_reshape(micro_array_path, save_A_and_wk_reshape_path, N_total_samples)