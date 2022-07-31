import random
import os


# data_path = 'D:/Ftp_Server/zgx/data/Acoustic-NewData/data/'
data_path = 'D:/Ftp_Server/zgx/data/two_point_DAMAS_FISTA_Net/two_point_data'
split_save_path = '../data/'

data_name = []
# source data num is 4000, choose train:test = 7:3
train_num = 1400

# for file_dir in os.listdir(data_path):
for file in os.listdir(data_path + '/'):
    if file.split('.')[-1] == 'txt':
        continue
    data_name.append(data_path + '/' + file)

random.shuffle(data_name)

for i in range(len(data_name)):
    if i < train_num:
        with open(split_save_path + 'Two_train.txt', 'a') as f:
            f.write(data_name[i] + '\n')
            print("train" + str(i))  
    else:
        with open(split_save_path + 'Two_test.txt', 'a') as f:
            f.write(data_name[i] + '\n')  
            print("test" + str(i))  
