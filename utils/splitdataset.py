import random
import os


data_path = 'D:/Ftp_Server/zgx/data/Acoustic-NewData/'
split_save_path = '../data/'

data_name = []
# source data num is 4000, choose train:test = 7:3
train_num = 1400

for file_dir in os.listdir(data_path):
    for file in os.listdir(data_path + file_dir + '/'):
        if file.split('.')[-1] == 'txt':
            continue
        data_name.append(data_path + file_dir + '/' + file)

random.shuffle(data_name)

for i in range(len(data_name)):
    if i < train_num:
        with open(split_save_path + 'One_train.txt', 'a') as f:
            f.write(data_name[i] + '\n')
            print("train" + str(i))  
    else:
        with open(split_save_path + 'One_test.txt', 'a') as f:
            f.write(data_name[i] + '\n')  
            print("test" + str(i))  
