# train
python train.py --save_freq 10 --val_epochs 20 --batch_size 64 --train_dir ./data/One_train.txt --test_dir ./data/One_test.txt --MultiStepLR --learning_rate 1e-3 --LayNo 4
# two source train
python train.py --save_freq 10 --val_epochs 20 --batch_size 64 --train_dir ./data/Two_train.txt --test_dir ./data/Two_test.txt --label_dir D:/Ftp_Server/zgx/data/two_point_DAMAS_FISTA_Net/two_point_data_label/ --MultiStepLR --learning_rate 1e-3 --source_num 2 --more_source

# single_source_DAS_result test
python vis.py --ckpt ./models/models_init_One_sound_data/ckpt_no_epoch.pth --test_dir ./data/One_test.txt --show_DAS_result

# single source model test
python vis.py --ckpt ./save_models/07-31-00-36/last.pt --test_dir ./data/One_test.txt --LayNo 5
# two source model test
python vis.py --ckpt ./save_models/07-31-00-36/last.pt --test_dir ./data/Two_test.txt --LayNo 5 --more_source --source_num 2 --label_dir D:/Ftp_Server/zgx/data/two_point_DAMAS_FISTA_Net/two_point_data_label/
python vis.py --ckpt ./save_models/08-01-10-53/last.pt --test_dir ./data/Two_test.txt --LayNo 5 --more_source --source_num 2 --label_dir D:/Ftp_Server/zgx/data/two_point_DAMAS_FISTA_Net/two_point_data_label/


python vis.py --ckpt ./save_models/07-27-08-51/last.pt --test_dir ./data/One_test.txt --add_noise --dB_value -10

# test for generalization 
python vis_genralization.py --ckpt ./save_models/07-31-00-36/last.pt --test_dir ./data/generalization/generalization_Two.txt
python vis_genralization.py --ckpt ./save_models/07-31-00-36/last.pt --test_dir ./data/generalization/generalization_One.txt