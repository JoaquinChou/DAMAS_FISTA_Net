# train resnet20
python train_baseline_resnet20.py --MultiStepLR

# train AlxNet
python train_baseline_Alxnet.py --MultiStepLR

# real data generalization for single sound source to resnet20
python test_baseline_generalization.py --mode resnet20 --ckpt ./save_models/04-04-21-21_resnet20_2/last.pt

# real data generalization for single sound source to AlxNet
python test_baseline_generalization.py --mode AlxNet --ckpt ./save_models/04-04-21-38_AlxNet/ckpt_epoch_120.pt --label_dir D:/Ftp_Server/zgx/data/Acoustic-NewData/generalization_one_data_VectorLabel/