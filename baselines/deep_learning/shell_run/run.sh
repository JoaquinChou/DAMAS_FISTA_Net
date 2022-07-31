python array_signal_2_das_result.py --add_noise --dB_value 0 --save_DAS_results_path ./DAS_results_0dB/

python train_baseline_resnet20.py --MultiStepLR

python train_baseline_Alxnet.py --MultiStepLR



# simulation data for single sound source
python test_baseline_generalization.py --mode resnet20 --ckpt ./save_models/04-04-21-21_resnet20_2/last.pt --label_dir D:/Ftp_Server/zgx/data/Acoustic-NewData/data/One.txt --DAS_results_dir ./DAS_results_0dB/One/

# generalization to real data
python test_baseline_generalization.py --mode resnet20 --ckpt ./save_models/04-04-21-21_resnet20_2/last.pt --label_dir D:/Ftp_Server/zgx/data/Acoustic-NewData/generalization_one_data/One.txt --DAS_results_dir ./generalization_data_DAS_results/One/


# simulation data for single sound source
python test_baseline_generalization.py --mode AlxNet --ckpt ./save_models/04-04-21-38_AlxNet/ckpt_epoch_120.pt --label_dir D:/Ftp_Server/zgx/data/Acoustic-NewData/VectorLabel/  --DAS_results_dir ./DAS_results_0dB/One/
# generalization to real data
python test_baseline_generalization.py --mode AlxNet --ckpt ./save_models/04-04-21-38_AlxNet/ckpt_epoch_120.pt --label_dir D:/Ftp_Server/zgx/data/Acoustic-NewData/generalization_one_data_VectorLabel/  --DAS_results_dir ./generalization_data_DAS_results/One/