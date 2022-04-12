# train
python train.py --save_freq 2 --val_epochs 6 --batch_size 128 --train_dir ./data/One_train.txt --test_dir ./data/One_test.txt --MultiStepLR --learning_rate 1e-3

# single_source_DAS_result test
python vis.py --ckpt ./models/models_init_One_sound_data/ckpt_no_epoch.pth --test_dir ./data/One_test.txt --show_DAS_result

# single source model test
python vis.py --ckpt ./save_models/04-06-22-29/last.pt --test_dir ./data/One_test.txt 

# test for generalization 
python vis_genralization.py --ckpt ./save_models/04-06-22-29/last.pt --test_dir ./data/generalization/generalization_Two.txt
