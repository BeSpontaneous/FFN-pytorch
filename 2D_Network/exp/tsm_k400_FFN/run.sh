
#### scripts for training TSM_FFN on Kinetics400
### v_h=16, v_m=8, v_l=4

CUDA_VISIBLE_DEVICES=0,1 python main_FFN.py kinetics_torrent RGB \
     --arch_file resnet_TSM_FFN \
     --arch resnet50 --num_segments_H 16 --num_segments_M 8 --num_segments_L 4 \
     --amp --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --lambda_act 1 --model_path 'models_FFN' \
     --shift --shift_div=8 --shift_place=blockres --npb --round 1;