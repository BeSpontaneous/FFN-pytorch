
#### scripts for training TSM on Kinetics400
### v=16

CUDA_VISIBLE_DEVICES=0,1 python main.py kinetics_torrent RGB \
     --arch_file resnet_TSM \
     --arch resnet50 --num_segments 16 \
     --amp --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --model_path 'models' \
     --shift --shift_div=8 --shift_place=blockres --npb --round 1;