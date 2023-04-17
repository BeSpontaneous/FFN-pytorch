
#### scripts for evaluating TSM_FFN on Something-Something V1
### v=16

CUDA_VISIBLE_DEVICES=0,1 python main_FFN.py something RGB \
     --arch_file resnet_TSM_FFN \
     --arch resnet50 --num_segments_H 16 \
     --amp --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
     --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --lambda_act 1 --model_path 'models_FFN' \
     --shift --shift_div=8 --shift_place=blockres --npb --round test \
     --resume 'your_model_path' \
     --evaluate;