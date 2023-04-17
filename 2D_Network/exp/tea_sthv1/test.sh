
#### scripts for evaluating TEA on Something-Something V1
### v=16

CUDA_VISIBLE_DEVICES=0,1 python main.py something RGB \
     --arch_file resnet_TEA \
     --arch resnet50 --num_segments 16 \
     --amp --gd 20 --lr 0.02 --lr_steps 30 40 45 --epochs 50 \
     --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --model_path 'models' \
     --shift --shift_div=8 --shift_place=blockres --npb --round test \
     --resume 'your_model_path' \
     --evaluate;