python tools/run_net_FFN.py \
  --cfg configs/sth/SLOWFAST_16x8_R50_FFN.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/sthv1 \
  DATA.PATH_PREFIX /datasets/something_v1/20bn-something-something-v1 \
  DATA.PATH_LABEL_SEPARATOR "," \
  DATA.NUM_FRAMES_H 64 \
  DATA.NUM_FRAMES_M 32 \
  DATA.NUM_FRAMES_L 16 \
  TRAIN.LAMBDA 1.0 \
  NUM_GPUS 4 \
  OUTPUT_DIR log_slowfast_ffn;