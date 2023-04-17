python tools/run_net.py \
  --cfg configs/sth/SLOWFAST_16x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/sthv1 \
  DATA.PATH_PREFIX /datasets/something_v1/20bn-something-something-v1 \
  DATA.PATH_LABEL_SEPARATOR "," \
  DATA.NUM_FRAMES 64 \
  NUM_GPUS 4 \
  OUTPUT_DIR log_slowfast;