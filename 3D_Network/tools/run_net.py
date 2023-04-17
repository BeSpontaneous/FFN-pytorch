#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_net import test
from train_net import train


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        if cfg.TEST.NUM_ENSEMBLE_VIEWS == -1:
            num_view_list = [1, 3, 5, 7, 10]
            for num_view in num_view_list:
                cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view
                launch_job(cfg=cfg, init_method=args.init_method, func=test)
        else:
            launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
