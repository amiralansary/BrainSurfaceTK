#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reg_red50 \
--checkpoints_dir checkpoints/red50_12gb_ep70_lr_static_smallarch \
--name brains \
--ninput_edges 48735 \
--epoch_count 1 \
--norm group \
--num_aug 1 \
--verbose_plot \
--print_freq 10 \
--seed 0 \
--dataset_mode regression \
--niter 70 \
--niter_decay 0 \
--batch_size 1 \
--ncf 32 64 80 \
--fc_n 50 \
--pool_res 3000 2750 2500 \
--lr 0.0003 \
--init_type kaiming \