#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_test_mesh \
--checkpoints_dir checkpoints/testing \
--export_folder checkpoints/mesh_collapses \
--name brains \
--epoch_count 1 \
--norm group \
--num_aug 1 \
--verbose_plot \
--print_freq 10 \
--seed 0 \
--dataset_mode regression \
--niter 10 \
--niter_decay 0 \
--batch_size 1 \
--ncf 2 \
--pool_res 2500 \
--lr 0.0003 \
--init_type kaiming \
--num_groups 2 \