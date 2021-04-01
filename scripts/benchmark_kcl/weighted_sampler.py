#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:14:16 2021

@author: logan
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def make_sampler(arr):
    total = len(arr)
    frac_0 = total / np.sum(arr[:, -1] >= 37)
    weights = np.ones(len(arr)) * frac_0
    frac_1 = total / np.sum(arr[:, -1] < 32)
    frac_2 = total / (np.sum(arr[:, -1] < 37) - np.sum(arr[:, -1] <= 32))
    weights[np.where(arr[:, -1] < 32)] = frac_1
    weights[np.where(np.logical_and(arr[:, -1] < 37, arr[:, -1] >= 32))] = frac_2
    weights = np.tile(weights, 2)
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler


TRAIN_DIR = '/vol/biomedic3/aa16914/dhcp_brain_emma/data/age_prediction/birth_age/data_splits/train.npy'
train_set = np.load(TRAIN_DIR, allow_pickle=True)
sampler = make_sampler(train_set)
