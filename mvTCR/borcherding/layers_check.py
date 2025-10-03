# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 14:55:24 2025

@author: a4945
"""

import torch
import csv
from collections import OrderedDict

def load_state_dict_from_checkpoint(path, candidate_names=('state_dict','model','model_state_dict','state')):
    """
    Load a checkpoint file and return the contained state_dict (plain dict of tensors).
    Tries common wrapper keys; otherwise assumes the file *is* a state_dict.
    """
    ckpt = torch.load(path, map_location='cpu')
    if not isinstance(ckpt, dict):
        # unexpected but return as-is
        return ckpt
    # check candidate wrappers
    for name in candidate_names:
        if name in ckpt and isinstance(ckpt[name], dict):
            return ckpt[name]
    # sometimes ckpt might contain keys like 'net' or be exactly the state dict
    # fallback: try to find the first dict-like value that looks like a state_dict
    for v in ckpt.values():
        if isinstance(v, dict):
            # a heuristic: check if dict values look like tensors / ndarrays
            sample_vals = list(v.values())[:5]
            if all(hasattr(x, 'shape') for x in sample_vals):
                return v
    # assume ckpt itself is a state_dict
    return ckpt

def inspect_checkpoint(path, max_items=None, print_shapes=True):
    """
    Print (and return) ordered mapping key -> shape for the checkpoint.
    """
    sd = load_state_dict_from_checkpoint(path)
    ordered = OrderedDict()
    for i, (k, v) in enumerate(sd.items()):
        try:
            shape = tuple(v.shape)
        except Exception:
            # fallback for non-tensor values
            try:
                shape = (len(v),)
            except Exception:
                shape = None
        ordered[k] = shape
        if print_shapes:
            print(f"{k}: {shape}")
        if max_items and i+1 >= max_items:
            break
    return ordered


ckpt_1 = inspect_checkpoint('./borcherding/beta/borcherding_beta_split_0_moe.pt')

ckpt_2 = inspect_checkpoint('./borcherding/beta/borcherding_beta_split_1_moe.pt')




