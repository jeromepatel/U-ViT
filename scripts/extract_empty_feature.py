import os
import sys
import importlib.util
import argparse
from tqdm import tqdm

# Add the parent directory to Python path to access libs module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['HF_HOME']="/export/data/jmakadiy/datasets/hf_cache"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import numpy as np
import libs.autoencoder
import libs.siglip2
import libs.stability_encoder

# Import MSCOCODatabase from local datasets.py file
datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets.py')
spec = importlib.util.spec_from_file_location("local_datasets", datasets_path)
local_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_datasets)
MSCOCODatabase = local_datasets.MSCOCODatabase
def main():
    prompts = [
        '',
    ]

    device = 'cuda'
    clip = libs.siglip2.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    save_dir = f'/localscratch/jmakadiy/coco256_features'
    latent = clip.encode(prompts)
    print(latent.shape)
    c = latent[0].detach().cpu().numpy()
    np.save(os.path.join(save_dir, f'empty_context.npy'), c)


if __name__ == '__main__':
    main()
