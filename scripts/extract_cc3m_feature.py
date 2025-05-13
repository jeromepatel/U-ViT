import torch
import os
import numpy as np
import sys
# sys.path.append("/weka/prior-default/georges/research/U-ViT/libs")
import libs.autoencoder
import libs.clip
import datasets
import argparse
from tqdm import tqdm
from PIL import Image

from datasets import CC3MDataset


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--shard', type=str, required=True)
    args = parser.parse_args()
    print(args)


    if args.split == "train":
        datas = CC3MDataset(
            path=f'/weka/oe-data-default/georges/datasets/cc3m-wds/cc3m-train-{{{args.shard}}}.tar',
            resolution=resolution
        )
        # save_dir = f'/weka/prior-default/georges/datasets/mscoco{resolution}_features/train'
        save_dir = f'/weka/oe-training-default/georges/datasets/cc3m{resolution}_featuresv3/train'
        # save_dir = f'/weka/oe-training-default/georges/datasets/mscoco/uvit/mscoco{resolution}_features/train'
    elif args.split == "val":
        datas = CC3MDataset(
            path='/weka/oe-data-default/georges/datasets/cc3m-wds/cc3m-validation-{'+ args.shard + '}.tar',
            resolution=resolution
        )
        # save_dir = f'/weka/prior-default/georgdes/datasets/mscoco{resolution}_features/val'
        save_dir = f'/weka/oe-training-default/georges/datasets/cc3m{resolution}_featuresv3/val'
        # save_dir = f'/weka/oe-training-default/georges/datasets/mscoco/uvit/mscoco{resolution}_features/val'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    os.makedirs(save_dir, exist_ok=True)

    autoencoder = libs.autoencoder.get_model('/weka/prior-default/georges/checkpoints/uvit_repo/assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            img, x, caption, key_ = data
            idx = key_

            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)
            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}.npy'), moments)
            # import pdb; pdb.set_trace()
            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join(save_dir, f'{idx}.png'))

            with open(os.path.join(save_dir, f'{idx}.txt'), 'w') as f:
                f.write(caption)

            latent = clip.encode([caption])
            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)


if __name__ == '__main__':
    main()
