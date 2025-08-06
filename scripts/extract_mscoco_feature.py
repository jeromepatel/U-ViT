import os
import sys
import importlib.util
import argparse
from tqdm import tqdm
from PIL import Image

# Add the parent directory to Python path to access libs module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['HF_HOME']="/export/data/jmakadiy/datasets/hf_cache"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import libs.autoencoder
import libs.siglip2
import libs.stability_encoder
from numpy import einsum

from utils import load_encoders
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


# Import MSCOCODatabase from local datasets.py file
datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets.py')
spec = importlib.util.spec_from_file_location("local_datasets", datasets_path)
local_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_datasets)
MSCOCODatabase = local_datasets.MSCOCODatabase

debug = False

def preprocess_raw_image(x, enc_type, resolution=256):
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')

    return x

def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--data_dir', default='/localscratch/jmakadiy/', type=str, help='Path to the dataset directory')
    parser.add_argument('--encoder_type', default='stability', choices=['original', 'stability'], 
                        help='Type of VAE encoder to use: original (U-ViT custom) or stability (Stability AI SD VAE EMA)')
    parser.add_argument('--vae_name', default='stabilityai/sd-vae-ft-ema', type=str,
                        help='Name of the Stability VAE model to use (only used when encoder_type=stability)')
    parser.add_argument('--enc_type', default='dinov2-vit-b', type=str,
                        help='Type of encoder to use for image features')
    args = parser.parse_args()
    print(args)


    if args.split == "train":
        datas = MSCOCODatabase(root=f'{args.data_dir}/coco2014/train2014',
                             annFile=f'{args.data_dir}/coco2014/annotations/captions_train2014.json',
                             size=resolution)
        save_dir = f'{args.data_dir}/coco{resolution}_features/train'
    elif args.split == "val":
        datas = MSCOCODatabase(root=f'{args.data_dir}/coco2014/val2014',
                             annFile=f'{args.data_dir}/coco2014/annotations/captions_val2014.json',
                             size=resolution)
        save_dir = f'{args.data_dir}/coco{resolution}_features/val'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(save_dir, exist_ok=True)

    # Choose encoder type
    if args.encoder_type == 'original':
        print("Using original U-ViT autoencoder")
        autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
        autoencoder.to(device)
    elif args.encoder_type == 'stability':
        print(f"Using Stability AI VAE: {args.vae_name}")
        autoencoder = libs.stability_encoder.get_stability_vae_model(
            vae_name=args.vae_name
        )
        # Initialize the encoder on the device
        autoencoder.init(device)
    else:
        raise ValueError(f"Unknown encoder type: {args.encoder_type}")

    # Use SigLIP2 for text encoding
    print("Using SigLIP2 text encoder")
    clip = libs.siglip2.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    # Load the encoder for image features
    encoders, encoder_types, architectures = load_encoders(args.enc_type, device)

    encoder = encoders[0]  # we only use one encoder for now
    encoder_type = encoder_types[0]
    architecture = architectures[0]
    # Process one image at a time
    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas), total=len(datas), desc="Processing samples"):
            try:
                raw_img, x, captions = data
                # print(f"raw image shape: {raw_img.shape}, type: {type(raw_img)}")
                # prepare single‐image tensor
                if len(x.shape) == 3:
                    x = x[None, ...]
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float()
                else:
                    x = x.float()
                x = x.to(device)

                # encode image moments
                if args.encoder_type == 'original':
                    moments = autoencoder(x, fn='encode_moments').squeeze(0)
                else:
                    moments = autoencoder.encode_moments(x)
                    if moments.dim() > 3:
                        moments = moments.squeeze(0)
                # print(f"moments shape: {moments.shape}, type: {type(moments)}")
                # print(f"raw image shape: {raw_img.shape}, type: {type(raw_img)}")
                # save image latent
                # np.save(os.path.join(save_dir, f'{idx}.npy'),
                #         moments.detach().cpu().numpy())
                # save original image dinov2 features
                #temp save raw image and reconstructed image to a file  
                if debug:
                    temp_save_path = '/export/data/jmakadiy/temp_raw_reconstructed'
                    os.makedirs(temp_save_path, exist_ok=True)
                    raw_img_save_path = os.path.join(temp_save_path, f'{idx}_raw.png')
                    #convert to PIL image and save
                    if isinstance(raw_img, np.ndarray):
                        # change from CHW to HWC
                        raw_img = np.transpose(raw_img, (1, 2, 0))
                    elif isinstance(raw_img, torch.Tensor):
                        # change from CHW to HWC and move to CPU
                        raw_img = raw_img.cpu().permute(1, 2, 0).numpy()
                    # now raw_img is H x W x C and can be cast to uint8
                    raw_img = Image.fromarray(raw_img.astype(np.uint8)).convert('RGB')
                    reconstructed_img_save_path = os.path.join(temp_save_path, f'{idx}_reconstructed.png')

                    # Save raw image
                    raw_img.save(raw_img_save_path)
                    if args.encoder_type == 'stability':
                        moments = moments.unsqueeze(0)  # Ensure batch dimension is present
                        mean, logvar = torch.chunk(moments, 2, dim=1)
                        print(f"mean shape: {mean.shape}, logvar shape: {logvar.shape}")
                        logvar = torch.clamp(logvar, -30.0, 20.0)
                        std = torch.exp(0.5 * logvar)
                        z = (mean + std * torch.randn_like(mean))
                        print(f"z shape: {z.shape}, type: {type(z)}")
                        reconstructed_img = autoencoder.decode(z)
                        if isinstance(reconstructed_img, torch.Tensor):
                            reconstructed_img = reconstructed_img.squeeze(0).cpu().numpy()
                        #reconstruct the image and save it
                        reconstructed_img = np.transpose(reconstructed_img, (1, 2, 0))
                        reconstructed_img = (reconstructed_img + 1.0) / 2.0
                        reconstructed_img = Image.fromarray((reconstructed_img * 255).astype(np.uint8)).convert('RGB')
                        reconstructed_img.save(reconstructed_img_save_path)

                    if idx == 5:
                        break
                    else:
                        continue

                if isinstance(raw_img, np.ndarray):
                    raw_img = torch.from_numpy(raw_img).float()

                #print(f"raw image shape after conversion: {raw_img.shape}, type: {type(raw_img)}")
                raw_img = raw_img.unsqueeze(0) if raw_img.dim() == 3 else raw_img
                raw_img = raw_img.to(device)
                

                processed_img = preprocess_raw_image(raw_img, encoder_type, resolution)
                # print(f"shape of final raw image: {processed_img.shape}, type: {type(processed_img)}")
                z = encoder.forward_features(processed_img)

                #concatenate the x_norm_clstoken and x_norm_patchtokens
                z_final = torch.cat((z['x_norm_clstoken'], z['x_norm_patchtokens'].squeeze(0)), dim=0)

                # print("dimension of z_final:", z_final.shape)
                # Save the processed image features
                np.save(os.path.join(save_dir, f'{idx}_dinov2.npy'), z_final.detach().cpu().numpy())
                
                #print(f"{z.keys()}, {z['x_norm_clstoken'].shape}, image shape, {z['x_norm_patchtokens'].shape}, image type: {type(processed_img)}")

                # encode and save captions
                latent = clip.encode(captions)
                for i, c_tensor in enumerate(latent):
                    np.save(os.path.join(save_dir, f'{idx}_{i}.npy'),
                            c_tensor.detach().cpu().numpy())
            except Exception as e:

                print(f"Error processing sample {idx}: {e}")
                continue

if __name__ == '__main__':
    main()

''''
Output from above with dimensions:
loading annotations into memory...
Done (t=0.41s)
creating index...
index created!
Using device: cuda
Using Stability AI VAE: stabilityai/sd-vae-ft-ema
Using SigLIP2 text encoder
Processing samples:   0%| 0/82783 [00:00<?, ?it/s]raw image shape: (3, 256, 256), type: <class 'numpy.ndarray'>
raw image shape after conversion: torch.Size([3, 256, 256]), type: <class 'torch.Tensor'>
shape of final raw image: torch.Size([1, 3, 224, 224]), type: <class 'torch.Tensor'>
dict_keys(['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'masks']), torch.Size([1, 768]), image shape, torch.Size([1, 256, 768]), image type: <class 'torch.Tensor'>                                                                                                                                                                             | 0/82783 [00:00<?, ?it/s]
dimension of z_final: torch.Size([257, 768])         
'''