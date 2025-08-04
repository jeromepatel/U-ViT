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


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--data_dir', default='/localscratch/jmakadiy/', type=str, help='Path to the dataset directory')
    parser.add_argument('--encoder_type', default='stability', choices=['original', 'stability'], 
                        help='Type of VAE encoder to use: original (U-ViT custom) or stability (Stability AI SD VAE EMA)')
    parser.add_argument('--vae_name', default='stabilityai/sd-vae-ft-ema', type=str,
                        help='Name of the Stability VAE model to use (only used when encoder_type=stability)')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size for processing (both image VAE encoding and text encoding)')
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

    device = "cuda"
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

    with torch.no_grad():
        # Initialize progress bar
        pbar = tqdm(total=len(datas), desc="Processing samples")
        
        # Process data in batches
        batch_data = []
        batch_indices = []
        
        for idx, data in enumerate(datas):
            try:
                # Try to access the data to catch corrupted images early
                if data is None:
                    print(f"Skipping sample {idx}: data is None")
                    pbar.update(1)
                    continue
                    
                batch_data.append(data)
                batch_indices.append(idx)
                
            except Exception as e:
                print(f"Skipping sample {idx} due to error: {e}")
                pbar.update(1)
                continue
            
            # Process when batch is full or at the end
            if len(batch_data) == args.batch_size or idx == len(datas) - 1:
                try:
                    # Prepare batch tensors
                    batch_images = []
                    batch_captions = []
                    
                    for raw_image, x, captions in batch_data:
                        if len(x.shape) == 3:
                            x = x[None, ...]  # Add batch dimension: (C,H,W) -> (1,C,H,W)
                        elif len(x.shape) == 4 and x.shape[0] == 1:
                            x = x.squeeze(0)  # Remove extra batch dim: (1,C,H,W) -> (C,H,W)
                            x = x[None, ...]  # Add it back: (C,H,W) -> (1,C,H,W)
                        
                        # Ensure tensor is float32
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x).float()
                        else:
                            x = x.float()
                        
                        batch_images.append(x)
                        batch_captions.extend(captions)  # Flatten captions from all samples
                    
                    # Stack images into batch tensor: [(1,C,H,W), (1,C,H,W)] -> (B,C,H,W)
                    batch_tensor = torch.cat(batch_images, dim=0).to(device)
                    
                    # Encode images with selected VAE
                    if args.encoder_type == 'original':
                        batch_moments = autoencoder(batch_tensor, fn='encode_moments')
                    else:  # stability
                        batch_moments = autoencoder.encode_moments(batch_tensor)
                    
                    # Save individual image encodings
                    for i, (batch_idx, moments) in enumerate(zip(batch_indices, batch_moments)):
                        moments_np = moments.detach().cpu().numpy()
                        np.save(os.path.join(save_dir, f'{batch_idx}.npy'), moments_np)
                    
                    # Encode text captions in batch
                    if batch_captions:
                        latent_batch = clip.encode(batch_captions)
                        
                        # Map caption encodings back to their samples
                        caption_idx = 0
                        for batch_idx, (_, _, captions) in zip(batch_indices, batch_data):
                            for i in range(len(captions)):
                                c = latent_batch[caption_idx].detach().cpu().numpy()
                                np.save(os.path.join(save_dir, f'{batch_idx}_{i}.npy'), c)
                                caption_idx += 1
                    
                    # Update progress bar
                    pbar.update(len(batch_data))
                        
                except Exception as e:
                    print(f"Error processing batch starting at sample {batch_indices[0]}: {e}")
                    # Try to process individual samples in the batch
                    for batch_idx, (raw_image, x, captions) in zip(batch_indices, batch_data):
                        try:
                            if len(x.shape) == 3:
                                x = x[None, ...]
                            
                            if isinstance(x, np.ndarray):
                                x = torch.from_numpy(x).float().to(device)
                            else:
                                x = x.float().to(device)
                            
                            # Encode single image
                            if args.encoder_type == 'original':
                                moments = autoencoder(x, fn='encode_moments').squeeze(0)
                            else:  # stability
                                moments = autoencoder.encode_moments(x)
                                if moments.dim() > 3:
                                    moments = moments.squeeze(0)
                            
                            moments_np = moments.detach().cpu().numpy()
                            np.save(os.path.join(save_dir, f'{batch_idx}.npy'), moments_np)
                            
                            # Encode text
                            latent = clip.encode(captions)
                            for i in range(len(latent)):
                                c = latent[i].detach().cpu().numpy()
                                np.save(os.path.join(save_dir, f'{batch_idx}_{i}.npy'), c)
                                
                            # Update progress for individual processing
                            pbar.update(1)
                                
                        except Exception as single_e:
                            print(f"Error processing individual sample {batch_idx}: {single_e}")
                            pbar.update(1)  # Still update progress even if failed
                            continue
                
                # Reset batch
                batch_data = []
                batch_indices = []
        
        # Close progress bar
        pbar.close()


if __name__ == '__main__':
    main()
