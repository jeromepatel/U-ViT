"""Converting between pixel and latent representations using Stability AI EMA VAE for SDXL."""

import os
import warnings
import numpy as np
import torch
from diffusers.models import AutoencoderKL

# Silence deprecation warnings
warnings.filterwarnings('ignore', 'torch.utils._pytree._register_pytree_node is deprecated.')
warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Abstract base class for encoders/decoders that convert back and forth
# between pixel and latent representations of image data.

class Encoder:
    def __init__(self):
        pass

    def init(self, device):
        # Hook for lazy initialization on a given device
        pass

    def __getstate__(self):
        return self.__dict__

    def encode_pixels(self, x):  # raw pixels => raw latents
        raise NotImplementedError  # to be overridden by subclass

#----------------------------------------------------------------------------
# Pre-trained EMA VAE encoder for SD.

class StabilityVAEEncoder(Encoder):
    def __init__(self,
        vae_name: str = 'stabilityai/sd-vae-ft-ema',  # EMA checkpoint repo
    ):
        super().__init__()
        self.vae_name = vae_name
        self._vae = None

    def init(self, device):  # force lazy init to happen now
        super().init(device)
        if self._vae is None:
            self._vae = load_stability_ema_vae(self.vae_name, device=device)
        else:
            self._vae.to(device)

    def __getstate__(self):
        state = super().__getstate__()
        # Do not pickle the VAE weights
        state['_vae'] = None
        return state

    def encode_moments(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to latent moments (mean and std concatenated)."""
        self.init(x.device)
        # Input should already be normalized to [-1,1] from MSCOCO dataset preprocessing
        x = x.to(torch.float32)
        
        # Encode directly - supports both single images and batches
        d = self._vae.encode(x)['latent_dist']
        # Concatenate mean and std for downstream tasks, ensure float32
        mean = d.mean.to(torch.float32)
        std = d.std.to(torch.float32)
        return torch.cat([mean, std], dim=1)

#----------------------------------------------------------------------------

def load_stability_ema_vae(
    vae_name: str = 'stabilityai/sd-vae-ft-ema',
    device: torch.device = torch.device('cpu'),
) -> AutoencoderKL:
    """Load the EMA VAE model from HuggingFace for SDXL."""
    # Suppress HF progress bars and warnings
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

    # Load the VAE from the specified repo
    vae = AutoencoderKL.from_pretrained(
        vae_name,
        torch_dtype=torch.float32,
    )

    # Ensure the model is in float32 and moved to device
    return vae.eval().requires_grad_(False).to(device).float()

#----------------------------------------------------------------------------

def get_stability_vae_model(
    use_ema: bool = True,
    custom_repo: str = None,
    vae_name: str = None,
) -> StabilityVAEEncoder:
    """Factory: returns an EMA-based SDXL VAE encoder by default."""
    if vae_name:
        repo = vae_name
    elif custom_repo:
        repo = custom_repo
    else:
        repo = 'stabilityai/sd-vae-ft-ema' if use_ema else 'stabilityai/sd-vae-ft-mse'
    return StabilityVAEEncoder(vae_name=repo)