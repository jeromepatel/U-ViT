import torch
import torch.nn as nn
from transformers import AutoTokenizer, Siglip2TextModel


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenSigLIP2Embedder(AbstractEncoder):
    """Uses the SigLIP2 transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="google/siglip2-base-patch16-256", device="cuda", max_length=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version, model_max_length=max_length)
        self.transformer = Siglip2TextModel.from_pretrained(version, torch_dtype=torch.float32)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode_batch(self, texts):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            # print(texts)
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            # Move inputs to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1)
    
    def forward(self, text):
        # Handle both single strings and lists of strings
        if isinstance(text, str):
            text = [text]
            
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, 
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state.float()  # Ensure float32 output
        return z

    def encode(self, text):
        return self(text)


# Backward compatibility alias - keep the original CLIP class name but use SigLIP2
class FrozenCLIPEmbedder(FrozenSigLIP2Embedder):
    """Backward compatibility: Uses SigLIP2 but keeps CLIP interface"""
    pass
