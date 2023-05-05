import open_clip
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch

import openvino.runtime as ov


def ln(x, b, s):
    _eps = 1e-5
    output = (x - x.mean(1)[:,None]) / np.sqrt((x.var(1)[:,None] + _eps))
    output = s[None, :] * output + b[None,:]
    return output

class OpenVINOCLIP():
    IMAGE_ENCODER_MODEL = "image_encoder.xml"
    TEXT_ENCODER_MODEL = "text_encoder.xml"
    
    def __init__(
                self, 
                model_folder,
                tokenizer,
                attn_mask,
                token_embedding,
                positional_embedding,
                ln_final,
                text_projection,
                cache_dir: str = None,
                device="cpu"):
        self.model_folder = model_folder
        self.tokenizer = tokenizer
        self.attn_mask = attn_mask
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.ln_final = ln_final
        self.text_projection = text_projection
        self.cache_dir = cache_dir
        self.device = device
        
        self._compile_models()
        
    def _compile_models(self):
        core = ov.Core()
        image_encoder = core.read_model(Path(self.model_folder) / self.IMAGE_ENCODER_MODEL)
        self.image_encoder = core.compile_model(image_encoder, self.device)
        
        text_encoder = core.read_model(Path(self.model_folder) / self.TEXT_ENCODER_MODEL)
        self.text_encoder = core.compile_model(text_encoder, self.device)        
        
    def encode_image(self, image, normalize: bool = False):
        features = self.image_encoder(image)
        features = torch.from_numpy(features[0])
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        x = self.text_encoder(text)
        x = torch.from_numpy(x[0])
        return F.normalize(x, dim=-1) if normalize else x

    def __call__(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()
    
    def eval(self):
        pass
        
    def to(self, device):
        self.device = device # TODO recompilation

def load_openvino_clip(model_name: str, pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    """Load and build OpenVINO-based CLIP model exported from OpenCLIP.

    Args:
        model_name (str, optional): Folder with image_encoder.xml/bin, text_encoder.xml/bin and 
                                      model_index.txt.
        cache_dir (str, optional): Defaults to None.
        device (str, optional): Defaults to "cpu".

    Returns:
        _type_: _description_
    """
    MODEL_DESCRIPTION = "model_index.txt"
    with open(Path(model_name) / MODEL_DESCRIPTION) as file:
        lines = [line.rstrip() for line in file]
        orig_model_name, pretrained = lines[0].split(",")
        print(f"model {orig_model_name} with pretrained {pretrained}")
    
    torch_model, _, transform = open_clip.create_model_and_transforms(orig_model_name, pretrained=pretrained, cache_dir=cache_dir)
    tokenizer = open_clip.get_tokenizer(orig_model_name)
    
    model = OpenVINOCLIP(
                        model_name, 
                        tokenizer, 
                        torch_model.attn_mask, 
                        torch_model.token_embedding, 
                        torch_model.positional_embedding, 
                        torch_model.ln_final, 
                        torch_model.text_projection, 
                        cache_dir=cache_dir, 
                        device=device.upper() if device == "cpu" else "GPU"
            )
    
    return model, transform, tokenizer