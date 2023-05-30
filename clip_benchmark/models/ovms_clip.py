import open_clip
from pathlib import Path
import torch.nn.functional as F
import torch

import ovmsclient


class OVMSCLIP():
    
    def __init__(
                self,
                service_url):
        self.client = ovmsclient.make_grpc_client(service_url)
        self.image_encoder_model_name = "image_encoder"
        self.image_encoder_model_input_name = "image"
        self.text_encoder_model_name = "text_encoder"
        self.text_encoder_model_input_name = "input_ids"
        
    def encode_image(self, image, normalize: bool = False):
        features = torch.from_numpy(
            self.client.predict(
                inputs={
                    self.image_encoder_model_input_name: image.numpy()},
                model_name=self.image_encoder_model_name)
        )
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        x =  torch.from_numpy(
            self.client.predict(
                inputs={
                    self.text_encoder_model_input_name: text.numpy()},
                model_name=self.text_encoder_model_name)
        )
        return F.normalize(x, dim=-1) if normalize else x

    def __call__(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, None
    
    def eval(self):
        pass
        
    def to(self, device):
        pass

def load_ovms_clip(model_name: str, pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    """
    Initialize gRPC client to request CLIP models.
    Requires OVMS to be running on port 8913

    Args:
        model_name (str, optional): Unused
        cache_dir (str, optional): Unused
        device (str, optional): Unused

    Returns:
        _type_: _description_
    """
    MODEL_DESCRIPTION = "model_index.txt"
    with open(Path(model_name) / MODEL_DESCRIPTION) as file:
        lines = [line.rstrip() for line in file]
        orig_model_name, pretrained = lines[0].split(",")
        print(f"model {orig_model_name} with pretrained {pretrained}")
    
    _, _, transform = open_clip.create_model_and_transforms(orig_model_name, pretrained=pretrained, cache_dir=cache_dir)
    tokenizer = open_clip.get_tokenizer(orig_model_name)
    
    model = OVMSCLIP("localhost:8913")
    
    return model, transform, tokenizer
