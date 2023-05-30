from typing import Union
import torch
from .open_clip import load_open_clip
from .japanese_clip import load_japanese_clip
from .openvino_clip import load_openvino_clip
from .ovms_clip import load_ovms_clip

# loading function must return (model, transform, tokenizer)
TYPE2FUNC = {
    "open_clip": load_open_clip,
    "ja_clip": load_japanese_clip, 
    "openvino_clip": load_openvino_clip,
    "ovms_clip": load_ovms_clip,
}
MODEL_TYPES = list(TYPE2FUNC.keys())


def load_clip(
        model_type: str,
        model_name: str,
        pretrained: str,
        cache_dir: str,
        device: Union[str, torch.device] = "cuda"
):
    assert model_type in MODEL_TYPES, f"model_type={model_type} is invalid!"
    load_func = TYPE2FUNC[model_type]
    return load_func(model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device)
