from functools import lru_cache
from PIL import Image
import torch
from torchvision import transforms
from transformer_net import TransformerNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=3)
def _load_style_model(path: str) -> torch.nn.Module:
    """
    Load a style model from path. 
    Try TorchScript first; if that fails, load state_dict & script it.
    Cached for subsequent calls.
    """
    try:
        model = torch.jit.load(path).to(DEVICE)
        model.eval()
        return model
    except Exception:
        # Fallback: raw state_dict
        state_dict = torch.load(path, map_location=DEVICE)
        net = TransformerNet()
        net.load_state_dict(state_dict)
        net.eval()
        scripted = torch.jit.script(net).to(DEVICE)
        return scripted

def stylize(content_image: Image.Image, style_model_path: str) -> Image.Image:
    """
    Stylize a PIL image given a .pth (state‐dict or TorchScript).
    """
    model = _load_style_model(style_model_path)

    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    img_t = preprocess(content_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out_t = model(img_t).cpu()

    out_t = out_t.squeeze().clamp(0, 255)
    arr = out_t.numpy().transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(arr)
