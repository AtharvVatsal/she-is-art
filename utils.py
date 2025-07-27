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
    1) If it's a TorchScript archive, load it.
    2) Else, load the raw state_dict into TransformerNet.
    """
    try:
        # Try loading a scripted model
        model = torch.jit.load(path).to(DEVICE)
        model.eval()
        return model
    except (RuntimeError, torch.jit.Error):
        # Fallback: load raw state_dict
        state_dict = torch.load(path, map_location=DEVICE)

        # Remove any running stats keys if present
        cleaned = {
            k: v for k, v in state_dict.items()
            if not (k.endswith("running_mean") or k.endswith("running_var"))
        }

        net = TransformerNet().to(DEVICE)
        net.load_state_dict(cleaned, strict=False)
        net.eval()
        return net

def stylize(content_image: Image.Image, style_model_path: str) -> Image.Image:
    """
    Stylize a PIL image given a .pth (TorchScript or state_dict).
    """
    model = _load_style_model(style_model_path)

    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    tensor = preprocess(content_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor).cpu()

    output = output.squeeze().clamp(0, 255)
    arr = output.numpy().transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(arr)
