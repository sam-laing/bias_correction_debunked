from pathlib import Path
import torch
from torchvision.models import resnet50, ResNet50_Weights

def save_pretrained_resnet50(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / "resnet50_imagenet1k.pth"

    print(f"Loading pretrained ResNet-50 weights...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), weights_path)

    print(f"Saved weights to: {weights_path}")

if __name__ == "__main__":
    save_path = Path("/fast/slaing/pretrained_weights/resnet50/")
    save_pretrained_resnet50(save_path)
