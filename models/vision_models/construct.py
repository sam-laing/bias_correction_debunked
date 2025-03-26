from fractions import Fraction
import wandb  
import torchvision
import torch

def construct_model(cfg):
    if cfg.dataset == "cifar10": 
        num_classes = 10
    elif cfg.dataset == "cifar100":
        num_classes = 100
    elif cfg.dataset == "tiny_imagenet":
        num_classes = 200
    elif cfg.dataset == "imagenet":
        num_classes = 1000
    elif cfg.dataset == "cub":
        num_classes = 200
    elif cfg.dataset == "svhn":
        num_classes = 10
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} not implemented or misspelled")


    #cifar models
    if cfg.model == "resnet20":
        from .resnet_cifar import make_resnet_cifar 
        model = make_resnet_cifar(depth=20, num_classes=num_classes)
    elif cfg.model == "resnet32":
        from .resnet_cifar import make_resnet_cifar 
        model = make_resnet_cifar(depth=32, num_classes=num_classes)
    elif cfg.model == "resnet56":
        from .resnet_cifar import make_resnet_cifar 
        model = make_resnet_cifar(depth=56, num_classes=num_classes)
        
    elif cfg.model == "resnet50":
        import torchvision
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif cfg.model == "wide_resnet50_2":
        import torchvision
        model = torchvision.models.wide_resnet50_2(num_classes=num_classes) 
    elif cfg.model == "resnet101":
        import torchvision
        model = torchvision.models.resnet101(num_classes=num_classes)
        
    elif cfg.model == "ViT":
        from torchvision.models import vision_transformer
        # Need to handle smaller image size for TinyImageNet (64x64)
        if cfg.dataset == "tiny_imagenet":
            # Option 1: Use a smaller patch size appropriate for 64x64 images
            model = vision_transformer.VisionTransformer(
                image_size=64,
                patch_size=8,  # Smaller patch size
                num_layers=8,
                num_heads=8,
                hidden_dim=512,
                mlp_dim=2048,
                num_classes=num_classes, 
                dropout=0.1,
            )
        elif cfg.dataset == "imagenet":
            # For larger datasets, use the pre-trained model
            model = vision_transformer.vit_base_patch16_224_in21k(num_classes=num_classes)

        elif cfg.dataset == "cifar10":
            model = vision_transformer.VisionTransformer(
                image_size=32,
                patch_size=4,
                num_layers=6,
                num_heads=8,
                hidden_dim=256,
                mlp_dim=512,
                num_classes=num_classes
            )
    

    else:
        raise NotImplementedError(f"Model {cfg.model} not implemented or misspelled")
            
    return model
