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

        """
        if hasattr(cfg, "dropout") and cfg.dropout is not None:
            from .dropout_wrapper import DropoutWrapper
            model = DropoutWrapper(make_resnet_cifar(depth=56, num_classes=num_classes), dropout=cfg.dropout, num_mc_samples=10)
        """
        
    elif cfg.model == "resnet50":
        import torchvision
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif cfg.model == "wide_resnet50_2":
        import torchvision
        model = torchvision.models.wide_resnet50_2(num_classes=num_classes) 
    elif cfg.model == "resnet101":
        import torchvision
        model = torchvision.models.resnet101(num_classes=num_classes)

    elif cfg.model == "resnet9":
        from .tiny_resnet import create_model
        model = create_model()


    elif cfg.model.startswith("resnet"):
        assert cfg.dataset == "cifar10" or cfg.dataset == "cifar100", "must use cifar10/100 for variable size resnet, custom code handles it"

        # get the number that cfg.model endswith (resnetN)
        depth = int(cfg.model[6:])

        from .resnet_cifar import make_resnet_cifar
        assert (depth - 2) % 6 == 0, "depth should be 6n+2 (e.g., 20, 32, 44, 56, 110, 1202)"
        n = (depth - 2) // 6
        model = make_resnet_cifar(depth=depth, num_classes=num_classes, n=n)

    elif cfg.model == "densenet121":
        import torchvision
        model = torchvision.models.densenet121(num_classes=num_classes)

        
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
            if cfg.pretrained:
                pretrained_path = "/fast/slaing/pretrained_weights/VIT_tiny_imagenet/vit_b_16_weights.pth"

                try:
                    pretrained_weights = torch.load(pretrained_path)

                    filtered_weights = {
                        k:v for k, v in pretrained_weights.items() if not k.startswith("head.head")
                    }
                    model.load_state_dict(filtered_weights, strict=False)
                    print(f"Pretrained weights loaded from {pretrained_path}")
                
                except Exception as e:
                    print(f"Error loading pretrained weights: {str(e)}")
                    raise Exception(f"Pretrained weights not found at {pretrained_path}. Please check the path.")
                    

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
