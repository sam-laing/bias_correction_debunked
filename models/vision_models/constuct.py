from fractions import Fraction
import wandb  
import torchvision
import torch

def construct_model(cfg):
    if cfg.dataset == "cifar10": 
        num_classes = 10
    elif cfg.dataset == "cifar100":
        from vision_models.resnet_cifar import make_resnet_cifar
        num_classes = 100
    elif cfg.dataset == "tiny_imagenet":
        num_classes = 200
    elif cfg.dataset == "cub":
        num_classes = 200
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} not implemented or misspelled")


    #cifar models
    if cfg.model == "resnet20":
        from vision_models.resnet_cifar import make_resnet_cifar 
        model = make_resnet_cifar(depth=20, num_classes=num_classes)
    elif cfg.model == "resnet32":
        from vision_models.resnet_cifar import make_resnet_cifar 
        model = make_resnet_cifar(depth=32, num_classes=num_classes)
    elif cfg.model == "resnet56":
        from vision_models.resnet_cifar import make_resnet_cifar 
        model = make_resnet_cifar(depth=56, num_classes=num_classes)
        
    elif cfg.model == "resnet50":
        import torchvision
        model = torchvision.models.resnet50(pretrained=False)
    elif cfg.model == "wide_resnet50_2":
        import torchvision
        model = torchvision.models.wide_resnet50_2(pretrained=False)
    elif cfg.model == "resnet101":
        import torchvision
        model = torchvision.models.resnet101(pretrained=False)
    
    elif cfg.model == "ViT":
        raise NotImplementedError("ViT not implemented yet")

    else:
        raise NotImplementedError(f"Model {cfg.model} not implemented or misspelled")
            
    return model
