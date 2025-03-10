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

    elif cfg.model == "basic_cnn":
        if cfg.dataset == "cifar10" or cfg.dataset == "svhn":
            # make a cnn for cfg.batch_size x 
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 16 * 16, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10)
            )
        elif cfg.dataset == "cifar100":
            # make a cnn for cfg.batch_size x 
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 16 * 16, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 100)
            )


        elif cfg.dataset == "tiny_imagenet":
            # make a tiny imagenet appropriate cnn
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2),
                torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2),
                torch.nn.Conv2d(192, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2),
                torch.nn.Flatten(),
                torch.nn.Linear(256 * 7 * 7, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(4096, 200)
            )

        elif cfg.dataset == "imagenet":
            # make an image net appropriate cnn
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2),
                torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2),
                torch.nn.Conv2d(192, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2),
                torch.nn.Flatten(),
                torch.nn.Linear(256 * 7 * 7, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(4096, 1000)
            )
            
    return model
