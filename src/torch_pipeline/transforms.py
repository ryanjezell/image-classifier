import torch
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(cfg):
    a = cfg.augmentation
    ops = [
        v2.ToImage(),
        v2.RandomResizedCrop(cfg.data.image_size, scale=(0.8, 1.0)),
    ]
    if a.flip_horiz:
        ops.append(v2.RandomHorizontalFlip(p=0.5))
    if a.max_rotate > 0:
        ops.append(v2.RandomRotation(degrees=a.max_rotate))
    if a.max_lighting > 0:
        ops.append(v2.ColorJitter(brightness=a.max_lighting, contrast=a.max_lighting))
    ops.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return v2.Compose(ops)


def build_eval_transform(cfg):
    size = cfg.data.image_size
    resize_size = int(size * 1.14)
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(resize_size),
        v2.CenterCrop(size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_inference_transform(image_size: int, mean=None, std=None):
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    resize_size = int(image_size * 1.14)
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(resize_size),
        v2.CenterCrop(image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
