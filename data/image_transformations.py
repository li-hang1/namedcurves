import random
import torchvision.transforms.functional as F
from torchvision import transforms

class RandomCropPair:
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2):
        i, j, h, w = transforms.RandomCrop.get_params(img1, self.size)  # 输出随机裁剪的左上角坐标和宽高，第一个参数输入图像，第二个参数输入裁剪大小，如果是个整数就裁成正方形
        img1 = F.crop(img1, i, j, h, w)  # 从img1的(i, j)位置开始，裁掉一个大小为(h, w)的子区域。
        img2 = F.crop(img2, i, j, h, w)
        return img1, img2

class ResizePair:
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2):
        # antialias=True is used to avoid torchvision warning
        img1 = F.resize(img1, self.size, antialias=True)  # 改变图像大小，第一个参数是图像，第二个参数是缩放后大小，如果是整数表示短边缩放到n，长边按比例变化。
        img2 = F.resize(img2, self.size, antialias=True)
        return img1, img2

class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            img1 = F.hflip(img1)  # 对图像进行水平翻转
            img2 = F.hflip(img2)
        return img1, img2

class RandomVerticalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            img1 = F.vflip(img1)  # 对图像进行垂直翻转
            img2 = F.vflip(img2)
        return img1, img2

def get_transforms(transforms_config):
    transform_list = []
    for transform in transforms_config:
        transform_type = transform['type']
        params = transform['params']
        if transform_type == 'RandomCrop':
            transform_list.append(RandomCropPair(**params))
        elif transform_type == 'Resize':
            transform_list.append(ResizePair(**params))
        elif transform_type == 'RandomHorizontalFlip':
            transform_list.append(RandomHorizontalFlipPair(**params))
        elif transform_type == 'RandomVerticalFlip':
            transform_list.append(RandomVerticalFlipPair(**params))
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")

    return transform_list