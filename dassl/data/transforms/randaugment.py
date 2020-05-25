"""
Credit to
1) https://github.com/ildoonet/pytorch-randaugment
2) https://github.com/kakaobrain/fast-autoaugment
"""
import numpy as np
import random
import PIL
import torch
import PIL.ImageOps
import PIL.ImageDraw
import PIL.ImageEnhance
from PIL import Image


def ShearX(img, v):
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):
    # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):
    assert 0. <= v <= 2.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):
    assert 0. <= v <= 2.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):
    assert 0. <= v <= 2.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):
    assert 0. <= v <= 2.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):
    # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):
    # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v/2.))
    y0 = int(max(0, y0 - v/2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):
    # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


class Lighting:
    """Lighting noise (AlexNet - style PCA - based noise)."""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault:
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def randaugment_list():
    # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # augs = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4)  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    augs = [
        (AutoContrast, 0, 1), (Equalize, 0, 1), (Invert, 0, 1),
        (Rotate, 0, 30), (Posterize, 4, 8), (Solarize, 0, 256),
        (SolarizeAdd, 0, 110), (Color, 0.1, 1.9), (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9), (Sharpness, 0.1, 1.9), (ShearX, 0., 0.3),
        (ShearY, 0., 0.3), (CutoutAbs, 0, 40), (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100)
    ]

    return augs


def randaugment_list2():
    augs = [
        (AutoContrast, 0, 1), (Brightness, 0.1, 1.9), (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9), (Equalize, 0, 1), (Identity, 0, 1),
        (Invert, 0, 1), (Posterize, 4, 8), (Rotate, -30, 30),
        (Sharpness, 0.1, 1.9), (ShearX, -0.3, 0.3), (ShearY, -0.3, 0.3),
        (Solarize, 0, 256), (TranslateX, -0.3, 0.3), (TranslateY, -0.3, 0.3)
    ]

    return augs


def fixmatch_list():
    # https://arxiv.org/abs/2001.07685
    augs = [
        (AutoContrast, 0, 1), (Brightness, 0.05, 0.95), (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95), (Equalize, 0, 1), (Identity, 0, 1),
        (Posterize, 4, 8), (Rotate, -30, 30), (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3), (ShearY, -0.3, 0.3), (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3), (TranslateY, -0.3, 0.3)
    ]

    return augs


class RandAugment:

    def __init__(self, n=2, m=10):
        assert 0 <= m <= 30
        self.n = n
        self.m = m
        self.augment_list = randaugment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, minval, maxval in ops:
            val = (self.m / 30) * (maxval-minval) + minval
            img = op(img, val)

        return img


class RandAugment2:

    def __init__(self, n=2, p=0.6):
        self.n = n
        self.p = p
        self.augment_list = randaugment_list2()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, minval, maxval in ops:
            if random.random() > self.p:
                continue
            m = random.random()
            val = m * (maxval-minval) + minval
            img = op(img, val)

        return img


class RandAugmentFixMatch:

    def __init__(self, n=2):
        self.n = n
        self.augment_list = fixmatch_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, minval, maxval in ops:
            m = random.random()
            val = m * (maxval-minval) + minval
            img = op(img, val)

        return img
