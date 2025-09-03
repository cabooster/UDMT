import torch
from torch import nn
import torch.nn.functional as F

# Compatible with different versions of torchvision
try:
    from torchvision.models.resnet import BasicBlock, Bottleneck
except ImportError:
    # If import from torchvision.models.resnet fails, try other methods
    try:
        from torchvision.models import resnet
        BasicBlock = resnet.BasicBlock
        Bottleneck = resnet.Bottleneck
    except ImportError:
        # Final fallback: custom implementation
        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(BasicBlock, self).__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                     padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out += identity
                out = self.relu(out)
                return out

        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(Bottleneck, self).__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(planes * self.expansion)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)
                out = self.conv3(out)
                out = self.bn3(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out += identity
                out = self.relu(out)
                return out

from udmt.gui.tabs.ST_Net.ltr.models.layers.normalization import InstanceL2Norm
# from ltr.models.layers.normalization import InstanceL2Norm
from udmt.gui.tabs.ST_Net.ltr.models.layers.transform import InterpCat
# from ltr.models.layers.transform import InterpCat


def residual_basic_block(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                         interp_cat=False, final_relu=False, init_pool=False):
    """Construct a network block based on the BasicBlock used in ResNet 18 and 34."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    if init_pool:
        feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))
    if final_conv:
        feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
        if final_relu:
            feat_layers.append(nn.ReLU(inplace=True))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)


def residual_basic_block_pool(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                              pool=True):
    """Construct a network block based on the BasicBlock used in ResNet."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))
    if final_conv:
        feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
    if pool:
        feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))

    return nn.Sequential(*feat_layers)


def residual_bottleneck(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                        interp_cat=False, final_relu=False, final_pool=False):
    """Construct a network block based on the Bottleneck block used in ResNet."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    for i in range(num_blocks):
        planes = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim // 4
        feat_layers.append(Bottleneck(4*feature_dim, planes))
    if final_conv:
        feat_layers.append(nn.Conv2d(4*feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
        if final_relu:
            feat_layers.append(nn.ReLU(inplace=True))
        if final_pool:
            feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)