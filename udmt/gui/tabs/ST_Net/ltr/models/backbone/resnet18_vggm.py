import math
import torch
import torch.nn as nn
from collections import OrderedDict
from .base import Backbone

# Compatible with different versions of torchvision
try:
    from torchvision.models.resnet import BasicBlock
except ImportError:
    # If import from torchvision.models.resnet fails, try other methods
    try:
        from torchvision.models import resnet
        BasicBlock = resnet.BasicBlock
    except ImportError:
        # Final fallback: custom BasicBlock
        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                         base_width=64, dilation=1, norm_layer=None):
                super(BasicBlock, self).__init__()
                if norm_layer is None:
                    norm_layer = nn.BatchNorm2d
                if groups != 1 or base_width != 64:
                    raise ValueError('BasicBlock only supports groups=1 and base_width=64')
                if dilation > 1:
                    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
                
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False)
                self.bn1 = norm_layer(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                     padding=1, bias=False)
                self.bn2 = norm_layer(planes)
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


class SpatialCrossMapLRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class ResNetVGGm1(Backbone):

    def __init__(self, block, layers, output_layers, num_classes=1000, frozen_layers=()):
        self.inplanes = 64
        super(ResNetVGGm1, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.vggmconv1 = nn.Conv2d(3,96,(7, 7),(2, 2), padding=3)
        self.vgglrn = SpatialCrossMapLRN(5, 0.0005, 0.75, 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)


    def forward(self, x, output_layers=None):
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        if 'vggconv1' in output_layers:
            c1 = self.vgglrn(self.relu(self.vggmconv1(x)))
            if self._add_output_and_check('vggconv1', c1, outputs, output_layers):
                return outputs

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        x = self.layer4(x)

        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self._add_output_and_check('fc', x, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


def resnet18_vggmconv1(output_layers=None, path=None, **kwargs):
    """Constructs a ResNet-18 model with first-layer VGGm features.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['vggconv1', 'conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNetVGGm1(BasicBlock, [2, 2, 2, 2], output_layers, **kwargs)

    if path is not None:
        model.load_state_dict(torch.load(path), strict=False)
    return model