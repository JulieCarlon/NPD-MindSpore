
from mindcv.models.resnet import ResNet
from typing import List, Optional, Type, Union

import mindspore.common.initializer as init
from mindspore import Tensor, nn

class BasicBlock(nn.Cell):
    """define the basic block of resnet"""
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        assert groups == 1, "BasicBlock only supports groups=1"
        assert base_width == 64, "BasicBlock only supports base_width=64"

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, pad_mode="pad")
        self.bn1 = norm(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, pad_mode="pad")
        self.bn2 = norm(channels)
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    """
    Bottleneck here places the stride for downsampling at 3x3 convolution(self.conv2) as torchvision does,
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    """
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        width = int(channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)
        self.bn1 = norm(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, pad_mode="pad", group=groups)
        self.bn2 = norm(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion,
                               kernel_size=1, stride=1)
        self.bn3 = norm(channels * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out

class Resnet_npd(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
    ) -> None:
        super(Resnet_npd, self).__init__(block, layers, num_classes, in_channels, groups, base_width, norm)
        self.plug_layer = nn.SequentialCell([nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=1, stride=1,padding=0,has_bias=False),nn.BatchNorm2d(512)])
        self._init_plug_layer()
        self._freeze_backbone()

    def _init_plug_layer(self):
        ## init plug layer as identity
        self.plug_layer[0].weight.set_data(init.initializer('zeros', self.plug_layer[0].weight.shape, self.plug_layer[0].weight.dtype))
        for i in range(self.plug_layer[0].weight.shape[0]):
            self.plug_layer[0].weight[i][i][0][0] = 1.0
        self.plug_layer[1].gamma.set_data(init.initializer('ones', self.plug_layer[1].gamma.shape, self.plug_layer[1].gamma.dtype))
        self.plug_layer[1].beta.set_data(init.initializer('zeros', self.plug_layer[1].beta.shape,  self.plug_layer[1].beta.dtype))

    def _freeze_backbone(self):
        for _, cell in self.cells_and_names():
            if 'plug_layer' in _: continue
            if isinstance(cell, nn.Conv2d):
                cell.weight.requires_grad = False
                if cell.bias is not None:
                    cell.bias.requires_grad = False
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.requires_grad = False
                cell.beta.requires_grad = False
            elif isinstance(cell, nn.Dense):
                cell.weight.requires_grad = False
                if cell.bias is not None:
                    cell.bias.requires_grad = False      
        self.plug_layer.requires_grad = True

    def construct(self, x):
        x = self.forward_features(x)
        x = self.plug_layer(x)
        x = self.forward_head(x)
        return x

def _create_resnet(pretrained=False, **kwargs):
    return Resnet_npd(**kwargs)

def resnet18_npd(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 18 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **model_args)

def resnet34_npd(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 34 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **model_args)




