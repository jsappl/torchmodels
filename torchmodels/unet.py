"""The famous U-Net architecture as described in https://arxiv.org/abs/1505.04597."""

import torch
import torch.nn as nn


class UNet(nn.Module):
    """A convolutional neural network architecture commonly used for semantic image segmentation."""

    def __init__(self, in_channels: int, out_channels: int, features: int) -> None:
        """Initialize the layers in the model.

        Args:
            in_channels: Color channels of the input image.
            out_channels: Color channels of the output image.
            features: Number of feature channels double for each layer.
        """
        super().__init__()

        self.doubleconv1 = self._double_convolution(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.doubleconv2 = self._double_convolution(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.doubleconv3 = self._double_convolution(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.doubleconv4 = self._double_convolution(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._double_convolution(features * 8, features * 16)

        self.deconv1 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.doubleconv5 = self._double_convolution((features * 8) * 2, features * 8)
        self.deconv2 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.doubleconv6 = self._double_convolution((features * 4) * 2, features * 4)
        self.deconv3 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.doubleconv7 = self._double_convolution((features * 2) * 2, features * 2)
        self.deconv4 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.doubleconv8 = self._double_convolution(features * 2, features)

        self.outconv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """A single forward pass through the model.

        A correct forward pass is only guaranteed for tensor heights and widths of powers of two.

        Args:
            input_: The input image to be processed.

        Returns:
            The U-Net model output.
        """
        encoded1 = self.doubleconv1(input_)
        encoded2 = self.doubleconv2(self.pool1(encoded1))
        encoded3 = self.doubleconv3(self.pool2(encoded2))
        encoded4 = self.doubleconv4(self.pool3(encoded3))

        bottleneck = self.bottleneck(self.pool4(encoded4))

        decoded1 = self.deconv1(bottleneck)
        decoded2 = self.deconv2(self.doubleconv5(torch.cat((decoded1, encoded4), dim=1)))
        decoded3 = self.deconv3(self.doubleconv6(torch.cat((decoded2, encoded3), dim=1)))
        decoded4 = self.deconv4(self.doubleconv7(torch.cat((decoded3, encoded2), dim=1)))

        return self.outconv(self.doubleconv8(torch.cat((decoded4, encoded1), dim=1)))

    @staticmethod
    def _double_convolution(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
