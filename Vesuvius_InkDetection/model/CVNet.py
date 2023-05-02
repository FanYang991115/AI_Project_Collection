import torch
from torch import nn
import segmentation_models_pytorch as smp


class CVNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CVNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=input_channel, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(
            in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1
        )

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
        )
        return block

    def forward(self, X):
        contracting_11_out = self.contracting_11(X)  # [-1, 64, 512, 512]
        contracting_12_out = self.contracting_12(
            contracting_11_out
        )  # [-1, 64, 256, 256]
        contracting_21_out = self.contracting_21(
            contracting_12_out
        )  # [-1, 128, 256, 256]
        contracting_22_out = self.contracting_22(
            contracting_21_out
        )  # [-1, 128, 128, 128]
        contracting_31_out = self.contracting_31(
            contracting_22_out
        )  # [-1, 256, 128, 128]
        contracting_32_out = self.contracting_32(
            contracting_31_out
        )  # [-1, 256, 64, 64]
        contracting_41_out = self.contracting_41(
            contracting_32_out
        )  # [-1, 512, 64, 64]
        contracting_42_out = self.contracting_42(
            contracting_41_out
        )  # [-1, 512, 32, 32]
        middle_out = self.middle(contracting_42_out)  # [-1, 1024, 32, 32]
        expansive_11_out = self.expansive_11(middle_out)  # [-1, 512, 64, 64]
        expansive_12_out = self.expansive_12(
            torch.cat((expansive_11_out, contracting_41_out), dim=1)
        )  # [-1, 1024, 64, 64] -> [-1, 512, 64, 64]
        expansive_21_out = self.expansive_21(expansive_12_out)  # [-1, 256, 128, 128]
        expansive_22_out = self.expansive_22(
            torch.cat((expansive_21_out, contracting_31_out), dim=1)
        )  # [-1, 512, 128, 128] -> [-1, 256, 128, 128]
        expansive_31_out = self.expansive_31(expansive_22_out)  # [-1, 128, 256, 256]
        expansive_32_out = self.expansive_32(
            torch.cat((expansive_31_out, contracting_21_out), dim=1)
        )  # [-1, 256, 256, 256] -> [-1, 128, 256, 256]
        expansive_41_out = self.expansive_41(expansive_32_out)  # [-1, 64, 512, 512]
        expansive_42_out = self.expansive_42(
            torch.cat((expansive_41_out, contracting_11_out), dim=1)
        )  # [-1, 128, 512, 512] -> [-1, 64, 512, 512]
        output_out = self.output(expansive_42_out)  # [-1, num_classes, 512, 512]

        if self.training:
            return output_out
        else:
            return torch.sigmoid(output_out)


class CustomModel(nn.Module):
    def __init__(self, Config, weight=None):
        super().__init__()
        self.Config = Config

        self.encoder = smp.Unet(
            encoder_name=Config.BACKBONE,
            encoder_weights=weight,
            in_channels=Config.Z_DIM,
            classes=1,
            activation=None,
        )

    def forward(self, image):
        output = self.encoder(image)

        if self.training:
            return output
        else:
            return torch.sigmoid(output)


def build_model(Config, weight="imagenet"):
    print('model_name', Config.NB)
    print('backbone', Config.BACKBONE)

    model = CustomModel(Config, weight)

    return model