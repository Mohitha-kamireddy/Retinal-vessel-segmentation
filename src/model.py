import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    # Two consecutive 3x3 convolutions, each followed by BatchNorm and ReLU.
    # This is the basic building block used at every level of the U-Net.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    # Standard U-Net for binary segmentation.
    #
    # Encoder: repeatedly applies DoubleConv then downsamples with max pooling,
    # doubling the number of channels at each stage while halving spatial size.
    #
    # Bottleneck: DoubleConv at the lowest resolution, connecting encoder and decoder.
    #
    # Decoder: repeatedly upsamples, concatenates the matching encoder feature map
    # (skip connection) and applies DoubleConv, halving channels while doubling
    # spatial size back to the input resolution.
    #
    # Output head: 1x1 convolution producing a single-channel logit map for
    # binary vessel segmentation (apply sigmoid outside the model, e.g. in the loss).
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for feature in features:
            self.encoders.append(DoubleConv(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path (upsampling + double conv), built in reverse order
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2
        for feature in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2)
            )
            # Input channels double because of concatenation with the skip connection
            self.decoders.append(DoubleConv(feature * 2, feature))
            prev_channels = feature

        # Final 1x1 conv maps to the desired number of output channels (1 for binary mask)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder: save feature maps before each downsampling step
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reverse so skip connections align with decoder stages (deepest first)
        skip_connections = skip_connections[::-1]

        # Decoder: upsample, concatenate skip connection, then double conv
        for idx in range(len(self.decoders)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]

            # Handle potential off-by-one size mismatch from odd input dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = self.decoders[idx](x)

        return self.final_conv(x)


if __name__ == "__main__":
    # Quick sanity check: python src/model.py
    model = UNet(in_channels=3, out_channels=1)
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)
