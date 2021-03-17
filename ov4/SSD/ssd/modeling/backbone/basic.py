import torch
from torch import nn


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        
        self.first_map = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, output_channels[0], 3, padding=1, stride=2),
        )
        self.second_map = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(output_channels[0], 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, output_channels[1], 3, padding=1, stride=2),
        )
        self.third_map = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(output_channels[1], 256, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, output_channels[2], 3, padding=1, stride=2),
        )
        self.forth_map = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(output_channels[2], 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, output_channels[3], 3, padding=1, stride=2),
        )
        self.fift_map = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(output_channels[3], 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, output_channels[4], 3, padding=1, stride=2),
        )
        self.sixt_map = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(output_channels[4], 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, output_channels[5], 3, padding=0, stride=2),
        )

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out_features.append(self.first_map(x))
        out_features.append(self.second_map(out_features[-1]))
        out_features.append(self.third_map(out_features[-1]))
        out_features.append(self.forth_map(out_features[-1]))
        out_features.append(self.fift_map(out_features[-1]))
        out_features.append(self.sixt_map(out_features[-1]))
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)


