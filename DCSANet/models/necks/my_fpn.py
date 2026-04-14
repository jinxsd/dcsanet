from collections import OrderedDict
from typing import List

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torch.nn import functional as F


class AdaptivePoolingFusion(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, low_feature, high_feature):
        low_feature = self.conv(low_feature)
        high_feature = self.conv(high_feature)

        low_feature_pool = self.pool(low_feature)
        high_feature_pool = self.pool(high_feature)

        fused_feature = low_feature + high_feature + (low_feature_pool - high_feature_pool)
        return fused_feature

class SR_3DFusion_Block_Upsample(nn.Module):
    def __init__(self, in_channels):
        super(SR_3DFusion_Block_Upsample, self).__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(4, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        )

    def forward(self, features_list):
        target_h, target_w = features_list[0].shape[-2:] 
        
        aligned_features = []
        for feat in features_list:
            if feat.shape[-2:] != (target_h, target_w):
                aligned_feat = F.interpolate(feat, size=(target_h, target_w), mode='nearest')
            else:
                aligned_feat = feat
            
            aligned_features.append(aligned_feat.unsqueeze(2))
        
        stacked_features = torch.cat(aligned_features, dim=2)
        fused_3d = self.conv3d(stacked_features)
        fused_2d = fused_3d.squeeze(2)
        original_p2 = features_list[0]
        new_p2 = original_p2 + self.refine(fused_2d)
        
        return new_p2

class SR_EnhancedFPN_P2(nn.Module):
    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            extra_block: bool = False,
            norm_layer: nn.Module = None,
    ):
        super(SR_EnhancedFPN_P2, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.fusion_modules = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block_module = Conv2dNormActivation(in_channels, out_channels, kernel_size=1, padding=0, norm_layer=norm_layer, activation_layer=None)
            layer_block_module = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, norm_layer=norm_layer, activation_layer=None)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        for _ in range(len(in_channels_list) - 1):
            self.fusion_modules.append(AdaptivePoolingFusion(out_channels))

        self.sr_3d_fusion = SR_3DFusion_Block_Upsample(out_channels)
        self.extra_block = extra_block

    def forward(self, x: OrderedDict) -> OrderedDict:
        keys = list(x.keys())
        x = list(x.values())
        results = [self.inner_blocks[idx](x[idx]) for idx in range(len(x))]

        for idx in range(len(x) - 1, 0, -1):
            feat_shape = results[idx - 1].shape[-2:]
            results[idx - 1] = self.fusion_modules[idx - 1](
                low_feature=results[idx - 1],
                high_feature=F.interpolate(results[idx], size=feat_shape, mode="nearest"),
            )

        output_features = [self.layer_blocks[idx](results[idx]) for idx in range(len(x))]
        new_p2 = self.sr_3d_fusion(output_features)
        final_output = OrderedDict()
        final_output[keys[0]] = new_p2
        for idx in range(1, len(output_features)):
            final_output[keys[idx]] = output_features[idx]

        if self.extra_block:
            final_output["pool"] = F.max_pool2d(output_features[-1], 1, 2, 0)

        return final_output