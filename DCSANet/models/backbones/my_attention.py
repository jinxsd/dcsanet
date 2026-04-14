from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class PluginConfig:
    def __init__(self, block: nn.Module, stages: List[int], conv_pos: List[int]):
        assert max(stages) < 4, "stage index should be less than 4"
        assert max(conv_pos) < 3, "conv_pos should be less than 3"
        self.block = block
        self.stages = stages
        self.conv_pos = conv_pos


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_experts=4):
        super(DynamicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_experts = num_experts 
        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels, kernel_size, kernel_size))  
        self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=1)
        )
        
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / fan_in**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        b, c, h, w = x.shape
        attention = self.router(x)
        
        aggregate_weight = torch.einsum('bk,koihw->boihw', attention, self.weight)
        aggregate_bias = torch.einsum('bk,ko->bo', attention, self.bias)

        x_grouped = x.view(1, b * c, h, w)
        weight_grouped = aggregate_weight.view(b * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        bias_grouped = aggregate_bias.view(b * self.out_channels)

        out = F.conv2d(x_grouped, weight_grouped, bias=bias_grouped, padding=self.padding, groups=b)
        out = out.view(b, self.out_channels, h, w)
        return out

class JSA_Module(nn.Module):
    def __init__(self, in_channels):
        super(JSA_Module, self).__init__()
        self.dynamic_conv = DynamicConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.offset_conv = nn.Conv2d(in_channels, 2 * 3 * 3, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        self.scheduler = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out_dynamic = self.dynamic_conv(x)
        
        offsets = self.offset_conv(x)
        out_deform = self.deform_conv(x, offsets)
        
        select_map = self.scheduler(x)
        
        out = select_map * out_deform + (1 - select_map) * out_dynamic
        
        return out

class DPAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DPAM, self).__init__()
        self.se_block = SEBlock(in_channels, reduction=reduction)
        self.jsa_module = JSA_Module(in_channels)

    def forward(self, x):
        #residual = x
        x_se = self.se_block(x)
        out = self.jsa_module(x_se)
        return out

if __name__ == "__main__":

    input_tensor = torch.randn(2, 64, 32, 32)
    model = DPAM(in_channels=64)
    output = model(input_tensor)
    
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output.shape}")
    assert output.requires_grad == True
    print("DPAM forward pass successful.")