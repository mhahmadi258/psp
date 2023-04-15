import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear


class BackboneEncoderFirstStage(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderFirstStage, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten())
        
        self.output_seq_3 = torch.nn.LSTM(256 * 7 * 7, 512)
        self.output_lin_3 = Linear(512, 512 * 9)
        
        
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten())
        
        self.output_seq_4 = torch.nn.LSTM(128 * 7 * 7, 512)
        self.output_lin_4 = Linear(512, 512 * 5)
        
        
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten())
        
        self.output_seq_5 = torch.nn.LSTM(64 * 7 * 7, 512)
        self.output_lin_5 = Linear(512, 512 * 4)
        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    def forward(self, x):
        lc_part_4 = list()
        lc_part_3 = list()
        lc_part_2 = list()
        for i in range(x.shape[1]):
            x_frame = x[:,i,:,:,:]
            x_frame = self.input_layer(x_frame)
            
            for l in self.modulelist[:3]:
                x_frame = l(x_frame)
            lc_part_4.append(self.output_layer_5(x_frame))
            
            for l in self.modulelist[3:7]:
                x_frame = l(x_frame)
            lc_part_3.append(self.output_layer_4(x_frame))
            
            for l in self.modulelist[7:21]:
                x_frame = l(x_frame)
            lc_part_2.append(self.output_layer_3(x_frame))
            
            
        lc_part_4 = torch.stack(lc_part_4)
        lc_part_3 = torch.stack(lc_part_3)
        lc_part_2 = torch.stack(lc_part_2)
        
        
        _, (lc_part_4, _) = self.output_seq_5(lc_part_4)
        _, (lc_part_3, _) = self.output_seq_4(lc_part_3)
        _, (lc_part_2, _) = self.output_seq_3(lc_part_2)
        
        lc_part_4 = self.output_lin_5(lc_part_4.squeeze(0)).view(-1, 4, 512)
        lc_part_3 = self.output_lin_4(lc_part_3.squeeze(0)).view(-1, 5, 512)
        lc_part_2 = self.output_lin_3(lc_part_2.squeeze(0)).view(-1, 9, 512)
            
        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        print(f'resnet50 backbone: {sum(p.numel() for p in self.body.parameters() if p.requires_grad)}')

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        print(f'mapping network: {sum(p.numel() for p in self.styles.parameters() if p.requires_grad)}')
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x
