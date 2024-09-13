import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba
import torch.fft as fft




class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]  # B=24, C =[24, 32, 48, 64]  D =[32*32,16*16,8*8,8*8]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # m1= self.fft_blk(x)
        # print(x.shape)

        
        # print(x.shape)
        assert C == self.input_dim
        # x =self.fft_blk(x)

        # print("x_after fft={}".format(x.shape))
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        # print("x1={},x2={},x3={},x4={} ".format(x1.shape,x2.shape,x3.shape,x4.shape))
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

class SFIMoudle(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(2*input_dim)
        self.fft_blk = FrequencyFilter(input_dim)
        self.enc_blk = DoubleConv(input_dim, input_dim)
        self.mamba = Mamba(
                d_model=input_dim//2, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(2*input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C,H ,W = x.shape  # B=24, C =[24, 32, 48, 64]  D =[32*32,16*16,8*8,8*8]
        result1 = []
        for i in range(0,C):
            result1.append(i)
            result1.append(C+i)
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        m1= self.fft_blk(x)
        m2= self.enc_blk(x)
        result = torch.cat([m1, m2], dim=1)
        assert C == self.input_dim
        perm =torch.randperm(2*C)
        print(perm)
        result = result[:,result1,:,:]
        x_flat = result.reshape(B,2*C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out
    
class Full_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.mamba = Mamba(
            d_model=8, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.mamba1 = Mamba(
            d_model=16, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.skip_scale= nn.Parameter(torch.ones(1))
        self.norm = nn.LayerNorm(8)
        self.proj = nn.Linear(8, 8)
        self.norm1 = nn.LayerNorm(16)
        self.proj1 = nn.Linear(16, 16)

        
    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        #拆分
        C =[8, 16, 24, 32, 48]
        # r1, r2, r3, r4, r5 = 
        B= t1.shape[0]  # B=24, C =[8, 16, 24, 32, 48]  D =[32*32,16*16,8*8,8*8]
        n_tokens = [t.shape[2:].numel() for t in [t1, t2, t3, t4, t5]]
        # img_dims = x.shape[2:]
        t_flat = [t.reshape(B, C[i], n_tokens[i]).transpose(-1, -2) for i, t in enumerate([t1, t2, t3, t4, t5])]

        # x_norm = self.norm(x_flat)
        t1_cat = torch.cat((t_flat[0][:,:,0:8],t_flat[1][:,:,0:8],t_flat[2][:,:,0:8],t_flat[3][:,:,0:8],t_flat[4][:,:,0:8]), dim=1)
        t2_cat = torch.cat((t_flat[1][:,:,8:16],t_flat[2][:,:,8:16],t_flat[3][:,:,8:16],t_flat[4][:,:,8:16]), dim=1)
        t3_cat = torch.cat((t_flat[2][:,:,16:24],t_flat[3][:,:,16:24],t_flat[4][:,:,16:24]), dim=1)
        t4_cat = torch.cat((t_flat[3][:,:,24:32],t_flat[4][:,:,24:32]), dim=1)
        t5_cat = t_flat[4][:,:,32:48]

        x_mamba1 =self.proj(self.norm(self.mamba(t1_cat) + self.skip_scale * t1_cat))
        x_mamba2 =self.proj(self.norm(self.mamba(t2_cat) + self.skip_scale * t2_cat))
        x_mamba3 =self.proj(self.norm(self.mamba(t3_cat) + self.skip_scale * t3_cat))
        x_mamba4 =self.proj(self.norm(self.mamba(t4_cat) + self.skip_scale * t4_cat))
        x_mamba5 =self.proj1(self.norm1(self.mamba1(t5_cat) + self.skip_scale * t5_cat))


        after_t1_flat = x_mamba1[:,0:16384,:]
        after_t2_flat = torch.cat((x_mamba1[:,16384:20480,:],x_mamba2[:,0:4096,:]), dim=2)
        after_t3_flat = torch.cat((x_mamba1[:,20480:21504,:],x_mamba2[:,4096:5120,:],x_mamba3[:,0:1024,:]), dim=2)
        after_t4_flat = torch.cat((x_mamba1[:,21504:21760,:],x_mamba2[:,5120:5376,:],x_mamba3[:,1024:1280,:],x_mamba4[:,0:256,:]), dim=2)
        after_t5_flat = torch.cat((x_mamba1[:,21760:,:],x_mamba2[:,5376:,:],x_mamba3[:,1280:,:],x_mamba4[:,256:,:],x_mamba5[:,:,:]), dim=2)
        # print(after_t1_flat.shape,after_t2_flat.shape,after_t3_flat.shape,after_t4_flat.shape,after_t5_flat.shape)
        x1 = after_t1_flat.transpose(-1, -2).reshape(B, 8, 128,128)
        x2 = after_t2_flat.transpose(-1, -2).reshape(B, 16, 64,64)
        x3 = after_t3_flat.transpose(-1, -2).reshape(B, 24, 32,32)
        x4 = after_t4_flat.transpose(-1, -2).reshape(B, 32, 16,16)
        x5 = after_t5_flat.transpose(-1, -2).reshape(B, 48, 8,8)
        
        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
        return x1,x2,x3,x4,x5





class FrequencyFilter(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.amp_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pha_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        )
        self.channel_adjust = nn.Conv2d(in_channels, in_channels, kernel_size = 1)

    def forward(self, x):
        b, c, h, w = x.shape
        msF = fft.rfft2(x + 1e-8, norm = 'backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        amp_fuse = self.amp_mask(msF_amp) + msF_amp
        pha_fuse = self.pha_mask(msF_pha) + msF_pha

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8

        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s = (h, w), norm = 'backward'))
        out = out + x
        out =self.channel_adjust(out)
        out = torch.nan_to_num(out, nan = 1e-5, posinf = 1e-5, neginf = 1e-5)

        return out
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.conv3(x)
        out = out + residual
        return self.relu(out)


class SFma_Unet(nn.Module):
    
    def __init__(self, num_classes=4, input_channels=3, c_list=[8,16,24,32,48,64],
                split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 =SFIMoudle(input_dim=c_list[2], output_dim=c_list[3])

        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge: 
            self.scab = Full_Bridge(c_list, split_att)
            print('Full_Bridge was used')
        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        ) 
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        ) 
        self.decoder3 =SFIMoudle(input_dim=c_list[3], output_dim=c_list[2])
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        
        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        return torch.sigmoid(out0)


