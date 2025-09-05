import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from spatial_correlation_sampler import SpatialCorrelationSampler
import time
from .pasa import Downsample
from .non_local import NLBlockND

__all__ = ['ResNeXt', 'resnet50', 'resnet101']

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool2d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None, filter_size=3,kernel_size=3, num_heads = 8,image_size= 224,inference= False,rdropout = 0.2):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        #self.conv1 = conv1x1(inplanes, mid_planes, stride)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        '''
        self.conv2 = nn.Conv2d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        '''
        
        if(stride==1):
            self.conv2 = conv3x3(mid_planes, mid_planes)
        else:
            self.conv2 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=mid_planes),
                conv3x3(mid_planes, mid_planes),)
        
        # print('stride {}'.format(stride))
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.dropout = nn.Dropout(p=rdropout, inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        
    def forward(self, x):

        residual = x
        #if x.shape[0] == 20 or x.shape[0] == 40 or x.shape[0] == 30  or x.shape[0] == 25:
        #    x = self.tsm1(x)
        #else:
        #    x = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.dropout(out) #https://arxiv.org/pdf/1801.05134  
        out += residual

        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                image_size = 224,
                num_heads = 8,
                kernel_size = 3,
                rdropout = 0.2,
                k_=21):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding = 3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1_1_def = nn.Conv2d(2, 2*7*7, kernel_size=7, stride=2, padding = 3,
                               bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU(inplace=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality, num_heads=num_heads, kernel_size=kernel_size, image_size=image_size//4)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2, filter_size=3,num_heads=num_heads, kernel_size=kernel_size, image_size=image_size//8)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2, filter_size=3,num_heads=num_heads, kernel_size=kernel_size, image_size=image_size//16)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2, filter_size=3,num_heads=num_heads, kernel_size=kernel_size, image_size=image_size//32)
        
        self.corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.conv_redir = conv(True,k_*k_,256,kernel_size=1, stride=1)
        self.conv_redir2 =  conv(True,256,64,kernel_size=1, stride=1)

        
        self.corr= SpatialCorrelationSampler(kernel_size=1,
                                             patch_size=k_,
                                             stride=1,
                                             padding=0,
                                             dilation=1,
                                             dilation_patch=2)
        
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsample = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.upsample_decon = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1,output_padding=1)
        self.avgpool = nn.AvgPool2d( (7, 7), stride=2)
        self.conv2 = nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        self.dropout1 = nn.Dropout(p=0.25, inplace=False)
        
        self.attention = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, ResNeXtBottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1,filter_size=3, non_local=False, num_heads=8, kernel_size=3, image_size=224,inference=False,rdropout = 0.2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = [Downsample(filt_size=filter_size, stride=stride, channels=self.inplanes),] if(stride !=1) else []
                downsample += [conv1x1(self.inplanes, planes * block.expansion, 1), nn.BatchNorm2d(planes * block.expansion)]
                # print(downsample)
                downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample,num_heads=num_heads, kernel_size=kernel_size, image_size=image_size, inference=inference,rdropout = 0.2))
        self.inplanes = planes * block.expansion
        
        last_idx = blocks
        if non_local:
            last_idx = blocks - 1
            
        for i in range(1, last_idx):
            layers.append(block(self.inplanes, planes,cardinality))
        if non_local:
            layers.append(NLBlockND(self.inplanes, mode='gaussian', dimension=2))
            layers.append(block(self.inplanes, planes,cardinality))

        return nn.Sequential(*layers)

    def encode(self, x):
        h1 = self.relu(x)
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        show_size = False
        # show_size = True
        if show_size:
            print('input shape {}'.format(x.shape))
            x = self.conv1(x)
            print('conv1 shape {}'.format(x.shape))
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            # x = self.conv_pool(x)
            print('maxpool shape {}'.format(x.shape))

            x = self.layer1(x)
            print('layer1 shape {}'.format(x.shape))
            # time.sleep(30)
            x = self.layer2(x)
            print('layer2 shape {}'.format(x.shape))
            x = self.layer3(x)
            print('layer3 shape {}'.format(x.shape))
            x = self.layer4(x)
            print('layer4 shape {}'.format(x.shape))

            at_map = self.attention(x)
            print('attention_shape {}'.format(at_map.shape))
            x = x * at_map
            print('x*at shape {}'.format(x.shape))
            mp = self.relu(x)
            print('x relu {}'.format(x.shape))

            x = self.avgpool(mp)
            print('avgpool shape {}'.format(x.shape))

            x = x.view(x.size(0), -1)
            print('flatten shape {}'.format(x.shape))


            x = self.fc(x)
            print('output shape {}'.format(x.shape))
            time.sleep(30)
        else:
            x_ = x
            x = self.conv1_1(x)


            x1 = x_[:,:int(x_.shape[1]/2),:,:]
            x2 = x_[:,int(x_.shape[1]/2):,:,:]
            xa = self.conv1a(x1)
            xa = self.bn1(xa)
            xa = self.relu(xa)
            xa = self.maxpool(xa)
            xb = self.conv1a(x2)
            xb = self.bn1(xb)
            xb = self.relu(xb)
            xb = self.maxpool(xb)
            out_correlation = self.corr(xa,xb)
            #use fully connected layer here
            b, ph, pw, h, w = out_correlation.size()
            out_correlation= out_correlation.view(b, ph * pw, h, w)/xa.size(1)
            out_correlation = self.corr_activation(out_correlation)
            
            out_correlation = self.conv_redir(out_correlation)
            out_correlation = self.conv_redir2(out_correlation)
            # or down to size of xa and xb then add to xa and xb
            #x = self.conv1_1(x)

            out_correlation = self.upsampling(out_correlation)
            x = x+out_correlation
            x = self.bn1_1(x)
            x = self.relu_1(x)
            x = self.maxpool_1(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            at_map = self.attention(x)
            x = x * at_map
            mp = self.relu(x)

            x = self.avgpool(mp)
            vec = x.view(x.size(0), -1)

            x = self.fc(vec)

            #return x,vec
            return x

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
