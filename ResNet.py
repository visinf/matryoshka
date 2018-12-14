import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import log, floor


class ResNet(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, num_penultimate_channels, \
        input_resolution, output_resolution, num_initial_channels=16, num_inner_channels=64, \
        num_downsampling=3, num_blocks=6):

        assert num_blocks >= 0

        super(ResNet, self).__init__()

        relu = nn.ReLU(True)
       
        model = [nn.BatchNorm2d(num_input_channels, True)]

        # additional down and upsampling blocks to account for difference in input/output resolution
        num_additional_down   = int(log(input_resolution / output_resolution,2)) if output_resolution <= input_resolution else 0
        num_additional_up     = int(log(output_resolution / input_resolution,2)) if output_resolution >  input_resolution else 0

        # number of channels to add during downsampling
        num_channels_down     = int(floor(float(num_inner_channels - num_initial_channels)/(num_downsampling+num_additional_down)))

        # adjust number of initial channels
        num_initial_channels += (num_inner_channels-num_initial_channels) % num_channels_down

        # initial feature block
        model += [nn.ReflectionPad2d(1),
            nn.Conv2d(num_input_channels, num_initial_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_initial_channels),
            relu]
        model += [nn.ReflectionPad2d(1),
            nn.Conv2d(num_initial_channels, num_initial_channels, kernel_size=3, padding=0)]

        # downsampling
        for i in range(num_downsampling+num_additional_down):                        
            model += [ResDownBlock(num_initial_channels, num_channels_down)]
            model += [ResSameBlock(num_initial_channels+num_channels_down)]
            num_initial_channels += num_channels_down
            pass

        # inner blocks at constant resolution
        for i in range(num_blocks):
            model += [ResSameBlock(num_initial_channels)]
            pass

        num_channels_up = int(floor(float(num_initial_channels - num_penultimate_channels)/(num_downsampling+num_additional_up)))

        # upsampling
        for i in range(num_downsampling+num_additional_up):
            model += [ResUpBlock(num_initial_channels, num_channels_up)]
            model += [ResSameBlock(num_initial_channels-num_channels_up)]
            num_initial_channels -= num_channels_up
            pass

        model += [nn.Conv2d(num_initial_channels, num_output_channels, kernel_size=3,padding=1)]
        
        self.model = nn.Sequential(*model)
        pass

        
    def forward(self, input):
        return self.model(input)        
    pass


class ResSameBlock(nn.Module):
    """ ResNet block for constant resolution.
    """
    
    def __init__(self, dim):
        super(ResSameBlock, self).__init__()

        self.model = nn.Sequential(*[nn.BatchNorm2d(dim, True), \
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),            
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)])

    def forward(self, x):
        return x + self.model(x)
    pass    


class ResUpBlock(nn.Module):
    """ ResNet block for upsampling.
    """

    def __init__(self, dim, num_up):
        super(ResUpBlock, self).__init__()
        
        self.model = nn.Sequential(*[nn.BatchNorm2d(dim, True),\
            nn.ReLU(False),
            nn.ConvTranspose2d(dim, -num_up+dim, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(-num_up+dim, True),
            nn.ReLU(True),
            nn.Conv2d(-num_up+dim, -num_up+dim, kernel_size=3, padding=1)])

        self.project = nn.Conv2d(dim,dim-num_up,kernel_size=1)
        pass

    def forward(self, x):        
        # xu = F.upsample(x,scale_factor=2,mode='nearest')
        xu = F.interpolate(x, scale_factor=2, mode='nearest')
        bs,_,h,w = xu.size()
        return self.project(xu) + self.model(x)
    pass


class ResDownBlock(nn.Module):
    """ ResNet block for downsampling.
    """
    
    def __init__(self, dim, num_down):
        super(ResDownBlock, self).__init__()
        self.num_down = num_down
        
        self.model = nn.Sequential(*[nn.BatchNorm2d(dim, True), \
            nn.ReLU(False),
            nn.Conv2d(dim, num_down+dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_down+dim, True),
            nn.ReLU(True),
            nn.Conv2d(num_down+dim, num_down+dim, kernel_size=3, padding=1)])
        pass

    def forward(self, x):
        xu = x[:,:,::2,::2]
        bs,_,h,w = xu.size()
        return torch.cat([xu, x.new_zeros(bs, self.num_down, h, w, requires_grad=False)],1) + self.model(x)
    pass
