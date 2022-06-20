import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import register_model
from torch.nn.utils.parametrizations import spectral_norm
import math

@register_model("dncnn_sn")
class DnCNN_SN(nn.Module):
    """DnCNN with spectral norm"""
    def __init__(self, depth=20, n_channels=64, image_channels=1, bias=False, kernel_size=3, sn_gain=0.7, n_power_iter=1):
        super(DnCNN_SN, self).__init__()
        kernel_size = 3
        padding = 1

        self.bias = bias
        self.depth = depth
        self.sn_gain = sn_gain

        self.first_layer = spectral_norm(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=self.bias),n_power_iterations=n_power_iter)

# DOES THIS WORK ON TENSORS?  MAKE CUSTOM LAYER?
        self.first_scaling = nn.Linear(1,1)
        #parametrize.register_parametrization(self.first_scaling, "weight", Bounded(0,self.sn_gain))

        self.hidden_layer_list = [None] * (self.depth - 2)
        self.hidden_scaling_list = [None] * (self.depth - 2)
        for i in range(self.depth-2):
            self.hidden_layer_list[i] = spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=self.bias),n_power_iterations=n_power_iter) 
            self.hidden_scaling_list[i] = nn.Linear(1,1)
            #parametrize.register_parametrization(self.hidden_scaling_list[i], "weight", Bounded(0,self.sn_gain))
        self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list)
        self.hidden_scaling_list = nn.ModuleList(self.hidden_scaling_list)

        self.last_layer = spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=self.bias),n_power_iterations=n_power_iter)
        self.last_scaling = nn.Linear(1,1)
        #parametrize.register_parametrization(self.last_scaling, "weight", Bounded(0,self.sn_gain))

        self._initialize_weights()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--in-channels", type=int, default=1, help="number of channels")
        parser.add_argument("--hidden-size", type=int, default=64, help="hidden dimension")
        parser.add_argument("--num-layers", default=20, type=int, help="number of layers")
        parser.add_argument("--n-power-iter", default=1, type=int, help="power iterations for spectral norm")
        parser.add_argument("--sn-gain", default=0.7, type=float, help="gain for spectral norm")
        parser.add_argument("--bias", action='store_true', help="use bias")

    @classmethod
    def build_model(cls, args):
        return cls(image_channels = args.in_channels, n_channels = args.hidden_size, depth = args.num_layers, n_power_iter=args.n_power_iter, sn_gain=args.sn_gain, bias=args.bias)

    def forward(self, x):
        out = self.first_scaling( self.first_layer(x) );
        out = F.relu(out);

        for i in range(self.depth-2):
            out = self.hidden_scaling_list[i]( self.hidden_layer_list[i](out) );
            out = F.relu(out)

        out = self.last_scaling( self.last_layer(out) );
        
        return x-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    print("hi")
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.constant_(m.weight, 0.025)
#                 m.weight = 0.025 # inspired by DnCNN's batch-norm weight initialization 


#@register_model("bounded") # needed?
class Bounded(nn.Module):
    def __init__(self, lower=0, upper=1):
        super(Bounded, self).__init__()
        self.lower = lower
        self.upper = upper
        
    def forward(self,x):
        x = F.threshold(x,self.lower,self.lower) # lower thresholding 
        x = -F.threshold(-x,-self.upper,-self.upper) # upper thresholding
    
