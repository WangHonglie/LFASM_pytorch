import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import pretrainedmodels
from resnet_cbam import *
from layers import (
        CMul,
        Flatten,
        ConcatTable,
        Identity,
        Reshape,
        SpatialAttention2d,
        WeightedSum2d)

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

class SegMaskNet(nn.Module):
    def __init__(self, class_num, pretrained=False):
        super(SegMaskNet, self).__init__()
        #trunk
        # trunk_bone = models.resnet50(pretrained=False)
        trunk_bone = resnet50_cbam(pretrained=True)
        self.trunk0 = nn.Sequential(
                    trunk_bone.conv1,
                    trunk_bone.bn1,
                    trunk_bone.relu)
        self.trunk_pool0 = trunk_bone.maxpool
        self.trunk_split = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True))

        self.trunk1 = nn.Sequential(
                    trunk_bone.layer1,
                    trunk_bone.layer2,
                    trunk_bone.layer3,
                    trunk_bone.layer4)

        self.trunk_avgpool = nn.AvgPool2d(8)
        self.trunk_fc = nn.Linear(2048, class_num)# 50->2048 34->512

        ## mask
        # resnet = models.resnet34(pretrained=False)
        resnet = resnet34_cbam(pretrained=False)

        self.layer0 = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu)
        self.pool0 = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.decoder4 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True));

        self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(256+256, 128, 2, stride=2, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True));

        self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(128+128, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True));

        self.bn2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True));

        self.attn = SpatialAttention2d(128,act_fn='relu')
        self.attn_pool = WeightedSum2d()
        self.localization = nn.Sequential(
                        nn.Conv2d(3, 8, kernel_size=7),
                        nn.MaxPool2d(2, stride=2),
                        nn.ReLU(True),
                        nn.Conv2d(8, 10, kernel_size=5),
                        nn.MaxPool2d(2, stride=2),
                        nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
                        nn.Linear(10 * 55 * 55, 32),
                        nn.ReLU(True),
                        nn.Linear(32, 3 * 2)

        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
         xs = self.localization(x)
         xs = xs.view(-1, 10 * 55 * 55)
         theta = self.fc_loc(xs)
         theta = theta.view(-1, 2, 3)
         print(theta)
         exit()
         grid = F.affine_grid(theta, x.size())
         x = F.grid_sample(x, grid)

         return x

    def forward(self, x, cnt=0):
        batchSize = x.size()[0]

        ori = x
        x = self.stn(x)

        l0 = self.layer0(x)
        l1 = self.pool0(l0)
        l1 = self.layer1(l1)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        d4 = self.decoder4(l4)
        d4 = torch.cat((l3,d4), 1)
        d3 = self.decoder3(d4)
        d3 = torch.cat((l2,d3), 1)

        d2 = self.decoder2(d3)
        bn2 = self.bn2(l1)
        d2 = torch.cat((bn2,d2), 1)

        attn_score = self.attn(d2)

        gf = self.trunk0(ori)
        gf = self.trunk_pool0(gf)
        gf = self.trunk_split(gf)

        lf = self.attn_pool([gf, attn_score])
        gl = self.trunk1(torch.cat((gf,lf), 1))
        gl = self.trunk_avgpool(gl)
        feature = gl.view( gl.size(0), -1)


        featureOut = self.trunk_fc(feature)

        return gl, featureOut

class NONLocalNet(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super(NONLocalNet, self).__init__()


        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
        self.fc = nn.Linear(2048, 2)# 50->2048 34->512

    def forward(self, y, x):
        '''
        :param x, y: (b, c, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_y = self.theta(y).view(batch_size, self.inter_channels, -1)
        theta_y = theta_y.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_y, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        z = torch.matmul(f_div_C, g_x)
        z = z.permute(0, 2, 1).contiguous()
        z = z.view(batch_size, self.inter_channels, *x.size()[2:])
        W_z = self.W(z).view(batch_size,-1)
        out = self.fc(W_z)
        # out = W_z + x

        return out

class SelfMatchNet(nn.Module):
    def __init__(self,class_num):
        super(SelfMatchNet, self).__init__()
        self.extractor = SegMaskNet(class_num)
        self.match = NONLocalNet(in_channels=2048)
    # def forward(self, x, y):
    #     f_x, out_x = self.extractor(x)
    #     f_y, out_y = self.extractor(y)
    #     match_out = self.match(f_x,f_y)
    #     return out_x, out_y, match_out
    def forward(self, x):
        f_x, out_x = self.extractor(x)
        f_y, out_y = self.extractor(x)
        match_out = self.match(f_x,f_y)
        return out_x, out_y, match_out

'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it.
    net = ft_net(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(output.shape)
