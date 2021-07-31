import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class Generator(nn.Module):
    def __init__(self, init_nc=32):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=init_nc, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc),
            nn.ReLU(),
            nn.Conv2d(in_channels=init_nc, out_channels=init_nc, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc),
            nn.ReLU()
        )
        self.pool0 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc, out_channels=init_nc, kernel_size=3, padding=(1, 1), stride=2),
            nn.InstanceNorm2d(init_nc),
            nn.ReLU()
        )
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc, out_channels=init_nc * 2, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=init_nc * 2, out_channels=init_nc * 2, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 2),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc * 2, out_channels=init_nc * 2, kernel_size=3, padding=(1, 1), stride=2),
            nn.InstanceNorm2d(init_nc * 2),
            nn.ReLU()
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc * 2, out_channels=init_nc * 4, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=init_nc * 4, out_channels=init_nc * 4, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 4),
            nn.ReLU()
        )
        self.pool2 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc * 4, out_channels=init_nc * 4, kernel_size=3, padding=(1, 1), stride=2),
            nn.InstanceNorm2d(init_nc * 4),
            nn.ReLU()
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc * 4, out_channels=init_nc * 8, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=init_nc * 8, out_channels=init_nc * 8, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 8),
            nn.ReLU()
        )
        self.pool3 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc * 8, out_channels=init_nc * 8, kernel_size=3, padding=(1, 1), stride=2),
            nn.InstanceNorm2d(init_nc * 8),
            nn.ReLU()
        )

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=init_nc * 8, out_channels=init_nc * 16, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 16),
            nn.ReLU(),
            nn.Conv2d(in_channels=init_nc * 16, out_channels=init_nc * 16, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 16),
            nn.ReLU()
        )

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(in_channels=init_nc * 16, out_channels=init_nc * 8, kernel_size=2, stride=2)
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc * 16, out_channels=init_nc * 8, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=init_nc * 8, out_channels=init_nc * 8, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 8),
            nn.ReLU()
        )

        self.upsample1 = nn.ConvTranspose2d(in_channels=init_nc * 8, out_channels=init_nc * 4, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc * 8, out_channels=init_nc * 4, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=init_nc * 4, out_channels=init_nc * 4, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 4),
            nn.ReLU()
        )

        self.upsample2 = nn.ConvTranspose2d(in_channels=init_nc * 4, out_channels=init_nc * 2, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc * 4, out_channels=init_nc * 2, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=init_nc * 2, out_channels=init_nc * 2, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc * 2),
            nn.ReLU()
        )

        self.upsample3 = nn.ConvTranspose2d(in_channels=init_nc * 2, out_channels=init_nc, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=init_nc * 2, out_channels=init_nc, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc),
            nn.ReLU(),
            nn.Conv2d(in_channels=init_nc, out_channels=init_nc, kernel_size=3, padding=(1, 1)),
            nn.InstanceNorm2d(init_nc),
            nn.ReLU()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=init_nc, out_channels=3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(self.pool0(e0))
        e2 = self.enc_conv2(self.pool1(e1))
        e3 = self.enc_conv3(self.pool2(e2))

        # bottleneck
        b = self.bottleneck_conv(self.pool3(e3))

        # decoder
        d0 = self.dec_conv0(torch.cat([e3, self.upsample0(b)], 1))
        d1 = self.dec_conv1(torch.cat([e2, self.upsample1(d0)], 1))
        d2 = self.dec_conv2(torch.cat([e1, self.upsample2(d1)], 1))
        d3 = self.dec_conv3(torch.cat([e0, self.upsample3(d2)], 1))
        final = self.final_conv(d3)
        return final


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # это константа. Убираем ее из дерева вычеслений
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def gram_matrix(self, input):
        batch_size, h, w, f_map_num = input.size()  # batch size(=1)
        # b=number of feature maps
        # (h,w)=dimensions of a feature map (N=h*w)

        features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(batch_size * h * w * f_map_num)


class VariationLoss(nn.Module):
    def __init__(self):
        super(VariationLoss, self).__init__()
        self.loss = 0

    def forward(self, input):
        a = F.mse_loss(input[:, :, :-1, :-1], input[:, :, 1:, :-1])
        b = F.mse_loss(input[:, :, :-1, :-1], input[:, :, :-1, 1:])
        self.loss = a+b
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std