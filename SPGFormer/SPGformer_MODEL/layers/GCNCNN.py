
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def noramlize(A: torch.Tensor):
    D = A.sum(1)
    D_hat = torch.diag(torch.pow(D, -0.5))
    A = torch.mm(torch.mm(D_hat, A), D_hat)
    return A

def spareadjacency(adjacency):

    if not adjacency.is_sparse:
        adjacency = adjacency.to_sparse()

    adjacency = adjacency.coalesce()
    indices = adjacency.indices()

    values = adjacency.values()


    size = adjacency.size()
    adjacency = torch.sparse_coo_tensor(indices, values, size, dtype=torch.float32, device=device)


    return adjacency




class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):#
        super(GCNLayer, self).__init__()

        self.Activition = nn.LeakyReLU()
        self.bn= nn.BatchNorm1d(output_dim)


        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count =A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        A = noramlize(A+self.I)
        ratio=1-(torch.sum(A!=0)/(nodes_count**2))
        print(ratio)
        if ratio>90:
            self.A = spareadjacency(A)
        else:
            self.A = A



    def forward(self, H):
        output = torch.mm(self.A, self.GCN_liner_out_1(H))
        output=self.bn(output)
        output = self.Activition(output)
        return output




class SGCN(nn.Module):#1,1,
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,hide):
        super(SGCN, self).__init__()

        self.class_count = class_count

        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.prelin=nn.Linear(changel,hide)

        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        self.GCN_Branch = nn.Sequential()
        layers_count=2
        for i in range(layers_count):
            if i==0:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(hide, hide, self.A))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(hide, hide, self.A))

        self.BN = nn.BatchNorm1d(hide)

    def forward(self, x: torch.Tensor):

        x = x.reshape([self.height * self.width, -1])
        x=self.BN(self.prelin(x))

        superpixels_flatten = torch.mm(self.norm_col_Q.t(), x)
        H1=self.GCN_Branch(superpixels_flatten)

        GCN_result = torch.matmul(self.Q, H1)

        return GCN_result



class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out



class CNN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int,hide):
        super(CNN, self).__init__()
        self.class_count = class_count

        self.channel = changel
        self.height = height
        self.width = width

        layers_count = 2

        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, hide, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(hide), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(hide, hide, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(hide, hide, kernel_size=5))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(hide, hide, kernel_size=5))




    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise
        hx = clean_x
        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # spectral-spatial convolution
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        return CNN_result

