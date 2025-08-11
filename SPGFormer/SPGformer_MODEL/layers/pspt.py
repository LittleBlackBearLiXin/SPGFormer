import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from SLIC.our_SLIC import spareaA, getimportA
from torch_scatter.composite import scatter_softmax


class Pixel_spareformer(nn.Module):
    def __init__(self, input_dim, output_dim, rowmask, colmask, dropout, use_dynamic_attention=False, ln=True,
                 concat=False):
        super(Pixel_spareformer, self).__init__()
        self.concat = concat
        self.input_dim = input_dim
        if self.concat:
            self.output_dim = output_dim // 2
        else:
            self.output_dim = output_dim
        self.rowlinv = nn.Linear(self.input_dim, self.output_dim)
        self.collinv = nn.Linear(self.input_dim, self.output_dim)
        self.BN = nn.BatchNorm1d(self.input_dim)
        self.use_dynamic_attention = use_dynamic_attention
        self.dropout = dropout
        self.ln = ln

        if self.use_dynamic_attention:
            self.rowlinq = nn.Linear(self.input_dim, self.output_dim)
            self.collinq = nn.Linear(self.input_dim, self.output_dim)
        if self.ln:
            self.ln = nn.LayerNorm(self.input_dim)
        self.rowsrc, self.rowdst, self.row_att_score = self.prefunction(rowmask)
        self.colsrc, self.coldst, self.col_att_score = self.prefunction(colmask)
        self.reset_parameters()

    def reset_parameters(self):
        self.rowlinv.reset_parameters()
        self.collinv.reset_parameters()
        self.BN.reset_parameters()
        if self.use_dynamic_attention:
            self.rowlinq.reset_parameters()
            self.collinq.reset_parameters()
        if self.ln:
            self.ln.reset_parameters()

    def forward(self, x):
        if self.ln:
            x = self.ln(x)
        rowv = self.rowlinv(x)
        colv = self.collinv(x)
        row_att_score, col_att_score = self.get_attscore(x)
        rowagg = rowv[self.rowdst] * row_att_score
        colagg = colv[self.coldst] * col_att_score
        rowout = scatter(rowagg, self.rowsrc, dim=0, dim_size=rowv.size(0), reduce='sum')  # head1
        colout = scatter(colagg, self.colsrc, dim=0, dim_size=colv.size(0), reduce='sum')  # head2
        if self.concat:
            out = torch.cat((rowout, colout), dim=1)
        else:
            out = rowout + colout
        out = self.BN(out+x)
        out = F.leaky_relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out

    def prefunction(self, mask):
        mask_index = mask._indices()
        src, dst = mask_index[0], mask_index[1]
        att_score = mask.values()
        return src, dst, att_score.view(-1, 1)

    def get_attscore(self, x):
        if self.use_dynamic_attention:
            rowQ = self.rowlinq(x)
            colQ = self.collinq(x)
            rowQ = F.layer_norm(rowQ, [rowQ.size(-1)])
            colQ = F.layer_norm(colQ, [colQ.size(-1)])
            row_att_score = scatter_softmax(torch.sum((rowQ[self.rowsrc] * rowQ[self.rowdst]), dim=1) /self.output_dim,
                                            self.rowsrc, dim=0)
            col_att_score = scatter_softmax(torch.sum((colQ[self.colsrc] * colQ[self.coldst]), dim=1) / self.output_dim,
                                            self.colsrc, dim=0)
            return row_att_score.view(-1, 1), col_att_score.view(-1, 1)
        else:
            return self.row_att_score, self.col_att_score




class Pspaformer(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, rowmask, colmask, hide=64,layers_count=0):
        super(Pspaformer, self).__init__()
        dropout = 0  # 0,0.5,0.2
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.BN = nn.BatchNorm1d(hide)
        self.prelin = nn.Linear(changel, hide)
        self.Pixel_spareformer_Branch = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.Pixel_spareformer_Branch.add_module('Pixel_Branch' + str(i),
                                                         Pixel_spareformer(input_dim=hide, output_dim=hide,
                                                                           rowmask=rowmask, colmask=colmask,
                                                                           dropout=dropout))
            else:
                self.Pixel_spareformer_Branch.add_module('Pixel_Branch' + str(i),
                                                         Pixel_spareformer(input_dim=hide, output_dim=hide,
                                                                           rowmask=rowmask, colmask=colmask,
                                                                           dropout=dropout))

        self.Softmax_linear = nn.Sequential(nn.Linear(hide, self.class_count))

    def forward(self, x: torch.Tensor):
        x = x.reshape([self.height * self.width, -1])
        x = (self.BN(self.prelin(x)))
        H1 =self.Pixel_spareformer_Branch(x)
        Y = self.Softmax_linear(H1)
        Y = F.softmax(Y, -1)
        return Y


class PSPT(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, rowmask, colmask, hide=64,layers_count=0):
        super(PSPT, self).__init__()
        dropout = 0  # 0,0.5,0.2
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.BN = nn.BatchNorm1d(hide)
        self.prelin = nn.Linear(changel, hide)
        self.Pixel_spareformer_Branch = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.Pixel_spareformer_Branch.add_module('Pixel_Branch' + str(i),
                                                         Pixel_spareformer(input_dim=hide, output_dim=hide,
                                                                           rowmask=rowmask, colmask=colmask,
                                                                           dropout=dropout))
            else:
                self.Pixel_spareformer_Branch.add_module('Pixel_Branch' + str(i),
                                                         Pixel_spareformer(input_dim=hide, output_dim=hide,
                                                                           rowmask=rowmask, colmask=colmask,
                                                                           dropout=dropout))

    def forward(self, x: torch.Tensor):
        x = x.reshape([self.height * self.width, -1])
        x = (self.BN(self.prelin(x)))
        H1 =self.Pixel_spareformer_Branch(x)

        return H1
