import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from SLIC.our_SLIC import spareaA, getimportA
from torch_scatter.composite import scatter_softmax


class Pixel_spareformer(nn.Module):
    def __init__(self, input_dim, output_dim, rowmask, colmask, dropout, use_dynamic_attention=True, ln=False,
                 concat=True):
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
        out = self.BN(out)
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


class MMPN(MessagePassing):
    def __init__(self, input_dim, output_dim, A, dropout,  ln=False, aggr='add', flag=0):
        super(MMPN, self).__init__(aggr=aggr)
        self.lin = nn.Linear(input_dim, output_dim)
        self.BN = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.ln = ln
        if self.ln:
            self.ln_layer = nn.LayerNorm(input_dim)
        self.edge_index, self.edge_value = self.pre_get_adjceny(A)
        self.import_edge_index, self.import_edge_value = self.pre_get_adjceny(getimportA(A, k=6, FLAG=flag))  # 6T,5F,5F

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.BN.reset_parameters()
        if self.ln:
            self.ln_layer.reset_parameters()

    def forward(self, x):
        if self.ln:
            x = self.ln_layer(x)
        h = self.lin(x)
        output = self.propagate(edge_index=self.edge_index, x=h, edge_weight=self.edge_value) + \
                 self.propagate(edge_index=self.import_edge_index, x=x, edge_weight=self.import_edge_value)
        output = self.BN(output)
        output = F.leaky_relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def pre_get_adjceny(self, A):
        A = spareaA(A)
        A = A.coalesce()
        edge_index = A._indices()
        edge_value = A.values()
        return edge_index, edge_value






class Hyperpixel_MMPN(nn.Module):  #
    def __init__(self, input, output, Q, A, dropout, flag=0):
        super(Hyperpixel_MMPN, self).__init__()
        self.Q = Q
        self.A = A
        self.norm_col_Q = (Q / (torch.sum(Q, dim=0, keepdim=True)))
        layers_count = 5
        self.MMPN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.MMPN_Branch.add_module('GCN_Branch' + str(i),
                                            MMPN(input_dim=input, output_dim=output, A=self.A, dropout=dropout, flag=flag))
            else:
                self.MMPN_Branch.add_module('GCN_Branch' + str(i),
                                            MMPN(input_dim=output, output_dim=output, A=self.A, dropout=dropout, flag=flag))

    def forward(self, x):
        Hyperpixel = torch.mm(self.norm_col_Q.t(), x)
        h = self.MMPN_Branch(Hyperpixel)
        result = torch.mm(self.Q, h)
        return result


class SPGformer(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, rowmask, colmask, Q, A, hide=128, flag=0):
        super(SPGformer, self).__init__()
        layers_count = 2  # 2,2,2
        dropout = 0.5  # 0.5,0.5,0.5
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.BN = nn.BatchNorm1d(hide)
        self.prelin = nn.Linear(changel, hide)
        if flag == 1:
            mmpndroup=0
        else:
            mmpndroup = 0.5
        self.HypMMPN = Hyperpixel_MMPN(input=hide, output=hide, Q=Q, A=A, dropout=mmpndroup, flag=flag)  # 0,0.5,0.5
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
        H1 = self.HypMMPN(x) + self.Pixel_spareformer_Branch(x)
        Y = self.Softmax_linear(H1)
        Y = F.softmax(Y, -1)
        return Y




class SPG(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, rowmask, colmask, Q, A, hide=128, flag=0):
        super(SPG, self).__init__()
        layers_count = 2  # 2,2,2
        dropout = 0.5  # 0.5,0.5,0.5
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.BN = nn.BatchNorm1d(hide)
        self.prelin = nn.Linear(changel, hide)
        if flag == 1:
            mmpndroup=0
        else:
            mmpndroup = 0.5
        self.HypMMPN = Hyperpixel_MMPN(input=hide, output=hide, Q=Q, A=A, dropout=mmpndroup, flag=flag)  # 0,0.5,0.5
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
        H1 = self.HypMMPN(x) + self.Pixel_spareformer_Branch(x)

        return H1