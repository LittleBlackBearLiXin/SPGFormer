import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from SLIC.our_SLIC import spareaA, getimportA


class MMPN(MessagePassing):
    def __init__(self, input_dim, output_dim, A, dropout,  ln=False, aggr='add', flag=0):
        super(MMPN, self).__init__(aggr=aggr)
        if flag == 1:
            ln = True
        self.lin = nn.Linear(input_dim, output_dim)
        self.BN = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.ln = ln
        if self.ln:
            self.ln_layer = nn.LayerNorm(input_dim)
        self.edge_index, self.edge_value = self.pre_get_adjceny(A)
        self.import_edge_index, self.import_edge_value = self.pre_get_adjceny(getimportA(A, k=5, FLAG=flag))  # 5T,5F,5F

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
        layers_count = 2
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


class mumpnnn(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int,Q, A, hide=64, flag=0):
        super(mumpnnn, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.BN = nn.BatchNorm1d(hide)
        self.prelin = nn.Linear(changel, hide)
        if flag == 1:
            mmpndroup=0.1
        else:
            mmpndroup = 0.5
        self.HypMMPN = Hyperpixel_MMPN(input=hide, output=hide, Q=Q, A=A, dropout=mmpndroup, flag=flag)  # 0,0.5,0.5
        self.Softmax_linear = nn.Sequential(nn.Linear(hide, self.class_count))

    def forward(self, x: torch.Tensor):
        x = x.reshape([self.height * self.width, -1])
        x = (self.BN(self.prelin(x)))
        H1 = self.HypMMPN(x)
        Y = self.Softmax_linear(H1)
        Y = F.softmax(Y, -1)
        return Y




class COMMMPN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int,Q, A, hide=64, flag=0):
        super(COMMMPN, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.BN = nn.BatchNorm1d(hide)
        self.prelin = nn.Linear(changel, hide)
        if flag == 1:
            mmpndroup=0.1
        else:
            mmpndroup = 0.5
        self.HypMMPN = Hyperpixel_MMPN(input=hide, output=hide, Q=Q, A=A, dropout=mmpndroup, flag=flag)  # 0,0.5,0.5

    def forward(self, x: torch.Tensor):
        x = x.reshape([self.height * self.width, -1])
        x = (self.BN(self.prelin(x)))
        H1 = self.HypMMPN(x)

        return H1
