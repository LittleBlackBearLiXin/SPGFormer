import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.GCNCNN import CNN,SGCN
from layers.SPGformer import SPG
from layers.mmpn import COMMMPN
from layers.pspt import PSPT

class Combine(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, rowmask, colmask, Q, A, hide=128, flag=0,model="CNNMPNN"):
        super(Combine, self).__init__()
        self.flag=flag
        self.model = model
        layers_count = 2  # 2,2,2
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        if  self.model=="CNNMPNN":
            print("CNN+MMPN")
            self.MMPN = COMMMPN(height, width, changel, class_count,Q, A, hide, flag)  # 0,0.5,0.5
            self.CNN = CNN(height,width,changel,class_count,hide)
        elif self.model=="CNNPSPT":
            print("CNN+PSPT")
            self.CNN = CNN(height, width, changel, class_count,hide)
            self.PSPT = PSPT(height, width, changel, class_count, rowmask, colmask, hide,layers_count)
        elif  self.model=="SGCNPSPT":
            print("PSPT+SGCN")
            self.SGCN = SGCN( height, width, changel, class_count, Q, A,hide)
            self.PSPT = PSPT(height, width, changel, class_count, rowmask, colmask, hide, layers_count)

        elif  self.model=="SGCNCNN":
            print("CNN+SGCN")
            self.SGCN = SGCN( height, width, changel, class_count, Q, A,hide)
            self.CNN = CNN(height, width, changel, class_count,hide)

        self.Softmax_linear = nn.Sequential(nn.Linear(hide, self.class_count))

    def forward(self, x: torch.Tensor):
        if self.model=="CNNMPNN":
            if self.flag==3:
                H1 = self.CNN(x)*0.005 + self.MMPN(x)
            else:
                H1 = self.CNN(x)*0.1 + self.MMPN(x)
        elif self.model=="CNNPSPT":
            if self.flag == 1:
                H1 = self.CNN(x)*0.005 + self.PSPT(x)
            else:
                H1 = self.CNN(x) + self.PSPT(x)
        elif self.model=="SGCNPSPT":
            H1 = self.SGCN(x) + self.PSPT(x)
        elif self.model=="SGCNCNN":
            H1 = self.SGCN(x) + self.CNN(x)
        else:
            H1 = 0
        Y = self.Softmax_linear(H1)
        Y = F.softmax(Y, -1)
        return Y
