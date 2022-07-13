import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
def one_hot_encode(labels_train):
    labels_train = labels_train.cpu()
    nKnovel = 1 + labels_train.max()
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    label_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size)-1,labels_train_unsqueeze,1)
    return label_train_1hot

class DistillKL(nn.Module):

    def __init__(self,T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self,y_s,y_t):
        p_s = F.log_softmax(y_s / self.T,dim=1)
        p_t = F.softmax(y_t / self.T,dim=1)
        loss = F.kl_div(p_s,p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def rotrate_concat(inputs):
    out = None
    for x in inputs:
        x_90 = x.transpose(2,3).flip(2)
        x_180 = x.flip(2).flip(3)
        x_270 = x.flip(2).transpose(2,3)
        if out is None:
            out = torch.cat((x, x_90, x_180, x_270),0)
        else:
            out = torch.cat((out, x, x_90, x_180, x_270),0)
    return out

def record_data(epoch,train_acc,train_ic,val_acc,val_ic,save_path):
    if epoch==0:
        df = pd.DataFrame(columns=['epoch','train_acc','train_ic','val_acc','val_ic'])
        df.to_csv(save_path,index=False)
    list = [epoch,train_acc,train_ic,val_acc,val_ic]
    data = pd.DataFrame([list])
    data.to_csv(save_path,mode='a',header=False,index=False)

