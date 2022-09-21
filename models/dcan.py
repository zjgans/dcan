import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet
from models.dpta import Self_Dynamic_Prototype

from ddf.ddf import DDFPack

class DCANet(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.num_proto = args.proto_size
        self.num_spt = args.way * args.shot

        self.encoder = ResNet(args=args)
        self.encoder_dim = 640
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)
        self.ddf = DDFPack(in_channels=640)

        self.dynamic_prototype = Self_Dynamic_Prototype(args.proto_size,args, 640, 320, tem_update=0.1, temp_gather=0.1)
        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(self.encoder_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.lifddf = LIFDDFilter(dim=640)

        self.eq_head = nn.Sequential(
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 4)
        )

    def forward(self, input,aux=False):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            if aux:
                return self.encode(input,aux)
            else:
                return self.encode(input)

        elif self.mode == 'coda':
            spt, qry = input
            return self.coda(spt, qry)

        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        logits = self.fc(x)
        logits_eq = self.eq_head(x)
        return logits,logits_eq

    def coda(self, spt, qry):

        spt = spt.squeeze(0)

        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)

        # (S * C * Hs * Ws, Q * C * Hq * Wq) -> Q * S * Hs * Ws * Hq * Wq
        corr4d = self.get_cross_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()

        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)

        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)

        # applying softmax for each side
        corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)

        # suming up matching scores
        attn_s = corr4d_s.sum(dim=[4, 5])
        attn_q = corr4d_q.sum(dim=[2, 3])

        # applying attention
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)

        # averaging embeddings for k > 1 shots
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        spt_attended = spt_attended.view(-1,*spt_attended.shape[2:])
        spt_attended = self.ddf(spt_attended)
        # spt_attended,qry_attended = self.lifddf(spt_attended,qry_attended)
        spt_attended = spt_attended.view(num_qry,self.args.way,*spt_attended.shape[1:])
        # qry_attended = qry_attended.view(num_qry,self.args.way,*qry_attended.shape[1:])

        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])

        qry_pooled = qry.mean(dim=[-1, -2])

        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_pooled)
        else:
            return similarity_matrix / self.args.temperature

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def get_cross_correlation_map(self, spt, qry):

        way = spt.shape[0]
        num_qry = qry.shape[0]

        # reduce channel size via 1x1 conv
        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        # num_way * C * H_p * W_p --> num_qry * way * H_p * W_p
        # num_qry * C * H_q * W_q --> num_qry * way * H_q * W_q
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        return similarity_map_einsum

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def encode(self, x, aux=False):
        x = self.encoder(x)

        if self.training:
            if aux:
                return x
            else:
                update_x,fea_loss,cst_loss,dis_loss = self.dynamic_prototype(x,x)
                return update_x,fea_loss,cst_loss,dis_loss
        else:
            update_x = self.dynamic_prototype(x,x)
            return update_x

