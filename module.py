import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_sparse import matmul

from functools import partial

from utils import conv_kernel_coefs, remez_approx, wiener_kernel_func, approx_kernel_order, obtain_act


class Conv(MessagePassing):

    def __init__(self, in_dim, emb_dim, kernel="gcn", aggr="add", deconv=False, bias=True):

        # kwargs.setdefault('aggr', 'add')
        super(Conv, self).__init__(aggr=aggr)

        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.coefs = conv_kernel_coefs(kernel)
        self.deconv = deconv

        self.lin = nn.Linear(in_dim, emb_dim, bias=bias)
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            self.lin.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight):

        if not self.deconv:
            x = self.lin(x)

        # message = x.clone()
        message = x
        out = self.coefs[0]*message

        for coef in self.coefs[1:]:
            message = self.propagate(edge_index, x=message, edge_weight=edge_weight)
            out += coef * message

        if self.deconv:
            out = self.lin(out)

        return out

    def message(self, x_j, edge_weight):
        
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        
        return matmul(adj_t, x, reduce=self.aggr)


class DeconvWiener(MessagePassing):

    def __init__(self, in_dim, emb_dim, gamma, act="prelu", kernel="gcn", aggr="add", transform="max", large=False, bias=True):

        # kwargs.setdefault('aggr', 'add')
        super(DeconvWiener, self).__init__(aggr=aggr)

        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.gamma = gamma
        self.no_channel = len(gamma)
        self.transform = transform
        self.act = obtain_act(act)
        self.large = large
        
        self.order = approx_kernel_order(kernel)
        self.func = partial(wiener_kernel_func, kernel=kernel)

        self.lin = nn.ModuleList()
        for _ in range(self.no_channel):
            if self.large and self.no_channel > 1:
                lin = nn.Linear(in_dim, in_dim, bias=bias)
            else:
                if self.no_channel > 1 and self.transform == 'concat':
                    lin = nn.Linear(in_dim, emb_dim // self.no_channel, bias=bias)
                else:
                    lin = nn.Linear(in_dim, emb_dim, bias=bias)
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                lin.bias.data.fill_(0.0)
            self.lin.append(lin)
        
        if self.large and self.no_channel > 1:
            self.fuse1 = nn.Linear(in_dim, emb_dim, bias=bias)
            nn.init.xavier_uniform_(self.fuse1.weight)
            if self.fuse1.bias is not None:
                self.fuse1.bias.data.fill_(0.0)
        
        if self.no_channel > 1:
            if self.transform == "sum":
                self.fuse = partial(torch.sum, dim=0)
            elif self.transform == "avg":
                self.fuse = partial(torch.mean, dim=0)
            elif self.transform == "max":
                self.fuse = partial(torch.max, dim=0)
            else:
                raise ValueError("Invalid aggregation method!")
    

    def forward(self, x, edge_index, edge_weight, avg_edge_index, avg_edge_weight, signal_mat=None, noise_src=None):
        
        # Signal estimation
        if signal_mat is not None:
            signal = signal_mat.var(0).mean().detach() + signal_mat.pow(0).mean(1).mean().detach()
        else:
            signal = x.var(0).sum().detach() + x.pow(2).mean(0).sum().detach()
        
        # Noise estimation
        if noise_src is not None:
            noise = noise_src
        else:
            diff = x - self.propagate(avg_edge_index, x=x, edge_weight=avg_edge_weight)
            noise = diff.pow(2).mean(0).sum().detach()

        # message = x.clone()
        message = x
        coefs = [None]*self.no_channel
        outs = [None]*self.no_channel
        for i in range(self.no_channel):
            penalty = (noise/(self.gamma[i]*signal)).item()
            coefs[i] = remez_approx(self.order, 0.0, 2.0, self.func, penalty)
            outs[i] = coefs[i][0]*message

        for k in range(self.order):
            message = self.propagate(edge_index, x=message, edge_weight=edge_weight)
            for i in range(self.no_channel):
                outs[i] += coefs[i][k+1]*message

        for i in range(self.no_channel):
            # outs[i] = self.act(self.lin[i](outs[i]))
            outs[i] = self.lin[i](outs[i])
            if self.no_channel > 1:
                outs[i] = self.act(outs[i])
        
        if self.no_channel == 1:
            out = outs[0]
        else:
            if self.transform == 'concat':
                out = torch.concat(outs, dim=1)
            else:
                out = torch.stack(outs)
                if self.transform == "max":
                    out = self.fuse(out)[0]
                else:
                    out = self.fuse(out)

            if self.large:
                out = self.fuse1(out)

        return out

    def message(self, x_j, edge_weight):
        
        return edge_weight.view(-1, 1) * x_j
    
    def message_and_aggregate(self, adj_t, x):
        
        return matmul(adj_t, x, reduce=self.aggr)