import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *
from utils import obtain_act, obtain_norm


class LinearPred(nn.Module):

    def __init__(self, emb_dim, nb_classes):
        super(LinearPred, self).__init__()
        
        self.fc = nn.Linear(emb_dim, nb_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
                self.fc.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layer, kernel="gcn", drop_ratio=0, 
                    act="prelu", norm=None, emb_act=None, bias=True):
        super(Encoder, self).__init__()

        self.num_layer = num_layer
        self.emb_dim = [in_dim] + [hid_dim] * (num_layer - 1) + [out_dim]
        self.drop_ratio = drop_ratio
        self.norm = norm

        self.encs = torch.nn.ModuleList()
        self.enc_acts = torch.nn.ModuleList()
        if self.norm:
            self.enc_norms = torch.nn.ModuleList()

        for i in range(self.num_layer):
            self.encs.append(Conv(self.emb_dim[i], self.emb_dim[i + 1], kernel=kernel, bias=bias))
            
            if self.norm:
                self.enc_norms.append(obtain_norm(self.norm)(self.emb_dim[i + 1]))
            
            if i != self.num_layer - 1 or emb_act:
                self.enc_acts.append(obtain_act(act))
            else:
                self.enc_acts.append(obtain_act())

    
    def forward(self, x, edge_index, edge_weight, all=True):

        enc_list = [x]
        for layer in range(self.num_layer):
            h = self.encs[layer](enc_list[layer], edge_index, edge_weight)
            h = self.enc_norms[layer](h) if self.norm else h
            h = F.dropout(self.enc_acts[layer](h), self.drop_ratio, training = self.training)

            enc_list.append(h)

        if all:
            return enc_list[1:]
        else:
            return enc_list[-1]

    
    def fast(self, x, adj, all=True):

        enc_list = [x]
        for layer in range(self.num_layer):
            h = self.encs[layer](enc_list[layer], adj, None)
            h = self.enc_norms[layer](h) if self.norm else h
            h = F.dropout(self.enc_acts[layer](h), self.drop_ratio, training = self.training)

            enc_list.append(h)

        if all:
            return enc_list[1:]
        else:
            return enc_list[-1]


class WDecoder(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layer, gamma, kernel="gcn", 
                    drop_ratio=0, skip="True", act="prelu", norm=None, dec_aggr="max", 
                    large=False, beta=1.0):
        super(WDecoder, self).__init__()

        self.num_layer = num_layer
        self.emb_dim = [in_dim] + [hid_dim] * (num_layer - 1) + [out_dim]
        self.drop_ratio = drop_ratio

        self.skip = skip
        self.beta = beta
        self.norm = norm

        self.decs = torch.nn.ModuleList()
        self.dec_acts = torch.nn.ModuleList()
        if self.norm:
            self.dec_norms = torch.nn.ModuleList()

        for i in range(self.num_layer):
            if i == self.num_layer - 1:
                large = large and len(gamma) > 1
                self.decs.append(DeconvWiener(self.emb_dim[i], self.emb_dim[i + 1], act=act, gamma=gamma, kernel=kernel, transform=dec_aggr, large=large))
                self.dec_acts.append(obtain_act())
            else:
                self.decs.append(DeconvWiener(self.emb_dim[i], self.emb_dim[i + 1], act=act, gamma=gamma, kernel=kernel, transform=dec_aggr))
                self.dec_acts.append(obtain_act(act))

                if self.norm:
                    self.dec_norms.append(obtain_norm(self.norm)(self.emb_dim[i + 1]))


    def forward(self, enc, edge_index, edge_weight, avg_edge_index, avg_edge_weight):

        coef = enc[-1].std().item() * self.beta
        dec_list = [enc[-1] + coef * torch.normal(0, 1, size=enc[-1].size(), device=enc[-1].device)]
        
        for layer in range(self.num_layer):
            if layer == 0:
                d = self.decs[layer](dec_list[layer], edge_index, edge_weight, avg_edge_index, avg_edge_weight)
            else:
                if self.skip:
                    adv_enc = enc[-(layer+1)]
                    coef = adv_enc.std().item() * self.beta
                    adv_enc = adv_enc + coef * torch.normal(0, 1, size=adv_enc.size(), device=enc[-(layer+1)].device)
                    d1 = self.decs[layer](adv_enc, edge_index, edge_weight, avg_edge_index, avg_edge_weight)

                    adv_dec = dec_list[layer]
                    d2 = self.decs[layer](adv_dec, edge_index, edge_weight, avg_edge_index, avg_edge_weight)

                    d = d1 + d2
                else:
                    adv_dec = dec_list[layer]
                    d = self.decs[layer](adv_dec, edge_index, edge_weight, avg_edge_index, avg_edge_weight)

            d = self.dec_norms[layer](d) if self.norm and layer < self.num_layer - 1 else d
            d = F.dropout(self.dec_acts[layer](d), self.drop_ratio, training=self.training)
            dec_list.append(d)
        
        return dec_list[-1]

    
    def fast(self, enc, adj, avg_adj):
 
        coef = enc[-1].std().item() * self.beta
        dec_list = [enc[-1] + coef * torch.normal(0, 1, size=enc[-1].size(), device=enc[-1].device)]

        for layer in range(self.num_layer):
            if layer == 0:
                d = self.decs[layer](dec_list[layer], adj, None, avg_adj, None)
            else:
                if self.skip:
                    adv_enc = enc[-(layer+1)]
                    coef = adv_enc.std().item() * self.beta
                    adv_enc = adv_enc + coef * torch.normal(0, 1, size=adv_enc.size(), device=enc[-(layer+1)].device)
                    d1 = self.decs[layer](adv_enc, adj, None, avg_adj, None)

                    adv_dec = dec_list[layer]
                    d2 = self.decs[layer](adv_dec, adj, None, avg_adj, None)

                    d = d1 + d2
                else:
                    adv_dec = dec_list[layer]
                    d = self.decs[layer](adv_dec, adj, None, avg_adj, None)

            d = self.dec_norms[layer](d) if self.norm and layer < self.num_layer - 1 else d
            d = F.dropout(self.dec_acts[layer](d), self.drop_ratio, training=self.training)
            dec_list.append(d)
        
        return dec_list[-1]
    

class WGDN(nn.Module):
    
    def __init__(self, in_dim, hid_dim, num_layer, gamma, kernel="gcn", drop_ratio=0, skip=True, act="prelu", 
                    norm=None, dec_aggr="max", large=False, beta=1.0, emb_act=False, simple=False):
        super(WGDN, self).__init__()
        
        if simple:
            dec_layer = 1
        else:
            dec_layer = num_layer

        self.encoder = Encoder(in_dim, hid_dim, hid_dim, num_layer, kernel, drop_ratio, act, norm, emb_act)
        self.decoder = WDecoder(hid_dim, hid_dim, in_dim, dec_layer, gamma, kernel, drop_ratio, skip, act, 
                                norm, dec_aggr, large, beta)


    def forward(self, *argv):

        if len(argv) == 5:
            x, edge_index, edge_weight, avg_edge_index, avg_edge_weight = argv

            enc_list = self.encoder(x, edge_index, edge_weight)
            recons = self.decoder(enc_list, edge_index, edge_weight, avg_edge_index, avg_edge_weight)
        elif len(argv) == 3:
            x, lap_adj, avg_adj = argv
            
            encs = self.encoder.fast(x, lap_adj)
            recons = self.decoder.fast(encs, lap_adj, avg_adj)
        else:
            raise ValueError("unmatched number of arguments.")

        return recons

    def get_embedding(self, *argv):

        if len(argv) == 3:
            x, edge_index, edge_weight = argv

            emb = self.encoder(x, edge_index, edge_weight, all=False)
        elif len(argv) == 2:
            x, lap_adj = argv

            emb = self.encoder.fast(x, lap_adj, all=False)
        else:
            raise ValueError("unmatched number of arguments.")

        return emb
