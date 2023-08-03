from audioop import bias
from pprint import pformat
from turtle import ht
import torch
from torch import nn
from torch.nn import init
from torch_geometric.utils import dense_to_sparse,contains_self_loops,to_dense_adj,to_undirected,to_dense_batch
#from GATconv import GATConv
from torch_geometric.nn.conv import GATConv
import numpy as np
import dgl
#from dgl.nn import EGATConv
from EGATConv import EGATConv,glorot,zeros
from torch_geometric.nn.dense.linear import Linear
from Utils import AdjProcessor
from multiheadattn import MultiheadAttention
from torch_geometric.nn import knn_graph


class ODconv(nn.Module):        # Origin-Destination Convolution
    def __init__(self, K:int, input_dim:int, hidden_dim:int, layer:int, use_bias=True, activation=None):
        super(ODconv, self).__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.head=4
        self.activation = activation() if activation is not None else None
        self.layer=layer
        if layer==0:       
            self.gat = EGATConv(12,input_dim,int(12*self.hidden_dim/8),int(input_dim/2),self.head,True)
            self.linear0 = Linear(input_dim, int(hidden_dim/2), bias=False, weight_initializer='glorot')
            self.b0 = nn.Parameter(torch.empty(int(self.hidden_dim/2)), requires_grad=True)
            zeros(self.b0)
            
            self.linear4 = Linear(input_dim, int(hidden_dim/2), bias=False, weight_initializer='glorot')
            self.b4 = nn.Parameter(torch.empty(int(self.hidden_dim/2)), requires_grad=True)
            zeros(self.b4)

        elif layer==1:
            self.linear1 = Linear(input_dim, int(hidden_dim/1), bias=False, weight_initializer='glorot')
            self.b1 = nn.Parameter(torch.empty(int(hidden_dim/1)), requires_grad=True)
            zeros(self.b1)

            self.linear3 = Linear(input_dim, int(hidden_dim/1), bias=False, weight_initializer='glorot')
            self.b3 = nn.Parameter(torch.empty(int(hidden_dim/1)), requires_grad=True)
            zeros(self.b3)

        elif layer==2:
            self.attn2=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(12,12), bias=True)
            self.softmax2=nn.Softmax(dim=0)
            self.linear2 = Linear(input_dim, int(hidden_dim), bias=False, weight_initializer='glorot')
            self.b2 = nn.Parameter(torch.empty(int(self.hidden_dim)), requires_grad=True)
            init.xavier_uniform_(self.attn2.weight)
            self.attn2.bias.data.zero_()
            zeros(self.b2)

        else:
            assert True, 'layer number is wrong'


    def forward(self, X:torch.Tensor, G:torch.Tensor, G_real=None, G_nor=None, G_add=None, hyper=None): 
        #x为 batch*12*12*维度 的边嵌入=本层的batch*12*12*1的真实OD矩阵+上层batch*12*12*32的输出；G为 12*47的点嵌入=构造的ODG矩阵 或 真实OD矩阵；G_real为真实的OD矩阵
        _,b,c,_=X.shape[0],X.shape[1],X.shape[2],X.shape[3]
        #print(X.shape,G.shape,G_real.shape,G_nor.shape)
        if self.layer==0:        

            x=X.squeeze()
            G_rea = G_real.squeeze()
            G_nor=G_nor.squeeze()
            hyper=hyper.squeeze()
            #print(G_nor.shape,hyper.shape)
            edgevalue=dense_to_sparse(G_rea)[0]
            if not contains_self_loops(edgevalue): print('ERROR!!!!')
            g = dgl.graph((edgevalue[0],edgevalue[1]),num_nodes = b)
            mode_1,_=self.gat(g, G_rea, x[edgevalue[0],edgevalue[1]])
            u, r ,v, z = torch.chunk(mode_1, self.head, dim=1)
            u=u.squeeze().view(b,c,-1)
            r=r.squeeze().view(b,c,-1)
            v=v.squeeze().view(b,c,-1)
            z=z.squeeze().view(b,c,-1)
            mode_2=torch.cat((u,r,v,z),dim=-1).unsqueeze(0)

            linearX=self.linear0(X)
            feat_set = list()
            feat_set.append(linearX)
            mode_2_prod_tmp = torch.einsum('bncl,nm->bmcl', linearX, G_nor[1])
            mode_2_prod = torch.einsum('bmcl,cd->bmdl', mode_2_prod_tmp, G_nor[1])
            feat_set.append(mode_2_prod)
            H = feat_set[0]+feat_set[1]+self.b0

            linearX2=self.linear4(X)
            feat_set2 = list()
            feat_set2.append(linearX2)
            mode_2_prod_tmp2 = torch.einsum('bncl,nm->bmcl', linearX2, hyper[1])
            mode_2_prod2 = torch.einsum('bmcl,cd->bmdl', mode_2_prod_tmp2, hyper[1])
            feat_set2.append(mode_2_prod2)
            H2 = feat_set2[0]+feat_set2[1]+self.b4

            return torch.cat((mode_2, H + H2), dim=-1)
            
        elif self.layer==1:       
            
            G_nor=G_nor.squeeze()
            hyper=hyper.squeeze()
            #print(G_nor.shape,hyper.shape)
            linearX=self.linear1(X)
            feat_set = list()
            feat_set.append(linearX)
            mode_2_prod_tmp = torch.einsum('bncl,nm->bmcl', linearX, G_nor[1])
            mode_2_prod = torch.einsum('bmcl,cd->bmdl', mode_2_prod_tmp, G_nor[1])
            feat_set.append(mode_2_prod)
            H = feat_set[0]+feat_set[1]+self.b1

            linearX2=self.linear3(X)
            feat_set2 = list()
            feat_set2.append(linearX2)
            mode_2_prod_tmp2 = torch.einsum('bncl,nm->bmcl', linearX2, hyper[1])
            mode_2_prod2 = torch.einsum('bmcl,cd->bmdl', mode_2_prod_tmp2, hyper[1])
            feat_set2.append(mode_2_prod2)
            H2 = feat_set2[0]+feat_set2[1]+self.b3

            return H+H2

        elif self.layer==2:      
            
            attG=torch.cat((G_add,G.unsqueeze(0)),0)
            _,btmp,_=torch.chunk(attG,3,dim=1)
            b_attn=self.softmax2(self.attn2(btmp))
            res_b=torch.sum(btmp*b_attn,0).squeeze()
            linearX=self.linear2(X)
            feat_set = list()
            feat_set.append(linearX)
            mode_2_prod_tmp = torch.einsum('bncl,nm->bmcl', linearX, res_b)
            mode_2_prod = torch.einsum('bmcl,cd->bmdl', mode_2_prod_tmp, res_b)
            feat_set.append(mode_2_prod)
            H = feat_set[0]+feat_set[1]+self.b2

            return H

        else:
            assert True, 'layer number is wrong'


class ODCRUcell(nn.Module):        # Origin-Destination Convolutional Recurrent Unit
    def __init__(self, num_nodes:int, K:int, input_dim:int, hidden_dim:int, layer:int, use_bias=True, activation=None):
        super(ODCRUcell, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.gates = ODconv(K, input_dim+hidden_dim, hidden_dim*2, layer, use_bias, activation)
        self.candi = ODconv(K, input_dim+hidden_dim, hidden_dim, layer, use_bias, activation)

    def init_hidden(self, batch_size:int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(batch_size, self.num_nodes, self.num_nodes, self.hidden_dim))
        return hidden

    def forward(self, G:torch.Tensor, Xt:torch.Tensor, Ht_1:torch.Tensor, G_real=None, G_nor=None, G_add=None, hyper=None):
        assert len(Xt.shape) == len(Ht_1.shape) == 4, 'ODCRU cell must take in 4D tensor as input [Xt, Ht-1]'

        XH = torch.cat([Xt, Ht_1], dim=-1)
        XH_conv = self.gates(X=XH, G=G, G_real=G_real, G_nor=G_nor, G_add=G_add, hyper=hyper)
        u, r = torch.split(XH_conv, int(self.hidden_dim), dim=-1)
    
        update = torch.sigmoid(u)
        reset = torch.sigmoid(r)

        candi = torch.cat([Xt, reset*Ht_1], dim=-1)
        candi_conv = torch.tanh(self.candi(X=candi, G=G, G_real=G_real, G_nor=G_nor, G_add=G_add, hyper=hyper))
        Ht = (1.0 - update) * Ht_1 + update * candi_conv
        return Ht


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, output_dim: int):
        super(TransformerEncoderLayer, self).__init__()
        #self.activation = torch.tanh
        self.self_attn = MultiheadAttention(d_model, nhead, output_dim)
        self.linear1 = Linear(d_model, output_dim, bias=True, weight_initializer='glorot')
        #self.linear2 = Linear(dim_feedforward, output_dim, bias=True, weight_initializer='glorot')

    def forward(self, src):
        x = src + self.self_attn(src, src, src)
        x = self.linear1(x)
        return x


class ODCRUencoder(nn.Module):
    def __init__(self, num_nodes:int, K:int, input_dim:int, hidden_dim:int, num_layers:int, use_bias=True, activation=None, return_all_layers=True):
        super(ODCRUencoder, self).__init__()
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.attention=TransformerEncoderLayer(d_model=self.hidden_dim[0], nhead=2, dim_feedforward=2*self.hidden_dim[0], output_dim=self.hidden_dim[0])

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i==0 else self.hidden_dim[i-1]
            if i==0:
                self.cell_list.append(ODCRUcell(num_nodes, K, cur_input_dim, self.hidden_dim[i],layer=0, use_bias=use_bias, activation=activation))
            else:
                self.cell_list.append(ODCRUcell(num_nodes, K, cur_input_dim, self.hidden_dim[i],layer=1, use_bias=use_bias, activation=activation))


    def forward(self, G:torch.Tensor, X_seq:torch.Tensor, H0_l=None, G_real=None, G_nor=None, G_add=None, hyper=None):

        assert len(X_seq.shape) == 5, 'ODCRU encoder must take in 5D tensor as input X_seq'
        
        batch_size, seq_len, node1, node2, _ = X_seq.shape
        if H0_l is None:
            H0_l = self._init_hidden(batch_size)

        out_seq_lst = list()    # layerwise output seq
        Ht_lst = list()        # layerwise last state
        in_seq_l = X_seq        # current input seq

        for l in range(self.num_layers):

            Ht = H0_l[l]
            out_seq_l = list()
            for t in range(seq_len):

                Ht = self.cell_list[l](G=G, Xt=in_seq_l[:,t,...], Ht_1=Ht,\
                         G_real=G_real[:,t,...], G_nor=G_nor[:,t,...], G_add=G_add, hyper=hyper[:,t,...])

                out_seq_l.append(Ht)
            out_seq_l = torch.stack(out_seq_l, dim=1)  # (B, T, N, C, h)
            out_seq_l = self.attention(out_seq_l)
            

            in_seq_l = out_seq_l    # update input seq
            out_seq_lst.append(out_seq_l)
            Ht_lst.append(Ht)

        return out_seq_lst, Ht_lst

    def _init_hidden(self, batch_size):
        H0_l = []
        for i in range(self.num_layers):
            H0_l.append(self.cell_list[i].init_hidden(batch_size))
        return H0_l

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ODCRUdecoder(nn.Module):
    def __init__(self, num_nodes:int, K:int, output_dim:int, hidden_dim:int, out_horizon:int, num_layers:int, use_bias=True, activation=None):
        super(ODCRUdecoder, self).__init__()
        self.out_horizon = out_horizon      # output steps
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = output_dim if i==0 else self.hidden_dim[i-1]
            self.cell_list.append(ODCRUcell(num_nodes, K, cur_input_dim, self.hidden_dim[i],layer=2, use_bias=use_bias, activation=activation))

    def forward(self, G, Xt, H0_l, G_real, G_nor, G_add):
        assert len(Xt.shape) == 4, 'ODCRU decoder must take in 4D tensor as input Xt'

        Ht_lst = list()        # layerwise hidden state
        Xin_l = Xt

        for l in range(self.num_layers):
            
            Ht_l = self.cell_list[l](G=G, Xt=Xin_l, Ht_1=H0_l[l], G_real=G_real, G_nor=G_nor, G_add=G_add)
            Ht_lst.append(Ht_l)
            Xin_l = Ht_l      # update input for next layer

        return Ht_l, Ht_lst

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ODCRN(nn.Module):
    def __init__(self, num_nodes:int, K:int, input_dim:int, hidden_dim:int, out_horizon:int, num_layers:int, DGCbool:bool=True, use_bias=True, activation=None):
        super(ODCRN, self).__init__()
        
        self.DGCbool = DGCbool

        self.encoder = ODCRUencoder(num_nodes, K, input_dim, hidden_dim, num_layers, use_bias, activation, return_all_layers=True)
        self.decoder = ODCRUdecoder(num_nodes, K, input_dim, hidden_dim, out_horizon, num_layers, use_bias, activation)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=use_bias)
    
    def forward(self, G:torch.Tensor, X_seq:torch.Tensor, G_real, G_nor, G_add, hyper): 

        assert len(X_seq.shape) == 5, 'ODCRN must take in 5D tensor as input X_seq: [4,7,12,12,1]'
        
        # encoding
        _, Ht_lst = self.encoder(G=G, X_seq=X_seq, H0_l=None, G_real=G_real, G_nor=G_nor, G_add=G_add, hyper=hyper)
        # initiate decoder input
        deco_input = torch.zeros(X_seq.shape[:1]+X_seq.shape[2:], device=X_seq.device)
        # decoding
        outputs = list()
        G_real=torch.sum(G_real,dim=1)
        for t in range(self.decoder.out_horizon):
            Ht_l, _ = self.decoder(G=G, Xt=deco_input, H0_l=Ht_lst, G_real=G_real, G_nor=G_nor, G_add=G_add,)
            output = self.linear(Ht_l)
            deco_input = output     # update decoder input
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)  # (B, horizon, N, C, h)

        return outputs

