"""Torch modules for graph attention networks with fully valuable edges (EGAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import imp
import torch as th
from torch import nn
from torch.nn import init
import math
from typing import Any
from torch import Tensor
from dgl.nn.functional import edge_softmax
from torch_geometric.nn.dense.linear import Linear
#from torch_geometric.nn.dense.linear import Linear
from dgl.base import DGLError
from dgl import function as fn

def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)
def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)
def zeros(value: Any):
    constant(value, 0.)

class EGATConv(nn.Module):

    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                 bias=True):

        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_ni = Linear(in_node_feats, out_edge_feats*num_heads, bias=False, weight_initializer='glorot')
        self.fc_nj = Linear(in_node_feats, out_edge_feats*num_heads, bias=False, weight_initializer='glorot')
        self.fc_fij = Linear(in_edge_feats, out_edge_feats*num_heads, bias=False, weight_initializer='glorot')
        self.fc_node = Linear(in_node_feats, out_node_feats*num_heads, bias=True, weight_initializer='glorot')
        
        self.attn = nn.Parameter(th.Tensor(size=(1, num_heads, out_edge_feats)))
        #self.attn = Linear(in_edge_feats, out_edge_feats*num_heads, bias=False, weight_initializer='glorot')
        
        self.bias_attn = nn.Parameter(th.Tensor(size=(num_heads , out_edge_feats,)))
        #self.bias = nn.Parameter(th.Tensor(size=(num_heads , out_node_feats,)))

        self.reset_parameters()

    def reset_parameters(self):
        self.fc_ni.reset_parameters()
        self.fc_nj.reset_parameters()
        self.fc_fij.reset_parameters()
        self.fc_node.reset_parameters()
        glorot(self.attn)
        zeros(self.bias_attn)


    def forward(self, graph, nfeats, efeats, get_attention=False):
        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue.')

            f_ni = self.fc_ni(nfeats).view(-1,self._num_heads,self._out_edge_feats)
            f_nj = self.fc_nj(nfeats).view(-1,self._num_heads,self._out_edge_feats)
            f_fij = self.fc_fij(efeats).view(-1,self._num_heads,self._out_edge_feats)
            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            f_out = graph.edata.pop('f_tmp') + f_fij
            f_out = f_out + self.bias_attn
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)
            e = nn.functional.leaky_relu(e,negative_slope=0.2)

            graph.edata['a'] = edge_softmax(graph, e)
            graph.ndata['h_out'] = self.fc_node(nfeats).view(-1, self._num_heads,
                                                             self._out_node_feats)
            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                             fn.sum('m', 'res'))
            h_out = graph.ndata['res']
            #print(h_out)
            #h_out = h_out + self.bias
            #input()
            if get_attention:
                return h_out, f_out, graph.edata.pop('a')
            else:
                return h_out, f_out