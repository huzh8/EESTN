from operator import mod
from re import A
import re
from site import addpackage
import scipy.sparse as ss
import numpy as np
import pandas as pd
import torch
import math
from torch_scatter import scatter_add
from torch.utils.data import Dataset, DataLoader
#from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import remove_self_loops,is_undirected,get_laplacian,\
    add_self_loops,contains_self_loops,dense_to_sparse,to_dense_adj,to_undirected,add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from tqdm import trange
from torch_geometric.nn import knn_graph

class DataInput(object):
    def __init__(self, data_dir:str):
        self.data_dir = data_dir

    def load_data(self):
        ODPATH = '../data/od_matrix.npy'
        ADJPATH = '../data/od_adj.npy'
        data = np.load(ODPATH)
        data = data[-576:,:,:,np.newaxis]
        ODdata = np.log(data + 1.0)      # log transformation

        adj = np.load(ADJPATH)
        dataset = dict()
        dataset['OD'] = ODdata
        dataset['adj'] = adj
        dataset['edge'] = ODdata 
        return dataset

def PositionalEncoding(d_model, max_len = 1000):

    position = torch.arange(1,max_len+1).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class DataGenerator(object):
    def __init__(self, obs_len:int, pred_len, data_split_ratio:tuple, kernel, K):
        self.kernel=kernel
        self.K=K
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def split2len(self, data_len:int):
        mode_len = dict()
        mode_len['train'] = int(self.data_split_ratio[0]/sum(self.data_split_ratio) * data_len)
        mode_len['validate'] = int(self.data_split_ratio[1]/sum(self.data_split_ratio) * data_len)
        mode_len['test'] = data_len - mode_len['train'] - mode_len['validate']
        return mode_len

    def get_data_loader(self, data:dict, params:dict):
        x_seq, real, y_seq ,x_nor, x_add, pe, hyper= self.get_feats(data['OD'])

        feat_dict = dict()
        feat_dict['x_seq'] = torch.from_numpy(np.asarray(x_seq)).float().to(params['GPU'])
        feat_dict['x_edge'] = torch.from_numpy(np.asarray(real)).float().to(params['GPU'])
        #这里加一项表示由X得到的正则化矩阵
        feat_dict['x_nor'] = torch.from_numpy(np.asarray(x_nor)).float().to(params['GPU'])
        feat_dict['x_add'] = torch.from_numpy(np.asarray(x_add)).float().to(params['GPU'])
        feat_dict['pe'] = torch.from_numpy(np.asarray(pe)).float().to(params['GPU'])
        feat_dict['hyper'] = torch.from_numpy(np.asarray(hyper)).float().to(params['GPU'])

        y_seq = torch.from_numpy(np.asarray(y_seq)).float().to(params['GPU'])
        mode_len = self.split2len(data_len=y_seq.shape[0])
        data_loader = dict()      
        for mode in ['train0','train', 'validate', 'test']:
            if mode=='train0':
                dataset = ODDataset(inputs=feat_dict, output=y_seq, mode='train', mode_len=mode_len)
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False)
            else:
                dataset = ODDataset(inputs=feat_dict, output=y_seq, mode=mode, mode_len=mode_len)
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=True) 
        
        return data_loader

    def get_feats(self, data:np.array):
        feat=list()
        for i in data:
            adj_preprocessor = AdjProcessor(self.kernel, self.K)
            adj = torch.from_numpy(i).float().squeeze()
            adj = adj_preprocessor.gcnprocess(adj,False)
            feat.append(adj.unsqueeze(0))
        adjdata=torch.cat(feat,dim=0).numpy()

        rel=list()
        for i in data:
            rel.append(i+1*np.eye(12)[:,:,np.newaxis])
        rel=np.array(rel)

        feat1=list()
        for i in data:
            adj_preprocessor = AdjProcessor(self.kernel, self.K)
            adj = torch.from_numpy(i).float().squeeze()
            adj = adj_preprocessor.gcnprocess(adj,False)
            feat1.append(adj.unsqueeze(0))
        adjdata1=torch.cat(feat1,dim=0).numpy()

        pe=torch.from_numpy(data)     
        pe = pe.numpy()

        feat2=list()
        for i in data:
            adj_preprocessor = AdjProcessor(self.kernel, self.K)
            adj = torch.from_numpy(i).float().squeeze()
            adj = torch.mm(adj,torch.mm(adj, adj))
            adj = adj_preprocessor.gcnprocess(adj,False)
            feat2.append(adj.unsqueeze(0))
        hypergraph=torch.cat(feat2,dim=0).numpy()


        x, real, y, z, u, posembe, hyper= [], [], [], [], [], [], []
        for i in range(self.obs_len, data.shape[0]-self.pred_len):
            x.append(data[i-self.obs_len : i])
            real.append(rel[i-self.obs_len : i])
            y.append(data[i : i+self.pred_len])
            z.append(adjdata[i-self.obs_len : i])
            u.append(adjdata1[i-self.obs_len : i])
            posembe.append(pe[i-self.obs_len : i])
            hyper.append(hypergraph[i-self.obs_len : i])
        return x, real, y, z, u, posembe, hyper


class ODDataset(Dataset):
    def __init__(self, inputs:dict, output:torch.Tensor, mode:str, mode_len:dict):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_seq'][item], self.inputs['x_edge'][item], self.inputs['x_nor'][item], self.inputs['x_add'][item],\
                 self.inputs['hyper'][item],self.output[item]

    def prepare_xy(self, inputs:dict, output:torch.Tensor):
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:       # test
            start_idx = self.mode_len['train']+self.mode_len['validate']

        x = dict()
        x['x_seq'] = inputs['x_seq'][start_idx : (start_idx + self.mode_len[self.mode])]
        x['x_edge'] = inputs['x_edge'][start_idx : (start_idx + self.mode_len[self.mode])]
        x['x_nor'] = inputs['x_nor'][start_idx : (start_idx + self.mode_len[self.mode])]
        x['x_add'] = inputs['x_add'][start_idx : (start_idx + self.mode_len[self.mode])]
        #x['pe'] = inputs['pe'][start_idx : (start_idx + self.mode_len[self.mode])]
        x['hyper'] = inputs['hyper'][start_idx : (start_idx + self.mode_len[self.mode])]
        #x['hyper_add'] = inputs['hyper_add'][start_idx : (start_idx + self.mode_len[self.mode])]
                   
        y = output[start_idx : start_idx + self.mode_len[self.mode]]
        return x, y


class AdjProcessor():
    def __init__(self, kernel_type:str, K:int):
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != 'localpool' else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def norm(self, edge_index, num_nodes, edge_weight, normalization, dtype, lambda_max=None,batch = None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        #print(edge_index.shape,edge_weight.shape)
        assert edge_weight is not None

        return edge_index, edge_weight
    @staticmethod
    def gcnnorm( edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
        fill_value = 2. if improved else 1.

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    @staticmethod
    def hygcnnorm( edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
        fill_value = 2. if improved else 1.

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight* edge_weight * deg_inv_sqrt[col]

    def gcnprocess(self, adj:torch.Tensor,direct=True):
        kernel_list = list()
        if direct==True:
            tmp=dense_to_sparse(adj)
            atmp=AdjProcessor.gcnnorm(tmp[0],tmp[1],len(adj))
            #atmp=self.gcnnorm(tmp[0], num_nodes = len(adj))
            P_forward = to_dense_adj(edge_index = atmp[0], edge_attr = atmp[1])[0]
            kernel_list = self.compute_chebyshev_polynomials(P_forward, kernel_list)
        else:
            tmp=dense_to_sparse(adj)
            #这里加一个到无向图的转换
            tmp=to_undirected(tmp[0],tmp[1])
            atmp=AdjProcessor.gcnnorm(tmp[0],tmp[1],len(adj))
            #atmp=self.gcnnorm(tmp[0], num_nodes = len(adj))
            P_forward = to_dense_adj(edge_index = atmp[0], edge_attr = atmp[1])[0]
            kernel_list = self.compute_chebyshev_polynomials(P_forward, kernel_list)
        kernels = torch.stack(kernel_list, dim=0)
        return kernels

    def hygcnprocess(self, adj:torch.Tensor,direct=True):
        kernel_list = list()
        if direct==True:
            tmp=dense_to_sparse(adj)
            atmp=AdjProcessor.hygcnnorm(tmp[0],tmp[1],len(adj))
            P_forward = to_dense_adj(edge_index = atmp[0], edge_attr = atmp[1])[0]
            kernel_list = self.compute_chebyshev_polynomials(P_forward, kernel_list)
        else:
            tmp=dense_to_sparse(adj)
            #这里加一个到无向图的转换
            tmp=to_undirected(tmp[0],tmp[1])
            atmp=AdjProcessor.hygcnnorm(tmp[0],tmp[1],len(adj))
            P_forward = to_dense_adj(edge_index = atmp[0], edge_attr = atmp[1])[0]
            kernel_list = self.compute_chebyshev_polynomials(P_forward, kernel_list)
        kernels = torch.stack(kernel_list, dim=0)
        return kernels

    def compute_chebyshev_polynomials(self, x, T_k):
        # compute Chebyshev polynomials up to order k. Return a list of matrices.
        # print(f"Computing Chebyshev polynomials up to order {self.K}.")
        for k in range(self.K + 1):
            if k == 0:
                #print(x.device)
                if x.device == "cpu":
                    T_k.append(torch.eye(x.shape[0]))
                else:
                    T_k.append(torch.eye(x.shape[0]).to(x.device))
                #T_k.append(x)
            elif k == 1:
                T_k.append(x)  #不包括自旋，其余两个包括
            else:
                #T_k.append(x)
                T_k.append(2 * torch.mm(x, T_k[k-1]) - T_k[k-2])
        #print(len(T_k))
        #for i in T_k:
        #    tmp,_=dense_to_sparse(i)
        #    print(is_undirected(tmp),contains_self_loops(tmp))
        #input()
        return T_k

    