import os
from pprint import pformat
from site import addpackage
import argparse
import numpy as np
from datetime import datetime
from torch import nn, optim
import torch
from torch.optim import lr_scheduler
import Utils
from Utils import AdjProcessor
from DGLODCRN import ODCRN
import random
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj,to_undirected
from torch_geometric.nn import knn_graph

seed=1029
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class ModelTrainer(object):
    def __init__(self, params:dict, data:dict, data_container):
        self.params = params
        self.data_container = data_container
        self.get_static_graph(graph=data['adj'])    # initialize static graphs and K values
        self.model = self.get_model().to(params['GPU'])
        self.criterion = self.get_loss()
        self.optimizer = self.get_optimizer()

    def get_static_graph(self, graph:np.array):
        #print(graph.shape)
        self.K = self.get_support_K(self.params['kernel_type'], self.params['cheby_order'])
        self.G = self.preprocess_adj(graph, self.params['kernel_type'], self.params['cheby_order'])
        '''graph = torch.tensor(graph)
        tmp=knn_graph(graph, k=7)
        tmp=to_undirected(tmp,num_nodes=12)
        atmp=AdjProcessor.gcnnorm(tmp,num_nodes=12)
        self.G_hyper = to_dense_adj(edge_index=atmp[0], edge_attr=atmp[1])[0].unsqueeze(0).unsqueeze(0).to(params['GPU'])'''
        return

    @staticmethod
    def get_support_K(kernel_type, cheby_order):
        if kernel_type == 'localpool':
            assert cheby_order == 1
            K = 1
        elif (kernel_type=='chebyshev')|(kernel_type=='random_walk_diffusion'):
            K = cheby_order + 1
        elif kernel_type == 'dual_random_walk_diffusion':
            K = cheby_order*2 + 1
        else:
            raise ValueError('Invalid kernel_type. Must be one of '
                             '[chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion].')
        return K


    def preprocess_adj(self, adj_mtx:np.array, kernel_type, cheby_order):
        adj_preprocessor = Utils.AdjProcessor(kernel_type, cheby_order)
        adj = torch.from_numpy(adj_mtx).float()
        adj = adj_preprocessor.gcnprocess(adj)
        return adj.to(self.params['GPU'])       # G: (support_K, N, N)

    def get_model(self):
        #if self.params['model'] == 'ODCRN':
            model = ODCRN(num_nodes=self.params['N'],
                                K=self.K,
                                input_dim=1,
                                hidden_dim=self.params['hidden_dim'],
                                out_horizon=self.params['pred_len'],
                                #activation=nn.Tanh,
                                num_layers=self.params['nn_layers'],
                                DGCbool=bool(self.params['use_DGC'])
                                )
        #else:
        #    raise NotImplementedError('Invalid model name.')
            return model

    def get_loss(self):
        if self.params['loss'] == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
        elif self.params['loss'] == 'MAE':
            criterion = nn.L1Loss(reduction='mean')
        elif self.params['loss'] == 'Huber':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_optimizer(self):
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(params=self.model.parameters(),
                                   lr=self.params['learn_rate'])
            scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[10*2+1,20*2+1,30*2+1,40*2+1,50*2+1,60*2+1,70*2+1],gamma = 0.9)
        else:
            raise NotImplementedError('Invalid optimizer name.')
        return optimizer,scheduler
    

    def train(self, data_loader:dict, modes:list, early_stop_patience):
        checkpoint = {'epoch': 0, 'state_dict': self.model.state_dict()}
        val_loss = np.inf
        patience_count = early_stop_patience

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training begins:')
        beginlist=[]
        endlist=[]
        for epoch in range(1, 1 + self.params['num_epochs']):
            starttime = datetime.now()
            running_loss = {mode: 0.0 for mode in modes}
            #训练时只对训练集数据进行了训练，验证集数据只作为输出和早停的依据。
            for mode in modes:
                if mode == 'train' or mode=='train0':
                    self.model.train()
                else:
                    self.model.eval()

                step = 0
                for x_seq, edge, x_nor, x_add, hyper, y_true in tqdm(data_loader[mode]):
                    with torch.set_grad_enabled(mode=(mode=='train' or mode=='train0')):
                        #print(pe.shape, x_seq.shape, edge.shape, x_nor.shape, x_add.shape, self.G.shape)
                        #hyper_add=torch.cat((hyper_add, self.G_hyper), dim=1)
                        y_pred = self.model( X_seq=x_seq, G=self.G, G_real=edge, G_nor=x_nor,\
                                G_add=x_add.squeeze(), hyper=hyper)
                        #loss = self.criterion(y_pred, y_true)
                        loss = 3*self.criterion(y_pred[:,0,...], y_true[:,0,...])+3*self.criterion(y_pred[:,1,...], y_true[:,1,...])+\
                            3*self.criterion(y_pred[:,2,...], y_true[:,2,...])+2*self.criterion(y_pred[:,3,...], y_true[:,3,...])+\
                                2*self.criterion(y_pred[:,4,...], y_true[:,4,...])+2*self.criterion(y_pred[:,5,...], y_true[:,5,...])+\
                                    1*self.criterion(y_pred[:,6,...], y_true[:,6,...])
                        if mode == 'train' or mode=='train0':
                            self.optimizer[0].zero_grad()
                            loss.backward()
                            self.optimizer[0].step()
                        
                    running_loss[mode] += loss * y_true.shape[0]
                    step += y_true.shape[0]
                    torch.cuda.empty_cache()
                #print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHere')
                self.optimizer[1].step()
                
                if mode == 'validate':
                    if epoch==1:
                        f = open(self.params['output_dir']  + '/' + self.params['model'] + '_process.txt', 'w')
                        f.write('begin\r\n')
                        f.close()
                    epoch_val_loss = running_loss[mode]/step
                    if epoch_val_loss <= val_loss:
                        print(f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_val_loss:.5}. '
                              f'Update model checkpoint..', f'used {(datetime.now() - starttime).seconds}s')
                        f = open(self.params['output_dir']  + '/' + self.params['model'] + '_process.txt', 'a')
                        f.write('Epoch '+str(epoch)+' validation loss drops from '+str(val_loss)+' to '+str(epoch_val_loss)+'\r\n')
                        f.close()
                        val_loss = epoch_val_loss
                        checkpoint.update(epoch=epoch, state_dict=self.model.state_dict())
                        torch.save(checkpoint, self.params['output_dir']+f'/{self.params["model"]}_od.pkl')
                        patience_count = early_stop_patience
                        #if epoch>30:
                        #    self.test(data_loader=data_loader, modes=['test'])
                    else:
                        print(f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.', f'used {(datetime.now() - starttime).seconds}s')
                        patience_count -= 1
                        if patience_count <= 0 and epoch>=80:
                            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                            print(f'    Early stopping at epoch {epoch}. {self.params["model"]} model training ends.')
                            return   
        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training ends.')
        torch.save(checkpoint, self.params['output_dir']+f'/{self.params["model"]}_od.pkl')
        return beginlist,endlist

    def test(self, data_loader:dict, modes:list):
        trained_checkpoint = torch.load(self.params['output_dir']+f'/{self.params["model"]}_od.pkl')
        self.model.load_state_dict(trained_checkpoint['state_dict'])
        self.model.eval()

        for mode in modes:
            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            print(f'     {self.params["model"]} model testing on {mode} data begins:')
            forecast, ground_truth = [], []
            #count=0
            for x_seq, edge, x_nor, x_add, hyper, y_true in tqdm(data_loader[mode]):
                #hyper_add=torch.cat((hyper_add, self.G_hyper), dim=1)
                y_pred = self.model( X_seq=x_seq, G=self.G, G_real=edge,G_nor=x_nor,\
                        G_add=x_add.squeeze(), hyper=hyper)
                forecast.append(y_pred.cpu().detach().numpy())
                ground_truth.append(y_true.cpu().detach().numpy())

            forecast = np.concatenate(forecast, axis=0)
            ground_truth = np.concatenate(ground_truth, axis=0)
            if mode == 'test':
                np.save(self.params['output_dir'] + '/' + self.params['model'] + '_prediction.npy', forecast)
                np.save(self.params['output_dir'] + '/' + self.params['model'] + '_groundtruth.npy', ground_truth)
            
            # evaluate on metrics
            MSE, RMSE, MAE, MAPE,R2 = self.evaluate(forecast, ground_truth)
            f = open(self.params['output_dir']  + '/' + self.params['model'] + '_prediction_scores.txt', 'a')
            f.write("%s, MSE, RMSE, MAE, MAPE, R2_score, %.10f, %.10f, %.10f, %.10f, %.10f\n" % (mode, MSE, RMSE, MAE, MAPE,R2))
            f.close()

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model testing ends.')
        return
    
    @staticmethod
    def evaluate(y_pred: np.array, y_true: np.array, precision=10):
        def MSE(y_pred: np.array, y_true: np.array):
            return np.mean(np.square(y_pred - y_true))
        def RMSE(y_pred:np.array, y_true:np.array):
            return np.sqrt(MSE(y_pred, y_true))
        def MAE(y_pred:np.array, y_true:np.array):
            return np.mean(np.abs(y_pred - y_true))
        def MAPE(y_pred:np.array, y_true:np.array, epsilon=1e-0):       # avoid zero division
            return np.mean(np.abs(y_pred - y_true) / (y_true + epsilon))
        def R2score(y_pred:np.array, y_true:np.array, epsilon=1e-0):
            return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

        print('MSE:', round(MSE(y_pred, y_true), precision))
        print('RMSE:', round(RMSE(y_pred, y_true), precision))
        print('MAE:', round(MAE(y_pred, y_true), precision))
        print('MAPE:', round(MAPE(y_pred, y_true)*100, precision), '%')
        print('R2_score',round(R2score(y_pred, y_true), precision))
        return MSE(y_pred, y_true), RMSE(y_pred, y_true), MAE(y_pred, y_true), MAPE(y_pred, y_true),R2score(y_pred, y_true)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run OD Prediction with ODCRN')

    # command line arguments
    parser.add_argument('-GPU', '--GPU', type=str, help='Specify GPU usage', default='cuda:1')
    parser.add_argument('-in', '--input_dir', type=str, default='../data')
    parser.add_argument('-out', '--output_dir', type=str, default='./final')
    parser.add_argument('-model', '--model', type=str, help='Specify model',default='EESTN-7to7-attn-3hyper-nodynamic-newloss-3332221-tmp')
    parser.add_argument('-DGC', '--use_DGC', type=int, default=0)       # 1: OCDRN w/ DGC 0: ODCRN w/o DGC
    parser.add_argument('-obs', '--obs_len', type=int, help='Length of observation sequence', default=7)
    parser.add_argument('-pred', '--pred_len', type=int, help='Length of prediction sequence', default=7)
    parser.add_argument('-split', '--split_ratio', type=float, nargs='+',
                        help='Relative data split ratio in train : validate : test'
                             ' Example: -split 5 1 2', default=[6.4, 1.6, 2])
    parser.add_argument('-batch', '--batch_size', type=int, default=1)
    parser.add_argument('-earlystop', '--early_stop', type=int, default=15)
    parser.add_argument('-hidden', '--hidden_dim', type=int, default=16)
    parser.add_argument('-kernel', '--kernel_type', type=str,
                        choices=['chebyshev', 'localpool', 'random_walk_diffusion', 'dual_random_walk_diffusion'],
                        default='random_walk_diffusion')    # GCN kernel type
    parser.add_argument('-K', '--cheby_order', type=int, default=2)      # GCN chebyshev order
    parser.add_argument('-nn', '--nn_layers', type=int, default=2)       # layers
    parser.add_argument('-epoch', '--num_epochs', type=int, default=200)
    parser.add_argument('-loss', '--loss', type=str, help='Specify loss function',
                        choices=['MSE', 'MAE', 'Huber'], default='MSE')
    parser.add_argument('-optim', '--optimizer', type=str, help='Specify optimizer', default='Adam')
    parser.add_argument('-lr', '--learn_rate', type=float, default=5e-4)
    parser.add_argument('-test', '--test_only', type=int, default=0)      # 1 for test only

    params = parser.parse_args().__dict__       # save in dict

    # paths
    os.makedirs(params['output_dir'], exist_ok=True)

    # 获得标准数据
    data_input = Utils.DataInput(data_dir=params['input_dir'])
    data = data_input.load_data()
    params['N'] = data['OD'].shape[1]

    # 根据标准数据划分出训练集和测试集
    data_generator = Utils.DataGenerator(obs_len=params['obs_len'], 
                                         pred_len=params['pred_len'], 
                                         data_split_ratio=params['split_ratio'],
                                         kernel=params['kernel_type'],K=params['cheby_order'])
    data_loader = data_generator.get_data_loader(data=data, params=params)

    # get model
    trainer = ModelTrainer(params=params, data=data, data_container=data_input)

    if bool(params['test_only']) == False:
        trainer.train(data_loader=data_loader,
                      early_stop_patience=params['early_stop'],
                      modes=['train', 'validate'])
    trainer.test(data_loader=data_loader,
                 modes=['test'])

