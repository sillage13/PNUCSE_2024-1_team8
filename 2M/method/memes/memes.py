#!/usr/bin/env python
# coding: utf-8

# In[2]:

# example excute 
# python3 memes.py --run 1 --rec "greedy_8000_6_2000_shuffled_3" --cuda cpu --feature fingerprint --features_path ./data/fingerprints.dat --iters 6 --capital 30000 --initial 8000 --periter 2000 --n_cluster 4000 --save_af True --result_tail 3 --acquisition_func greedy --shuffle True 

import numpy as np
#from joblib import Parallel, delayed
from scipy.stats import norm



import sys
import argparse
import os

import math
import torch
from torch.utils.data import DataLoader, TensorDataset
import gpytorch
#from matplotlib import pyplot as plt
import numpy as np
import pickle
import time
import tqdm
#import ipdb
from collections import defaultdict
import joblib
import random

# from vina import Vina


def load_data(filename):
    # with open(filename, 'rb') as f:
    #     return pickle.load(f)
    return joblib.load(filename, mmap_mode='r')
    
def save_data(target, filename):
    # with open(filename, 'wb') as f:
    #     pickle.dump(target, f)
    joblib.dump(target, filename)

check_dup = defaultdict(bool)
result_list = []
docking_count = 1
access_cnt = 0
all_scores = {}
smile_index_dict = load_data("./../../data/smile_index_dict.dat")

def scoring_function(smile,index):
    global docking_dict
    global docking_count
    global smile_index_dict
    print(docking_count)
    print(f"{smile_index_dict[smile]}, {docking_dict[smile]}")
    print()
    return -1 * docking_dict[smile]

    # ## add your scoring function here
    # ## if want to search for molecules while optimize negative value multiply by -1 before returning
    # ligand = ligands[index]
    # global docking_count
    # global access_cnt
    # global receptor
    # access_cnt += 1
    # print(ligand, docking_count, access_cnt)
    # global docking_dict2
    # score = all_scores.get(index, 100)
    # if ligand in docking_dict2:
    #     print("HIT")
    #     print(docking_dict2[ligand])
    #     print()
    #     return -1 * docking_dict2[ligand]

    # v = Vina(sf_name='vina')
    # v.set_receptor(f'./../data/receptor/{receptor}.pdbqt')
    # try:
    #     v.set_ligand_from_file(f"./../data/ligand/{ligand}")
    # except TypeError as e:
    #     print(f"in {ligand}")
    #     print(e)

    #     all_scores[index] = 0
    #     return 0

    # x_coords = []
    # y_coords = []
    # z_coords = []

    # with open(f"./../data/ligand/{ligand}", "r") as f:
    #     for line in f:
    #         if line.startswith("ATOM") or line.startswith("HETATM"):
    #             x = float(line[30:38])
    #             y = float(line[38:46])
    #             z = float(line[46:54])
    #             x_coords.append(x)
    #             y_coords.append(y)
    #             z_coords.append(z)

    # center_x = sum(x_coords) / len(x_coords)
    # center_y = sum(y_coords) / len(y_coords)
    # center_z = sum(z_coords) / len(z_coords)

    # v.compute_vina_maps(center=[center_x, center_y, center_z], box_size=[120, 120, 120])

    # v.dock(exhaustiveness=32, n_poses=10)
    # print()
    # score = v.score()[0]

    # docking_dict2[ligand] = score
    # # pickle.dump(docking_dict, open("./data/docking_result.dat", "wb"))

    # all_scores[index] = score
    # ## optimizing for -ve value
    # return -1*score
    # ## optimizing for +ve value
    # #return +1*score


parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--rec', required=True)
parser.add_argument('--cuda',required=True)
parser.add_argument('--feature', default='mol2vec')
parser.add_argument('--features_path', default='data/features.pkl')
parser.add_argument('--smiles_path', default='data/all.txt')
parser.add_argument('--iters',default='40')
parser.add_argument('--capital',default='15000')
parser.add_argument('--initial',default='5000')
parser.add_argument('--periter',default='500')
parser.add_argument('--n_cluster',default='20')
parser.add_argument('--acquisition_func', default="ei")
parser.add_argument('--save_af', required=False,default="False")
parser.add_argument('--eps', default='0.05')
parser.add_argument('--beta', default=1)
# parser.add_argument("--ligand_path", default="./data/ligands.txt")
# parser.add_argument("--rec_path", default="receptor.pdbqt")
parser.add_argument("--result_tail", default="1")
parser.add_argument("--total_count", default=40)
args = parser.parse_args()
run = int(args.run)
iters = int(args.iters)
capital = int(args.capital)
initial = int(args.initial)
periter = int(args.periter)
n_cluster = int(args.n_cluster)
af = args.acquisition_func
eps = float(args.eps)
beta = float(args.beta)
rec = args.rec
feat = args.feature
features_path = args.features_path
result_tail = args.result_tail
total_count = int(args.total_count)
device = args.cuda
# rec_path = args.rec_path
# receptor = rec_path.split('/')[-1].split('.')[0]

# docking_dict = load_data(f"data/{receptor}_docking_result.dat")
docking_dict = load_data(f"./../../data/smile_score_dict.dat")

if device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device('cuda:{}'.format(device))
save_af = args.save_af == "True"
print("Using device: ", device)

directory_path = 'gp_runs/{}-'.format(feat)+rec+'/'+af+'/run'+str(run)

try:
    os.makedirs(directory_path)
except:
    pass

with open(directory_path+'/config.txt','w') as f:
    f.write("periter:	"+str(periter)+"\n")
    f.write("initial:	"+str(initial)+"\n")
    f.write("feature:	"+str(feat)+"\n")
    f.write("capital:   "+str(capital)+"\n")
    f.write("eps:	"+str(eps)+"\n")
    f.write("beta:  "+str(beta)+"\n")
    f.write("rec:	"+str(rec)+"\n")
    f.close()

pickle_obj = joblib.load(features_path, mmap_mode='r')
features_ = np.array(pickle_obj)

features_ = np.nan_to_num(features_)

features = np.memmap("data/features.npy", mode='w+', dtype='int8', shape=features_.shape)
features[:] = features_[:]
del pickle_obj
del features_

features[:] = features - features.min()
#features[:] = features/features.max()
features[:] = 2 * features - 1
print(features.shape)


## loading cluster labels
labels = np.loadtxt(f"data/labels{n_cluster}.txt")


##
# with open(args.ligand_path,'r') as f:
#     ligands = f.readlines()
#     ligands = [i.strip('\n') for i in ligands]

# print(ligands)


## selecting inital points
X_index = []
for i in range(n_cluster):
    X_index.extend(np.random.choice(np.where(labels==i)[0],int(initial//n_cluster)))
X_index = np.array(X_index)

## loading all smiles from complete dataset
smiles_set = set()
with open(args.smiles_path,'r') as f:
    smiles = f.readlines()
    smiles = [i.strip('\n') for i in smiles]


##
# docking_dict = pickle.load(open(f"./../data/{receptor}_docking_result.dat", "rb"))
# for ligand, score in docking_dict.items():
#         all_scores[smiles.index(ligand)] = score
all_scores = np.array(list(docking_dict.values()))

## making inital dataset 
X_init = features[X_index]
Y_init = []

for index in X_index:
    check_dup[smiles[index]] = True
    score = scoring_function(smiles[index], index)
    # result_list.append((-score, smiles[index]))
    result_list.append((-score, smile_index_dict[smiles[index]]))
    docking_count += 1
    Y_init.append(score)

tmp_sum = 0
with open(directory_path+'/start_smiles.txt','w') as f:
    for index in X_index:
        f.write(str(smile_index_dict[smiles[index]]) + ',' + str(all_scores[index]) + '\n')
        tmp_sum += all_scores[index]
        smiles_set.add(smiles[index])
    f.write('\naverage: ' + str(tmp_sum / len(X_index)))
    f.close()

print("PHASE 2")

# Initialize samples
# X_init = np.array(X_init)
# Y_init = np.array(Y_init)
# X_sample = X_init.reshape((initial//periter, periter, X_init.shape[1]))
# Y_sample = Y_init.reshape((initial//periter, periter))
X_sample = X_init
Y_sample = Y_init

# Number of iterations
n_iter = iters

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GP:
    def __init__(self, train_x, train_y, af, eps=0.05, beta=1):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.dataset = TensorDataset(train_x, train_y)
        self.loader = DataLoader(self.dataset, batch_size=1000)
        # self.train_x = train_x
        # self.train_y = train_y
        self.train_x, self.train_y = next(iter(self.loader))
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood)
        self.eps = eps
        self.beta = beta
        
        if af == "pi":
            self.compute_af = self.compute_pi
        elif af == "ucb":
            self.compute_af = self.compute_ucb
        elif af == "random":
            self.compute_af = self.compute_random
        elif af == "greedy":
            self.compute_af = self.compute_greedy
        elif af == "ebaf":
            self.compute_af = self.compute_ebaf
        elif af == "haf":
            self.compute_af = self.compute_haf
        else:
            self.compute_af = self.compute_ei

    def train_gp(self,train_x, train_y):
        # train_x = train_x.to(device)    
        # train_y = train_y.to(device)
        model = self.model.to(device)
        likelihood = self.likelihood.to(device)
        
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.01)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        training_iter = 300
        pbar = tqdm.tqdm(range(training_iter))
        prev_best_loss = 1e5
        early_stopping = 0
        for i in pbar:
            epoch_loss = 0
            for batch_x, batch_y in self.loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                self.model.set_train_data(batch_x, batch_y)
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(batch_x)
                # Calc loss and backprop gradients
                loss = -mll(output, batch_y)
                loss.backward()

                epoch_loss += loss.item()
            
                optimizer.step()

            epoch_loss = epoch_loss/len(self.loader)

            pbar.set_description('Iter %d/%d - Loss: %.3f ' % (
                i + 1, training_iter, epoch_loss,
            ))
            if epoch_loss < prev_best_loss:
                prev_best_loss = epoch_loss
                early_stopping = 0
            else:
                early_stopping += 1

            if early_stopping >= 10:
                break

    def compute_ei(self,id,best_val):
        self.model.eval()
        self.likelihood.eval()
        means = np.array([])
        stds = np.array([])
        #20000 is system dependent. Change according to space in GPU
        eval_bs_size = 1000
        for i in tqdm.tqdm(range(0,len(features),eval_bs_size)):
            test_x = features[i:i+eval_bs_size]
            test_x = torch.FloatTensor(test_x).to(device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(test_x))
                m = observed_pred.mean
                s = observed_pred.stddev
            m = m.cpu().numpy()
            s = s.cpu().numpy()
            means = np.append(means,m)
            stds = np.append(stds,s)

        imp = means - best_val - self.eps
        Z = imp/stds
        eis = imp * norm.cdf(Z) + stds * norm.pdf(Z)
        eis[stds == 0.0] = 0.0
        if save_af:
            np.savetxt(directory_path+'/' + af +'s_' + str(id)+'.out',eis)
        return eis
    
    def compute_pi(self, id, best_val):
        self.model.eval()
        self.likelihood.eval()
        means = np.array([])
        stds = np.array([])
        #20000 is system dependent. Change according to space in GPU
        eval_bs_size = 1000
        for i in tqdm.tqdm(range(0,len(features),eval_bs_size)):
            test_x = features[i:i+eval_bs_size]
            test_x = torch.FloatTensor(test_x).to(device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(test_x))
                m = observed_pred.mean
                s = observed_pred.stddev
            m = m.cpu().numpy()
            s = s.cpu().numpy()
            means = np.append(means,m)
            stds = np.append(stds,s)
        
        z = (best_val - means)/stds
        pis = norm.cdf(-z)
        if save_af:
            np.savetxt(directory_path+'/' + af +'s_' + str(id)+'.out', pis)
        return pis
    
    def compute_ucb(self, id, _):
        self.model.eval()
        self.likelihood.eval()
        means = np.array([])
        stds = np.array([])
        #20000 is system dependent. Change according to space in GPU
        eval_bs_size = 1000
        for i in tqdm.tqdm(range(0,len(features),eval_bs_size)):
            test_x = features[i:i+eval_bs_size]
            test_x = torch.FloatTensor(test_x).to(device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(test_x))
                m = observed_pred.mean
                s = observed_pred.stddev
            m = m.cpu().numpy()
            s = s.cpu().numpy()
            means = np.append(means,m)
            stds = np.append(stds,s)

        ucb = means + beta * stds

        if save_af:
            np.savetxt(directory_path+'/' + af +'s_' + str(id)+'.out', ucb)
        return ucb
    
    def compute_random(self, id, _):
        random = np.random.rand(len(features))

        if save_af:
            np.savetxt(directory_path+'/' + af +'s_' + str(id)+'.out', random)
    
    def compute_greedy(self, id, _):
        self.model.eval()
        self.likelihood.eval()
        means = np.array([])
        stds = np.array([])
        #20000 is system dependent. Change according to space in GPU
        eval_bs_size = 1000
        for i in tqdm.tqdm(range(0,len(features),eval_bs_size)):
            test_x = features[i:i+eval_bs_size]
            test_x = torch.FloatTensor(test_x).to(device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(test_x))
                m = observed_pred.mean
                s = observed_pred.stddev
            m = m.cpu().numpy()
            s = s.cpu().numpy()
            means = np.append(means,m)
            stds = np.append(stds,s)

        greedy = means

        if save_af:
            np.savetxt(directory_path+'/' + af +'s_' + str(id)+'.out', greedy)
        return greedy 


    def compute_haf(self, id, best_val, lambda_1=0.5, lambda_2=0.5):
        self.model.eval()
        self.likelihood.eval()
        means = np.array([])
        stds = np.array([])
        eval_bs_size = 1000
        for i in tqdm.tqdm(range(0, len(features), eval_bs_size)):
            test_x = features[i:i+eval_bs_size]
            test_x = torch.FloatTensor(test_x).to(device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(test_x))
                m = observed_pred.mean
                s = observed_pred.stddev
            m = m.cpu().numpy()
            s = s.cpu().numpy()
            means = np.append(means, m)
            stds = np.append(stds, s)

        # Expected Improvement (EI)
        imp = means - best_val - self.eps
        Z = imp / stds
        ei = imp * norm.cdf(Z) + stds * norm.pdf(Z)
        ei[stds == 0.0] = 0.0
        
        # Upper Confidence Bound (UCB)
        ucb = means + beta * stds

        # Hybrid Acquisition Function (HAF)
        haf = lambda_1 * ei + lambda_2 * ucb

        if save_af:
            np.savetxt(directory_path + '/' + af + 's_' + str(id) + '.out', haf)
            
        return haf


    def compute_ebaf(self, id, _):
        self.model.eval()
        self.likelihood.eval()
        means = np.array([])
        stds = np.array([])
        eval_bs_size = 1000
        for i in tqdm.tqdm(range(0, len(features), eval_bs_size)):
            test_x = features[i:i+eval_bs_size]
            test_x = torch.FloatTensor(test_x).to(device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(test_x))
                m = observed_pred.mean
                s = observed_pred.stddev
            m = m.cpu().numpy()
            s = s.cpu().numpy()
            means = np.append(means, m)
            stds = np.append(stds, s)

        # Entropy-Based Acquisition Function (EBAF)
        entropy = -0.5 * np.log(2 * np.pi * np.e * stds**2)

        if save_af:
            np.savetxt(directory_path + '/' + af + 's_' + str(id) + '.out', entropy)

        return entropy





## Iterative algorithm
for i in range(iters):
    print("Fit Start")
    sys.stdout.flush()
    # initialize likelihood and model
    start_time = time.time()

    ## property 1
    print("Running")
    train_x, train_y = torch.FloatTensor(X_sample), torch.FloatTensor(Y_sample)
    gp = GP(train_x, train_y, af=af, eps=eps, beta=beta)
    gp.train_gp(train_x, train_y)
    print("Fit Done in :",time.time() - start_time)
    sys.stdout.flush()

    print("Calculatin " + af.upper())
    sys.stdout.flush()
    start_time = time.time()
    eis = gp.compute_af(i,max(Y_sample))

    next_indexes = eis.argsort()
    X_next = []
    Y_next = []
    count = 0
    indices = []
    # if len(X_sample) < 1000:
    #     periter = 200
    # else:
    #     periter = int(args.periter)
    for index in next_indexes[::-1]:
        if smiles[index] in smiles_set:
            continue
        else:
            count+=1
            indices.append(index)
            X_next.append(features[index])
            score = scoring_function(smiles[index],index)
            if smiles[index] not in check_dup:
                # result_list.append((-score, smiles[index]))
                result_list.append((-score, smile_index_dict[smiles[index]]))
                docking_count += 1
            check_dup[smiles[index]] = True

            Y_next.append(score)
        if count == periter:
            break
    if(len(X_next)==0):
        print("break")
        break
    X_next = np.vstack(X_next)
    
    with open(directory_path+'/iter_'+str(i)+'.txt','w') as f:
        for index in indices:
            if smiles[index] not in smiles_set:
                f.write(smiles[index] + ',' + str(all_scores[index]) + '\n')
    for index in indices:
        smiles_set.add(smiles[index])

    print("Iter "+ str(i) + " done")
    sys.stdout.flush()

    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.append(Y_sample, np.array(Y_next))

    if(len(Y_sample)>=capital):
        print("capital reached")
        break

    print(Y_sample.shape)
    sys.stdout.flush()

print("PHASE 3")
# In[ ]:
train_x, train_y = torch.FloatTensor(X_sample), torch.FloatTensor(Y_sample)
gp = GP(train_x, train_y, af=af, eps=eps, beta=beta)
gp.train_gp(train_x, train_y)

torch.save(gp.model, directory_path+"/model.pt")
print(gp.model)

predicts = np.array([])
gp.model.eval()
gp.likelihood.eval()
for i in tqdm.tqdm(range(0,len(features),1000)):
    pred_features = features[i:i+1000]
    pred_features = torch.FloatTensor(pred_features).to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = gp.likelihood(gp.model(pred_features))
        predicts = np.append(predicts, pred.mean.numpy())

high_predict_idx = np.array(predicts).argsort()[::-1]

average_predict = 0
average_score = 0
average_predict_top10 = 0
average_top10 = 0
predict_cnt = 0
# with open(directory_path+"/predict.txt", "w") as f:
for index in high_predict_idx:
    score = scoring_function(smiles[index], index)
    if smiles[index] not in check_dup:
        docking_count += 1
        # result_list.append((-score, smiles[index]))
        result_list.append((-score, smile_index_dict[smiles[index]]))
    check_dup[smiles[index]] = True
    average_predict += predicts[index]
    average_score += score
    if predict_cnt < 10:
        average_predict_top10 += predicts[index]
        average_top10 += score
    predict_cnt += 1
    # f.write(f"{smiles[index]}, {predicts[index]}, {score}\n")

    if docking_count >= total_count:
        break
    # f.write("\n")
    # f.write(f"average top10 predict score = {average_predict_top10/10}\n")
    # f.write(f"average top10 score = {average_top10/10}\n")
    
    # f.write("\n")
    # f.write(f"average predict score = {average_predict/predict_cnt}\n")
    # f.write(f"average score = {average_score/predict_cnt}")

result_list.sort(key=lambda x: (x[0], x[1]))
top_ligands = result_list[:10]

tmp_sum = 0
with open(directory_path+'/predict.txt','w') as f:
    for ligand in top_ligands:
        f.write(str(ligand[1]) + ',' + str(ligand[0]) + '\n')
        tmp_sum += ligand[0]
    f.write('\naverage: ' + str(tmp_sum / 10))
    f.write('\ndocking count: ' + str(docking_count))
    f.close()


print("Top10 ligands")
for ligand in top_ligands:
    print(f"{ligand[1]}, Score: {ligand[0]}")
print()
scores = [ligand[0] for ligand in top_ligands]
avg_score = sum(scores) / len(scores)
print(f"Average score of top10 ligands: {avg_score: .2f}")

# save_data(docking_dict2, f"./../data/{receptor}_docking_result.dat")
# save_data(result_list, f"./../result/{receptor}/{total_count}/memes_{af}_result{result_tail}.dat")
save_data(result_list, f"./../../result/4UNN/memes/{initial}_{iters}_{periter}/memes_{af}_fingerprint_result{result_tail}.dat")
