## zhaoshuzhi
import os
import numpy as np
import scipy.io as scio
from scipy.signal import hilbert
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import *
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from random import sample
import pandas as pd

def subsample(data,labels,percent):
    sample_list = []
    sample_num = int(percent*len(data))
    sample_list = [i for i in range(len(data))]
    sample_list = sample(sample_list,sample_num)
    data = [data[i] for i in sample_list]
    labels = [labels[i] for i in sample_list]
    
    return data,labels

##load ERP data from array
class ERP_matrix_datasets(Dataset):
    ##build a new class for own dataset
    import numpy as np
    def __init__(self, fmri_data_matrix, label_matrix,isTrain='train', transform=False):
        super(ERP_matrix_datasets, self).__init__()

        if not isinstance(fmri_data_matrix, np.ndarray):
            self.fmri_data_matrix = np.array(fmri_data_matrix)
        else:
            self.fmri_data_matrix = fmri_data_matrix
        
        self.Subject_Num = self.fmri_data_matrix.shape[0]
        self.Region_Num = self.fmri_data_matrix[0].shape[-1]

        if isinstance(label_matrix, pd.DataFrame):
            self.label_matrix = label_matrix
        elif isinstance(label_matrix, np.ndarray):
            self.label_matrix = pd.DataFrame(data=np.array(label_matrix))      
        self.data_type = isTrain
        self.transform = transform

    def __len__(self):
        return self.Subject_Num

    def __getitem__(self, idx):
        #step1: get one subject data
        fmri_trial_data = self.fmri_data_matrix[idx]
        fmri_trial_data = fmri_trial_data.reshape(1,-1)
        label_trial_data = np.array(self.label_matrix.iloc[idx])
#         print('fmri_trial_data\n{}\n======\nlabel_trial_data\n{}\n'.format(fmri_trial_data.shape,label_trial_data.shape))
        tensor_x = torch.stack([torch.FloatTensor(fmri_trial_data[ii]) for ii in range(len(fmri_trial_data))])  # transform to torch tensors
        tensor_y = torch.stack([torch.LongTensor([label_trial_data[ii]]) for ii in range(len(label_trial_data))])
#         print('tensor_x\n{}\n=======\ntensor_y\n{}\n'.format(tensor_x.size(),tensor_y.size()))
        return tensor_x, tensor_y

# Net fucntion of classification
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_out):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)
        self.drop = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(n_hidden,n_out)
    
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.drop(x)
        x = self.softmax(x)
        x = self.classifier(x)
        
        return x
    
def model_fit_evaluate(model,train_loader,test_loader,optimizer,loss_func,num_epochs=100):
    best_acc = 0 
    model_history={}
    model_history['train_loss']=[];
    model_history['train_acc']=[];
    model_history['test_loss']=[];
    model_history['test_acc']=[]; 
    for epoch in range(num_epochs):
        train_loss,train_acc =train(model, train_loader, optimizer,loss_func, epoch)
        model_history['train_loss'].append(train_loss)
        model_history['train_acc'].append(train_acc)

        test_loss,test_acc = test(model, test_loader,loss_func)
        model_history['test_loss'].append(test_loss)
        model_history['test_acc'].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            print("Model updated: Best-Acc = {:4f}".format(best_acc))
    for ii in range(20):
        torch.cuda.empty_cache()
    print("best testing accuarcy:",best_acc)    

##training the model
def train(model,train_loader,optimizer,loss_func, epoch):
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc = 0.
    train_loss = 0.
    total = 0
    t0 = time.time()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
#         print('out:',out[0],'\nlabel:',target.reshape(-1).tolist()[0])
        loss = loss_func(out[0],target.reshape(-1))
        pred = F.log_softmax(out, dim=1).argmax(dim=1).reshape(-1)[target.reshape(-1).tolist()[0]] #target.reshape(-1).tolist()[0]
#         print('pred:',pred)
        total += target.size(0)
        train_loss += loss.sum().item()
        acc += pred.eq(target.view_as(pred)).sum().item()
        
        loss.backward()
        optimizer.step()
        
    print("\nEpoch {}: \nTime Usage:{:4f} | Training Loss {:4f} | Acc {:4f}".format(epoch,time.time()-t0,train_loss/total,acc/total))
    return train_loss/total,acc/total

def test(model, test_loader,loss_func):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss=0.
    test_acc = 0.
    total = 0
    ##no gradient desend for testing
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ## saliency maps
            out = model(data)
            
            loss = loss_func(out[0],target.reshape(-1))
            test_loss += loss.sum().item()
            pred = F.log_softmax(out, dim=1).argmax(dim=1).reshape(-1)[target.reshape(-1).tolist()[0]]
            #pred = out.argmax(dim=1,keepdim=True) # get the index of the max log-probability
            total += target.size(0)
            test_acc += pred.eq(target.view_as(pred)).sum().item()            
    
    test_loss /= total
    test_acc /= total
    print('Test Loss {:4f} | Acc {:4f}'.format(test_loss,test_acc))
    return test_loss,test_acc    
    
def classfier(FT,labels,num_FT,num_class):
    block_dura = 64    
    test_size = 0.2
    randomseed=1234
    test_sub_num = len(FT)
    print('test_sub_num ',test_sub_num)
    rs = np.random.RandomState(randomseed)
    train_sid, test_sid = train_test_split(range(test_sub_num), test_size=test_size, random_state=rs, shuffle=True)
    print('training on %d subjects, validating on %d subjects' % (len(train_sid), len(test_sid)))
    ####train set 
    data_train = [FT[i] for i in train_sid]
    label_data_train = pd.DataFrame(np.array([labels[i] for i in train_sid]))
#     print(type(label_data_train),'\n',label_data_train)
    train_dataset = ERP_matrix_datasets(data_train, label_data_train, isTrain='train')
    train_loader = DataLoader(train_dataset)
    ####test set
    data_test = [FT[i] for i in test_sid]
    label_data_test = pd.DataFrame(np.array([labels[i] for i in test_sid]))
#     print(type(label_data_test),'\n',label_data_test)
    test_dataset = ERP_matrix_datasets(data_test, label_data_test, isTrain='test')
    test_loader = DataLoader(test_dataset)
    # model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden = int(num_FT/2)
    model = Net(num_FT,hidden,num_class).to(device)
    # initial
    loss_func = nn.CrossEntropyLoss()
    num_epochs=3
    optimizer = optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
    model_fit_evaluate(model,train_loader,test_loader,optimizer,loss_func,num_epochs)
    