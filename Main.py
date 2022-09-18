import mne
import os
import scipy.io as scio
import random
from mne.datasets import sample
import numpy as np
from mne.minimum_norm import make_inverse_operator, apply_inverse
import torch
import torch.nn as nn
from model.CRNN import CRNN
import torchvision.transforms as transforms
import PIL.Image as Image
from untils import subsample,classfier
import pandas as pd
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2,SelectFromModel
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

## different space slidwin with CRNN
def spCRNN(data,ERP_data,model,file,sWinSize):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FTS_data = []
    label = []
    mE = CRNN(1,1000,61).to(device)
    mR = nn.Bilinear(1000,1000,148).to(device)
    for num in range(0,np.array(data).shape[0],sWinSize) :
        temp = data[num:num+sWinSize,:]
#         print('temp shape:',np.array(temp).shape,'\nload model ...')
        if np.array(temp).shape[0] == sWinSize :
            ## model structure
            FT_sour = model(torch.cuda.FloatTensor(temp).reshape(1,np.array(temp).shape[0],700).permute(2,0,1).to(device))
#             print('ERP_data shape:',np.array(ERP_data['Category_1_Average']).shape)
#             corr = mR(FT_sour.reshape(1,-1),torch.mean(torch.FloatTensor(ERP_data['Category_1_Average']),dim=0).reshape(1,-1).to(device))
#             FT_sour = (FT_sour*abs(corr)).reshape(-1)
            temp = mE(torch.cuda.FloatTensor(ERP_data['Category_1_Average'].tolist()).reshape(1,61,700).permute(2,0,1).to(device))
            FT_sour = mR(FT_sour.reshape(1,-1),temp.reshape(1,-1)).reshape(-1)
            if 'aIFG' in file :
                # temp = [random.expovariate(0.4) for _ in range(0,1000)] # [i+j for i,j in zip(FT_sour.tolist(),temp)]
                FTS_data.append(FT_sour.tolist())
                label.append([0.0])  ## 0 represent stimulus with aIFG brain regions
            elif 'pIFG' in file :
                # temp = [random.betavariate(0.35,0.55) for _ in range(0,1000)]
                FTS_data.append(FT_sour.tolist())
                label.append([1.0])  ## 1 represent stimulus with pIFG brain regions
            elif 'sham' in file :
                # temp = [random.uniform(0.1,0.3) for _ in range(0,1000)]
                FTS_data.append(FT_sour.tolist())
                label.append([2.0])  ## 2 represent stimulus with sham
            else :
                print('*********************** Warning: do not identify condition ***********************')
    return FTS_data,label

def stastic(ERP_comp):
    MEN = np.mean(ERP_comp)
    std = np.std(ERP_comp)
    med = np.median(ERP_comp)
    Max = np.amax(ERP_comp)
    if ERP_comp.ndim == 2 :
        Pow = np.trapz(np.mean(ERP_comp,axis=0),np.array([i for i in range(np.array(ERP_comp).shape[1])]),dx=0.001)
    elif ERP_comp.ndim == 1 :
        Pow = np.trapz(ERP_comp,np.array([i for i in range(np.array(ERP_comp).shape[0])]),dx=0.001)
    return MEN,std,med,Max,Pow

def DetrieuxFT(source_dir,file):
    timeserials = {}
#     f = open('/media/lhj/Momery/causalML/IFG_Source_Causal/data/Destrieux1.txt','r')
#     Scounts = f.readlines()
    data_dir = source_dir+'/'+file[0:file.rfind('-')]+'/'+file[0:file.rfind('_source.mat')]+'_seg_blc'
    for ii in os.listdir(data_dir):
        if 'matrix_scout' in ii :
            data = scio.loadmat(data_dir+'/'+ii)
            ll = list((data.keys()))
            Des = data['Description']
            Des1 = []
            for i in range(len(Des)):
                temp1 = Des[i][0][0]
                temp1 = temp1[0:temp1.rfind('@')]
                Des1.append(temp1[0:temp1.rfind('.')])
            Des1 = np.unique(np.array([Des1])).tolist()
            print('Destrieux:',np.array(Des1).shape)
            Val = data['Value']
            for line in Des1 :
                temp = []
#                 line = line.rstrip('\n')
                for i in range(len(Des)):
                    if line in str(Des[i][0][0]) :
#                         print('++++++++++++++++ exist ++++++++++++++++++')
                        timeserials[line] = torch.cat((torch.tensor(temp),torch.tensor(Val[i])),0)                        
                        
#     print(len(timeserials),timeserials.keys())                           
    return timeserials

## load ERP.mat source functional activation dataset # ,'Power':np.trapz(np.mean(N1,dim=0),np.array([i for i in range(200,400)]),dx=0.001)
def load_ERPS(ERP_dir):
    import csv
    files = os.listdir(ERP_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sWinSize = 50
    FTS_data = []
    label = []
    print("++++++++++++++++++++++ load data ... ++++++++++++++++++++++") 
    with open('/media/lhj/Momery/causalML/IFG_Source_Causal/data/N1P2.csv','w') as file:
        filedNames = ['Name','TimeSerials','Mean','std','median','Max','Power']
        writer = csv.DictWriter(file,fieldnames=filedNames)
        writer.writeheader()
        for file in files: 
            data=scio.loadmat(ERP_dir+'/'+file) 
            ll = list((data.keys()))
            ERP_data = scio.loadmat('/media/lhj/Disk/Freesurfer/BrainStormWorkshop/ERP/'+file[0:file.rfind('_source.mat')]+'_seg_blc.mat')
            N1 = ERP_data['Category_1_Average'][:,200:400]
            P2 = ERP_data['Category_1_Average'][:,400:600]
            MEN,std,med,Max,Pow = stastic(N1)
#             print(type(N1),np.array(N1).shape,'\n',type(P2),np.array(P2).shape)
            writer.writerow({'Name':file[0:file.rfind('_source.mat')],'TimeSerials':'N1[200:400]','Mean':MEN,'std':std,'median':med,'Max':Max,'Power':Pow}) 
            MEN,std,med,Max,Pow = stastic(P2)
            writer.writerow({'Name':file[0:file.rfind('_source.mat')],'TimeSerials':'P2[400:600]','Mean':MEN,'std':std,'median':med,'Max':Max,'Power':Pow}) 
            ## extract timeserials with scounts of Detrieux
            source_dir = '/media/lhj/Disk/Freesurfer/BrainStormWorkshop/workspace/IFG_Individual/data'
            timeserials = DetrieuxFT(source_dir,file)
            for kk in timeserials.keys():
                MEN,std,med,Max,Pow = stastic(np.array(timeserials[kk].tolist()))
                writer.writerow({'Name':file[0:file.rfind('_source.mat')],'TimeSerials':kk,'Mean':MEN,'std':std,'median':med,'Max':Max,'Power':Pow})
            for name in ll :
                if 'A' in name or 'A1' in name :
                    model = CRNN(1,1000,200).to(device)
                    FTS_data1,label1 = spCRNN(data[name],ERP_data,model,file,200)
    #                 print('FTS1 shape:',np.array(FTS_data1).shape,'label1 shape:',np.array(label1).shape)
                    model = CRNN(1,1000,150).to(device)
                    FTS_data2,label2 = spCRNN(data[name],ERP_data,model,file,150)
    #                 print('FTS2 shape:',np.array(FTS_data2).shape,'label1 shape:',np.array(label2).shape)
                    model = CRNN(1,1000,100).to(device)
                    FTS_data3,label3 = spCRNN(data[name],ERP_data,model,file,100)
    #                 print('FTS3 shape:',np.array(FTS_data1).shape,'label3 shape:',np.array(label1).shape)
                    model = CRNN(1,1000,50).to(device)
                    FTS_data4,label4 = spCRNN(data[name],ERP_data,model,file,50)
    #                 print('FTS4 shape:',np.array(FTS_data2).shape,'label4 shape:',np.array(label2).shape)
                    FTS_data = torch.cat((torch.tensor(FTS_data),torch.tensor(FTS_data1),torch.tensor(FTS_data2),torch.tensor(FTS_data3),torch.tensor(FTS_data4)),0)
                    label = torch.cat((torch.tensor(label),torch.tensor(label1),torch.tensor(label2),torch.tensor(label3),torch.tensor(label4)),0)                                                
    FTS_data = FTS_data.tolist()
    label = label.tolist()
    print('FTS shape:',np.array(FTS_data).shape,'label shape:',np.array(label).shape)
    
    return FTS_data,label

## load source picture of brain
def load_Pho(Pho_dir):
    files = os.listdir(Pho_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True).to(device)
    FTS_data = []
    label = []
    print(model)
    print("++++++++++++++++++++++ load data ... ++++++++++++++++++++++") 
    for file in files: 
        image = Image.open(Pho_dir+'/'+file)
#         print(image.size,image.format,image.mode)
        crop_obj = transforms.CenterCrop((840,1058))
        image = crop_obj(image)
        image.save(Pho_dir+'/Crop1_'+file,format='PNG')
        enhance = transforms.Compose([transforms.RandomRotation(45),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(p=0.5),transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1),transforms.RandomGrayscale(p=0.025),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        for ii in range(500) :
            image1 = enhance(image)
            FTS_Pho = model(image1.reshape(1,image1.size()[0],image1.size()[1],image1.size()[2]).to(device)).reshape(-1)
#             print('image ResNet50 FT '+str(ii),FTS_Pho.size())
            if 'aIFG' in file :
                # temp = [random.expovariate(0.4) for _ in range(0,1000)]
                FTS_data.append(FTS_Pho.tolist())
                label.append([0.0])  ## 0 represent stimulus with aIFG brain regions
            elif 'pIFG' in file :
                # temp = [random.betavariate(0.2,0.5) for _ in range(0,1000)]
                FTS_data.append(FTS_Pho.tolist())
                label.append([1.0])  ## 1 represent stimulus with pIFG brain regions
            elif 'sham' in file :
                # temp = [random.uniform(0.2,0.5) for _ in range(0,1000)]
                FTS_data.append(FTS_Pho.tolist())
                label.append([2.0])  ## 2 represent stimulus with sham
            else :
                print('*********************** Warning: do not identify condition ***********************')
    print('FTS shape:',np.array(FTS_data).shape,'label shape:',np.array(label).shape)
    
    return FTS_data,label       

def visFeature(FTS_data,label):
    import matplotlib.pyplot as plt
    
    pca = PCA(n_components=2)
    FT = pca.fit_transform(np.array(FTS_data))
    for ii in range(np.array(label).shape[0]):
#         print('label:',label[ii],label[ii][0])
        if label[ii][0]==0 :
            print('aIFG')
#             plt.scatter(FT[ii][0],FT[ii][1],c="r",alpha=0.5,label="aIFG")
        elif label[ii][0]==1 :
#             print('pIFG')
            plt.scatter(FT[ii][0],FT[ii][1],c="b",alpha=0.5,label="pIFG") 
        elif label[ii][0]==2 :
            plt.scatter(FT[ii][0],FT[ii][1],c="g",alpha=0.5,label="sham")  
        else :
            print('*********** group did not exit **************')
    plt.title("groups distribution")
    plt.show()
    
def sublabel(FTS_data,label,class1,class2):
    sub_data = []
    sub_label = []
    for ii in range(np.array(label).shape[0]):
        if label[ii][0] == class1 :
            sub_data.append(FTS_data[ii])
            sub_label.append([0.0]) 
        elif label[ii][0] == class2 :
            sub_data.append(FTS_data[ii])
            sub_label.append([1.0]) 
    print('sub_data:',np.array(sub_data).shape,'\nsub_label:',np.array(sub_label).shape)
    return sub_data,sub_label

def FTdecom(FTS_data,label,num_Decom):
    pca = PCA(n_components=num_Decom)
    FT = pca.fit_transform(np.array(FTS_data))
    
    return FT,label

def mainFunction():
    ERP_dir = '/media/lhj/Disk/Freesurfer/BrainStormWorkshop/Result/Matrix1'
    FTS_data1,label1 = load_ERPS(ERP_dir)
    ## Visual distribution of Feature in different groups
#     FT,label = subsample(FTS_data1,label1,0.5)
#     visFeature(FTS_data1,label1)
    Pho_dir = '/media/lhj/Disk/Freesurfer/BrainStormWorkshop/Result/Pho_ind'
#     FTS_data2,label2 = load_Pho(Pho_dir)
    # Preprocess
#     FTS_data1,label1 = subsample(FTS_data1,label1,0.99)    
#     FTS_data2,label2 = subsample(FTS_data2,label2,0.99)

#     FTS_data = torch.cat((torch.tensor(FTS_data1),torch.tensor(FTS_data2)),0)
#     label = torch.cat((torch.tensor(label1),torch.tensor(label2)),0)

    ## VarianceThreshold
#     sel = VarianceThreshold(threshold=(.6*(1-.6)))
#     FTS_data = sel.fit_transform(FTS_data).tolist()
    ## SelectFromModel
#     lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(FTS_data1, label1)
#     FTS_data1 = SelectFromModel(lsvc, prefit=True).transform(FTS_data1)
        
#     FTS_data,label = subsample(FTS_data2.tolist(),label2.tolist(),0.99)
#     for num_FT in range(10,148,2):
#         FTS_data,label = FTdecom(FTS_data2,label2,num_FT)
#         print('+++++++++++ Number of Feature with '+str(num_FT)+' by PCA +++++++++++++++++++++')
#         FTS_data,label = sublabel(FTS_data,label,1,2)
#         print('+++++++++++ class between pIFG and sham +++++++++++++++++++++')
#         classfier(FTS_data,label,num_FT,3)
#     classfier(FTS_data2,label2,1000,3)
#     classfier(FTS_data.tolist(),label.tolist(),1000,3)
    

mainFunction()
    