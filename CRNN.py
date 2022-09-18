import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self,inchannel,output,sWinSize):
        super(CRNN, self).__init__()
        self.output = output
        self.conv1 = nn.Conv1d(1, 16, 3, stride=2)
        self.LSTM1 = nn.LSTM(700, 16, bidirectional=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv1d(16, 32, 3, stride=2)
        if sWinSize==50:
            self.LSTM2 = nn.LSTM(11, 32, bidirectional=True)  ## 200-49 50-11 150-36 100-24 700-174 1000-249 250-61 300-74 30-6 20-4 70-16 80-19
        elif sWinSize==200:
            self.LSTM2 = nn.LSTM(49, 32, bidirectional=True)
        elif sWinSize==100:
            self.LSTM2 = nn.LSTM(24, 32, bidirectional=True)
        elif sWinSize==150:
            self.LSTM2 = nn.LSTM(36, 32, bidirectional=True)  
        elif sWinSize==61:
            self.LSTM2 = nn.LSTM(14, 32, bidirectional=True)     
        self.bn2 = nn.BatchNorm2d(32)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2976,output)
    
    def forward(self, x):    
        out = self.conv1(x)
#         print('cov1 shape:',out.size())
        out,(h,c) = self.LSTM1(out.permute(2,1,0))
#         print('LSTM1 shape:',out.size(),h.size(),c.size())
        temp = out.reshape(1,out.size()[0],out.size()[1],out.size()[2])
        temp = temp.permute(0,2,3,1)
#         print('temp shape:',temp.size())
        out = self.bn1(temp)
#         print('bn1 shape:',out.size())
        out = self.relu(out) 
#         print('relu shape\n',out.size())
        out = self.maxpool(out)
#         print('maxpool shape\n',out.size())
        temp = out.reshape(out.size()[1],out.size()[2],out.size()[3]).permute(2,0,1)
        out = self.conv2(temp)
#         print('cov2 shape:',out.size())
        out,(h,c) = self.LSTM2(out.permute(2,1,0))
#         print('LSTM2 shape:',out.size(),h.size(),c.size())
        temp = out.reshape(1,out.size()[0],out.size()[1],out.size()[2])
        temp = temp.permute(0,2,3,1)
#         print('temp shape:',temp.size())
        out = self.bn2(temp)
#         print('bn2 shape:',out.size())
        out = self.relu(out) 
#         print('relu shape\n',out.size())
        out = self.maxpool(out)
        out = self.drop(out)
#         print('maxpool shape\n',out.size())        
        temp = out.reshape(1,-1)
#         print('temp shape:',temp.size())
        out = self.fc(temp).reshape(-1)
#         print('FC data\n',out.size())
        
        return out