import base64
import io
import os
import time
os.sys.path.append('/Users/soniajaiswal/miniforge3/lib/python3.9/site-packages')
import torch, torchvision
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import glob
from tqdm import tqdm
import random
from random import randrange, randint
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torch.nn as nn
import torch.nn.functional as F

import math, base64, io, os, time, cv2

import time
from IPython.display import clear_output

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from torch.utils.tensorboard import SummaryWriter
from time import time

import multiprocessing

frame_per_vid = 64
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Cuda availability : " + str(device))

class Sims(nn.Module):
    def __init__(self):
        super(Sims, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        '''(N, S, E)  --> (N, 1, S, S)'''
        f = x.shape[1]
        
        I = torch.ones(f).to(self.device)
        xr = torch.einsum('bfe,h->bhfe', (x, I))   #[x, x, x, x ....]  =>  xr[:,0,:,:] == x
        xc = torch.einsum('bfe,h->bfhe', (x, I))   #[x x x x ....]     =>  xc[:,:,0,:] == x
        diff = xr - xc
        out = torch.einsum('bfge,bfge->bfg', (diff, diff))
        out = out.unsqueeze(1)
        #out = self.bn(out)
        out = F.softmax(-out/13.544, dim = -1)
        return out

#---------------------------------------------------------------------------

class ResNet50Bottom(nn.Module):
    def __init__(self):
        super(ResNet50Bottom, self).__init__()
        self.original_model = torchvision.models.resnet50(pretrained=True, progress=True)
        self.activation = {}
        h = self.original_model.layer3[2].register_forward_hook(self.getActivation('comp'))
        
    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, x):
        self.original_model(x)
        output = self.activation['comp']
        return output

#---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x 

#----------------------------------------------------------------------------

class TransEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers = 1):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, 64)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                    nhead = n_head,
                                                    dim_feedforward = dim_ff,
                                                    dropout = dropout,
                                                    activation = 'relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
                
    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


#=============Model====================


class RepNet(nn.Module):
    def __init__(self, num_frames):
        super(RepNet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.num_frames = num_frames
        self.resnetBase = ResNet50Bottom()
        
        
        self.conv3D = nn.Conv3d(in_channels = 1024,
                                out_channels = 512,
                                kernel_size = 3,
                                padding = (3,1,1),
                                dilation = (3,1,1))
        self.bn1 = nn.BatchNorm3d(512)
        self.pool = nn.MaxPool3d(kernel_size = (1, 7, 7))
        
        self.sims = Sims()
        self.mha_sim = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        
        self.conv3x3 = nn.Conv2d(in_channels = 2,
                                 out_channels = 32,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.ln1 = nn.LayerNorm(512)
        
        self.transEncoder1 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 1)
        self.transEncoder2 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 1)
        
        #period length prediction
        self.fc1_1 = nn.Linear(512, 512)
        self.ln1_2 = nn.LayerNorm(512)
        self.fc1_2 = nn.Linear(512, self.num_frames//2)
        self.fc1_3 = nn.Linear(self.num_frames//2, 1)


        #periodicity prediction
        self.fc2_1 = nn.Linear(512, 512)
        self.ln2_2 = nn.LayerNorm(512)
        self.fc2_2 = nn.Linear(512, self.num_frames//2)
        self.fc2_3 = nn.Linear(self.num_frames//2, 1)

    def forward(self, x, ret_sims = False):
        batch_size, _, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.resnetBase(x)
        x = x.view(batch_size, self.num_frames, x.shape[1],  x.shape[2],  x.shape[3])
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv3D(x)))
                        
        x = x.view(batch_size, 512, self.num_frames, 7, 7)
        x = self.pool(x).squeeze(3).squeeze(3)
        x = x.transpose(1, 2)                           #batch, num_frame, 512
        x = x.reshape(batch_size, self.num_frames, -1)

        x1 = F.relu(self.sims(x))
        
        
        x = x.transpose(0, 1)
        _, x2 = self.mha_sim(x, x, x)
        x2 = F.relu(x2.unsqueeze(1))
        x = torch.cat([x1, x2], dim = 1)
        
        xret = x
        # print(xret.shape)
        
        x = F.relu(self.bn2(self.conv3x3(x)))     #batch, 32, num_frame, num_frame
        #print(x.shape)
        x = self.dropout1(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.num_frames, -1)  #batch, num_frame, 32*num_frame
        x = self.ln1(F.relu(self.input_projection(x)))  #batch, num_frame, d_model=512
        
        x = x.transpose(0, 1)                          #num_frame, batch, d_model=512
        
        #period
        x1 = self.transEncoder1(x)
        y1 = x1.transpose(0, 1)
        y1 = F.relu(self.ln1_2(self.fc1_1(y1)))
        y1 = F.relu(self.fc1_2(y1))
        y1 = F.relu(self.fc1_3(y1))

        #periodicity
        x2 = self.transEncoder2(x)
        y2 = x2.transpose(0, 1)
        y2 = F.relu(self.ln2_2(self.fc2_1(y2)))
        y2 = F.relu(self.fc2_2(y2))
        y2 = F.relu(self.fc2_3(y2)) 
        
        #y1 = y1.transpose(1, 2)                         #Cross enropy wants (minbatch*classes*dimensions)
        if ret_sims:
            return y1, y2, xret
        return y1, y2


def training_loop(writer,
                  n_epochs,
                  model,
                  train_set,
                  val_set,
                  batch_size,
                  lr = 6e-6,
                  ckpt_path = 'ckpt',
                  use_count_error = True,
                  saveCkpt= True,
                  train = True,
                  validate = True,
                  lastCkptPath = None,):    
    
    prevEpoch = 0
    trainLosses = []
    valLosses = []
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)

    if lastCkptPath != None :
        print("loading checkpoint")
        checkpoint = torch.load(lastCkptPath)
        prevEpoch = checkpoint['epoch']
        trainLosses = checkpoint['trainLosses']
        valLosses = checkpoint['valLosses']

        model.load_state_dict(checkpoint['state_dict'], strict = True)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        del checkpoint
    
        
    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    lossMAE = torch.nn.SmoothL1Loss()
    lossBCE = torch.nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              num_workers=1, 
                              shuffle = True)
    
    val_loader = DataLoader(val_set,
                            batch_size = batch_size,
                            num_workers=1,
                            drop_last = False,
                            shuffle = True)
    
    if validate and not train:
        currEpoch = prevEpoch
    else :
        currEpoch = prevEpoch + 1
        
    for epoch in range(currEpoch, n_epochs + currEpoch):
        #train loop
        if train :
            # pbar = tqdm(train_loader, total = len(train_loader))
            mae = 0
            mae_count = 0
            fscore = 0
            i = 1
            a=0
            for i_batch, sample_batch in enumerate(train_loader):
                X, y = sample_batch
                torch.cuda.empty_cache()
                model.train()
                X = X.to(device).float()
                y1 = y.to(device).float()
                y2 = getPeriodicity(y1).to(device).float()
                
                y1pred, y2pred = model(X)
                loss1 = lossMAE(y1pred, y1)
                loss2 = lossBCE(y2pred, y2)

                loss = loss1 + 5*loss2

                countpred = torch.sum((y2pred > 0) / (y1pred + 1e-1), 1)
                count = torch.sum((y2 > 0) / (y1 + 1e-1), 1)
                loss3 = torch.sum(torch.div(torch.abs(countpred - count), (count + 1e-1)))

                if use_count_error:    
                    loss += loss3
                
                optimizer.zero_grad()
                loss.backward()
                               
                optimizer.step()
                train_loss = loss.item()
                trainLosses.append(train_loss)
                mae += loss1.item()
                mae_count += loss3.item()
                
                del X, y, y1, y2, y1pred, y2pred
                i+=1
                # pbar.set_postfix({'Epoch': epoch,
                #                   'MAE_period': (mae/i),
                #                   'MAE_count' : (mae_count/i),
                #                   'Mean Tr Loss': np.mean(trainLosses[-i+1:])})
                writer.add_scalar('train_MAE_period', (mae/i), epoch*(len(train_loader))+(i))
                writer.add_scalar('train_MAE_count', (mae_count/i), epoch*(len(train_loader))+(i))
                writer.add_scalar('train_Mean_Loss', np.mean(trainLosses[-i+1:]), epoch*(len(train_loader))+(i))
                
            writer.add_scalar('epoch_train_MAE_period', (mae/i), epoch)
            writer.add_scalar('epoch_train_MAE_count', (mae_count/i), epoch)
            writer.add_scalar('epoch_train_Mean_Loss', np.mean(trainLosses[-i+1:]), epoch)
            print('epoch_train_MAE_period', (mae/i), epoch)
            print('epoch_train_MAE_count', (mae_count/i), epoch)
            print('epoch_train_Mean_Loss', np.mean(trainLosses[-i+1:]), epoch)
                
        if validate:
            #validation loop
            with torch.no_grad():
                mae = 0
                mae_count = 0
                fscore = 0
                i = 1
                # pbar = tqdm(val_loader, total = len(val_loader))
                for i_batch, sample_batch in enumerate(val_loader):
                    X, y = sample_batch
                    torch.cuda.empty_cache()
                    model.eval()
                    X = X.to(device).float()
                    y1 = y.to(device).float()
                    y2 = getPeriodicity(y1).to(device).float()
                    
                    y1pred, y2pred = model(X)
                    loss1 = lossMAE(y1pred, y1)
                    loss2 = lossBCE(y2pred, y2)
                    
                    loss = loss1 + loss2
                    
                    countpred = torch.sum((y2pred > 0) / (y1pred + 1e-1), 1)
                    count = torch.sum((y2 > 0) / (y1 + 1e-1), 1)
                    loss3 = lossMAE(countpred, count)

                    if use_count_error:    
                        loss += loss3
                                
                    val_loss = loss.item()
                    valLosses.append(val_loss)
                    mae += loss1.item()
                    mae_count += loss3.item()
                    
                    del X, y, y1, y2, y1pred, y2pred
                    i+=1
                    # pbar.set_postfix({'Epoch': epoch,
                    #                 'MAE_period': (mae/i),
                    #                 'MAE_count' : (mae_count/i),
                    #                 'Mean Val Loss':np.mean(valLosses[-i+1:])})
                    writer.add_scalar('val_MAE_period', (mae/i), epoch*(len(train_loader))+(i))
                    writer.add_scalar('val_MAE_count', (mae_count/i), epoch*(len(train_loader))+(i))
                    writer.add_scalar('val_Mean_Loss', np.mean(trainLosses[-i+1:]), epoch*(len(train_loader))+(i))

                writer.add_scalar('epoch_val_MAE_period', (mae/i), epoch)
                writer.add_scalar('epoch_val_MAE_count', (mae_count/i), epoch)
                writer.add_scalar('epoch_val_Mean_Loss', np.mean(trainLosses[-i+1:]), epoch)
                            
        #save checkpoint
        if saveCkpt:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLosses' : trainLosses,
                'valLosses' : valLosses
            }
            torch.save(checkpoint, ckpt_path + str(epoch) + '.pt')
            old_ckpt_path = ''
            if prevEpoch != epoch-1:
              old_ckpt_path = ckpt_path + str(epoch-1) + '.pt'
            try:
                os.remove(old_ckpt_path)
            except OSError:
                pass
        
        #lr_scheduler.step()

    return trainLosses, valLosses
def predSmall(frames, model):

    Xlist = []
    for img in frames:

        preprocess = transforms.Compose([
        transforms.Resize((112, 112), 2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        frameTensor = preprocess(img).unsqueeze(0)
        Xlist.append(frameTensor)

    X = torch.cat(Xlist)
    #print(X.shape)
    with torch.no_grad():
        model.eval()
        y1pred, y2pred, sim = model(X.unsqueeze(0).to(device), True)
    
    periodLength = y1pred.round().long()
    periodicity = y2pred > 0
    #print(periodLength.squeeze())
    #print(periodicity.squeeze())
    
    sim = sim[0,0,:,:]
    sim = sim.detach().cpu().numpy()
    
    return X, periodLength, periodicity, sim


def getAnim(X, countPred = None, count = None, idx = None):
    
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['animation.html'] = "jshtml"

    fig, ax = plt.subplots()
    axesimg = ax.imshow(np.zeros((100,100, 3)))
    
    def animate(i):
        title = " "
        if countPred is not None:
            title += "pred"
            title += str(countPred[i])
        if count is not None:
            title += " actual"
            title += str(count[i])
        if idx is not None:
            title += " id"
            title += str(idx)
        
        ax.set_title(title)
        img = X[i,:,:,:].transpose(0, 1).transpose(1,2).detach().cpu().numpy()
        ax.imshow(img)

    anim = FuncAnimation(fig, animate, frames=64, interval=500)
    
    return anim

def getAnimOriginal(frames, countPred):
    from google.colab.patches import cv2_imshow
    fig, ax = plt.subplots()
    for i in range(len(frames)):
      title = " "
      title += "pred"
      title += str(countPred[i])

      ax.set_title(title)
      img = Image.fromarray(frames[i])
      cv2_imshow(img)
      if cv2.waitKey(1) == ord('q'):
        break

def getCountInfer(period, periodicity = None):
    period = period.round().squeeze()

    count = []

    if periodicity is None:
        periodicity = period > 2
    else :
        periodicity = periodicity.squeeze() > 0

    numofReps = 0
    for i in range(len(period)):
        if period[i] == 0:
            numofReps+=0
        else:
            numofReps += max(0, periodicity[i]/period[i])

        count.append(int(numofReps))
    return count, period, periodicity
def gc(frames):
    
    newFrames = []
    num_frames = len(frames)
    for i in range(1, num_frames + 1):
        newFrames.append(frames[i * len(frames)//num_frames  - 1])
    
    return newFrames


def predRep(model, frames,j):
#def predRep():
    
    print("num_frames", j)
    periodicitybest = []
    Xbest = None
    countbest = [-1]
    simsbest = []
    periodbest = []
    
    #frames = getFramesInfer(vidPath, 64)
    periodicity = []
    periodLength = []
    sims = []
    X = []
    i = len(frames)    
    x, periodLengthj, periodicityj, sim = predSmall(frames[j*64:(j+1)*64], model)
    periodicity.extend(list(periodicityj.squeeze().cpu().numpy()))
    periodLength.extend(list(periodLengthj.squeeze().cpu().numpy()))
    X.append(x)
    sims.append(sim)
        
    X = torch.cat(X)
    numofReps = 0
    count = []
    for i in range(len(periodLength)):
        if periodLength[i] == 0:
            numofReps += 0
        else:
            numofReps += max(0, periodicity[i]/(periodLength[i]))

        count.append(round(float(numofReps), 2))
        
    if count[-1] > countbest[-1]:
        countbest = count
        Xbest = X
        periodicitybest = periodicity
        simsbest = sims
        periodbest = periodLength
    
    return Xbest, countbest, periodicitybest, periodbest, simsbest, frames
from multiprocessing import Process, Manager
  
def sender(L):
    print("sender has started")
    i = 0
    while i<192:
        cap = cv2.VideoCapture(0)        
        ret, frame = cap.read()
            
        if ret is False:             
            break
        cv2.imshow('Frame',frame) 
        cv2.waitKey(1)  
        img = Image.fromarray(frame)    
        L.append(img)
            
            
        i = i+1
       # cap.release()
          
    """
    function to send messages to other end of pipe
    """
    
    
    
def receiver(L):
    print("receiver has started")
    # Load model from the checkpoint
    
    a = time()
    model         = RepNet(frame_per_vid)
    model         = model.to(device)
    print("loading checkpoint")



    lastCkptPath  ='/Users/soniajaiswal/Documents/Masters/Quarter3/hci/live_rep_net/RepNetTraining330.pt'
    # lastCkptPath  = rootCountixFolder+'checkpoint/x3dbb5.pt'
    checkpoint    = torch.load(lastCkptPath, map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['state_dict'], strict = True)


    time_ = []
    
    frames = []
    k = 0
    j = 0
    sum_c = 0
    length_processed = 0 
    while True:
        
        
        frames = list(L)
       # print(len(frames))
        if(len(frames) >= 64*(j+1)):
            print("the time now is", time()-a)            
            fr = gc(frames)
            print("from inside length_processed", length_processed)
            print("from inside fr length", len(fr))
            
            
            
            Xbest, countbest, periodicitybest, periodbest, simsbest, vidFrames = predRep(model, fr[0:(j+1)*64],j)
            j = j + 1
            
            sum_c = sum_c + countbest[-1]
                
                
            print("counts till now", sum_c)
                #b = time()
                #time_.append(b-a)
                #print("time to infer", b - a)
        
        
       
  
    """
    function to print the messages received from other
    end of pipe
    """
    
def rand_r(conn):
    while(True):
        print("time is now", time())
  

if __name__ == "__main__":
    
    
    
    

    
    with Manager() as manager:
        L = manager.list()  # <-- can be shared between processes.
        
        p1 = Process(target=sender, args=(L,))
        p2 = Process(target=receiver, args=(L,))
        
        p2.start()
        p1.start()
        
        p2.join()
        p1.join()
        
    
  
    # creating new processes
    

    
    
    
    
   
