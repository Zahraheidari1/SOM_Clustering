import torch as th
import torch.nn as nn
import torch.optim as op
from numpy import save,load
from math import floor,ceil
class KcompetitiveLayer(nn.Module):
    def __init__(self,a:float,k:int)->None:
        super(KcompetitiveLayer,self).__init__()
        self.k=k
        self.a=a
    def forward(self,input:th.Tensor)->th.Tensor:
        #s=input.shape
        device=th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        input=input.flatten()
        pos_indices=input>0
        pos_indices=pos_indices.to(device)
        pos=input[pos_indices]
        pos=pos.to(device)
        pos_losers=th.ones(input.shape).type(th.bool)
        pos_losers=pos_losers.to(device)
        if len(pos)>ceil(self.k/2):
            pos,_=th.sort(pos)
            Epos=th.sum(pos[0:len(pos)-ceil(self.k/2)])*self.a
            input[pos_indices]+=Epos
            pos_losers[th.topk(input,ceil(self.k/2))[1]]=False
            input[th.logical_and(pos_losers,pos_indices)]=0
        neg_indices=input<0
        neg_indices=neg_indices.to(device)
        neg=input[neg_indices]
        neg=neg.to(device)
        neg_losers=th.ones(input.shape).type(th.bool)
        neg_losers=neg_losers.to(device)
        if len(neg)>floor(self.k/2) and floor(self.k/2)>0:
            neg=th.abs(neg)
            neg,_=th.sort(neg)
            Eneg=th.sum(neg[0:len(neg)-int(self.k/2)])*self.a*-1
            input[neg_indices]+=Eneg
            neg_losers[th.kthvalue(input,floor(self.k/2))[1]]=False
            input[th.logical_and(neg_losers,neg_indices)]=0
        return input#.reshape(s)
class KATE(nn.Module):
    def __init__(self,in_size:int,out_size:int,a:float,k:int)->None:
        super(KATE,self).__init__()
        self.l1=nn.Linear(in_size,out_size)
        self.l2=KcompetitiveLayer(a,k)
        self.l3=nn.Linear(out_size,in_size)
        self.l3.weight.data=self.l1.weight.T
    def forward(self,data):
        data=self.l1(data)
        data=self.l2(data)
        data=th.tanh(data)
        data=self.l3(data)
        data=th.sigmoid(data)
        return data
    def encode(self,data):
        data=self.l1(data)
        data=th.tanh(data)
        return data
def train(dataset,epochs,in_size,out_size,k,a,lr,path):
    kate=KATE(in_size,out_size,a,k).double()
    device=th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    kate.to(device)
    dataset=th.from_numpy(dataset)
    dataset=dataset.to(device)
    optim=op.Adadelta(kate.parameters(),lr=lr)
    criterion=nn.BCELoss()
    runnig_loss=0
    for e in range(epochs):
        runnig_loss=0
        for i in dataset:
            optim.zero_grad()
            output=kate(i)
            loss=criterion(output,i[0])
            loss.backward()
            runnig_loss+=loss.item()
            optim.step()
        print(f'{e+1}->{runnig_loss/len(dataset)}')
    results=th.tensor([])
    for i in dataset:
        results=th.cat((results,kate.encode(i)),dim=0)
    results=results.detach().numpy()
    save(path,results)
    return runnig_loss