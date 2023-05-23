import numpy as np
from scipy.special import expit
dsigmoid=lambda x:expit(x)*(1-expit(x))
dtanh=lambda x:1-np.tanh(x)**2
loss=lambda y,p:-1*np.sum(y*np.log(p)+(1-y)*np.log(1-p),axis=0)
binary_cross_entrophy=lambda y,p:-y/p+(1-y)/(1-p)
class Dense_layer:
    def __init__(self,input_size,output_size):
        np.random.seed(0)
        self.weight=np.random.randn(output_size,input_size)
        self.bias=np.zeros((output_size,1))
    def setWeight(self,w):
        self.weight=w
        self.bias=np.zeros((w.shape[0],1))
    def reverse_layer(self):
        r=Dense_layer(1,1)
        r.setWeight(self.weight.T)
        return r
    def forward(self,data):
        self.input=data
        return np.dot(self.weight,self.input)+self.bias
    def backward(self,err,learning_rate):
        weight_gradiant=np.dot(err,self.input.T)
        w=self.weight.T
        self.weight-=learning_rate*weight_gradiant
        self.bias-=learning_rate*err
        return np.dot(w,err)
class Activation_layer:
    def __init__(self,function):
        self.function=function
        if function=='tanh':
            self.f=np.tanh
            self.df=dtanh
        else:
            self.f=expit
            self.df=dsigmoid
    def forward(self,data):
        self.input=data
        return self.f(self.input)
    def backward(self,err,learning_rate):
        return np.multiply(err,self.df(self.input))
class Competetion_layer:
    def __init__(self,k,a) -> None:
        self.a=a
        self.k=k
        self.pos_winners=list()
        self.pos_losers=list()
        self.neg_winners=list()
        self.neg_losers=list()
    def forward(self,z):
        result=np.zeros((len(z),1))
        pos=list()
        neg=list()
        for j,i in enumerate(z):
            if i>0:
                pos.append([j,i[0]])
            elif i<0:
                neg.append([j,i[0]])
        pos=sorted(pos,key=lambda x:x[1])
        neg=sorted(neg,key=lambda x:x[1],reverse=True)
        P=len(pos)
        N=len(neg)
        if P-int(self.k/2)>0:
            Epos=0
            i=0
            while i<P-int(self.k/2):
                self.pos_losers.append(pos[i][0])
                Epos+=pos[i][1]
                pos[i][1]=0
                i+=1
            i=P-int(self.k/2)
            while i<P:
                pos[i][1]+=Epos*self.a
                i+=1
        if N-int(self.k/2)>0:
            Eneg=0
            i=0
            while i<N-int(self.k/2):
                self.neg_losers.append(neg[i][0])
                Eneg-=neg[i][1]
                neg[i][1]=0
                i+=1
            i=N-int(self.k/2)
            while i<N:
                neg[i][1]-=self.a*Eneg
                i+=1
        for i in pos+neg:
            result[i[0]][0]=i[1]
        for j,i in enumerate(result):
            if i>0:
                self.pos_winners.append(j)
            elif i<0:
                self.neg_winners.append(j)
        return result
    def backward(self,z):
        Gpos=np.sum(z[self.pos_winners],axis=0)[0]*self.a
        Gneg=np.sum(z[self.neg_winners],axis=0)[0]*self.a
        z[self.neg_losers]+=Gneg
        z[self.pos_losers]+=Gpos
        self.neg_winners.clear()
        self.neg_losers.clear()
        self.pos_losers.clear()
        self.pos_winners.clear()
        return z