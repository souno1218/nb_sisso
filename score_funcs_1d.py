from numba import njit
import numpy as np

@njit(error_model="numpy")#,fastmath=True)
def debug_1d(x,y):#x.shape=(N),y=ans
    x_=x-np.mean(x)
    return np.abs(x_[0]/np.sqrt(np.sum(x_**2)))

@njit(error_model="numpy")#,fastmath=True)
def LDA_1d(x,y):#x.shape=(N),y=ans
    mu_T=np.mean(x[y])
    mu_F=np.mean(x[~y])
    sigma=(np.sum((x[y]-mu_T)**2)+np.sum((x[~y]-mu_F)**2))/(y.shape[0])
    f=(mu_T-mu_F)*x-(mu_T-mu_F)*(mu_T+mu_F)/2+sigma*np.log(np.sum(y)/np.sum(~y))
    return np.sum((f>=0)==y)/y.shape[0]

@njit(error_model="numpy")#,fastmath=True)
def Hull_1d(x,y):#x.shape=(N),y=ans
    TF=np.empty(x.shape[0],dtype="bool")
    TF[y]=np.min(x[~y])>x[y]
    TF[y]|=x[y]>np.max(x[~y])
    TF[~y]=np.min(x[y])>x[~y]
    TF[~y]|=x[~y]>np.max(x[y])
    return np.mean(TF)

@njit(error_model="numpy")#,fastmath=True)
def QDA_1d(x,y):#x.shape=(N),y=ans
    mu_T,mu_F=np.mean(x[y]),np.mean(x[~y])
    pi_T,pi_F=np.sum(y),np.sum(~y)
    sigma_T=np.sum((x[y]-mu_T)**2)/(pi_T-1)
    sigma_F=np.sum((x[~y]-mu_F)**2)/(pi_F-1)
    value=-((x-mu_T)**2/sigma_T)+(((x-mu_F))**2/sigma_F)
    value+=(2*np.log(pi_T/pi_F)-np.log(np.abs(sigma_T/sigma_F)))
    return np.sum((value>0)==y)/x.shape[0]

@njit(error_model="numpy")
def DT_1d(x,y):#x.shape=(N),y=ans
    sort_index=np.argsort(x)
    sorted_y=y[sort_index].astype("int64")*2-1
    TF=np.empty(x.shape[0],dtype="bool")
    TF[0]=True
    TF[1:]=(x[sort_index[1:]]!=x[sort_index[:-1]])
    if np.all(TF):
        cumsum=np.cumsum(sorted_y)
    else:
        for i in np.arange(y.shape[0])[~TF][::-1]:
            sorted_y[i-1]+=sorted_y[i]
        cumsum=np.cumsum(sorted_y[TF])
    return max(np.sum(~y)+np.max(cumsum),np.sum(y)-np.min(cumsum))/y.shape[0]
