from numba import njit,int64,float64
import numpy as np
from numba.types import string
from numba.types import unicode_type

#If error_model is available for jitclass, replace all of them with class


### debug
@njit(error_model="numpy")#,fastmath=True)
def debug_1d(x,y):
    args=sub_debug_1d_fit(x,y)
    return sub_debug_1d_score(x,y,*args)

@njit(error_model="numpy")#,fastmath=True)
def sub_debug_1d_fit(x,y):
    mu=np.mean(x)
    sigma=np.sqrt(np.sum((x-mu)**2))
    return mu,sigma

@njit(error_model="numpy")#,fastmath=True)
def sub_debug_1d_score(x,y,mu,sigma):
    return np.abs((x[0]-mu)/sigma),0


### LDA
@njit(error_model="numpy")#,fastmath=True)
def LDA_1d(x,y):
    args=sub_LDA_1d_fit(x,y)
    return sub_LDA_1d_score(x,y,*args)

@njit(error_model="numpy")
def sub_LDA_1d_fit(x,y):
    pi_T=np.sum(y)
    pi_F=np.sum(~y)
    mu_T=np.mean(x[y])
    mu_F=np.mean(x[~y])
    sigma=(np.sum((x[y]-mu_T)**2)+np.sum((x[~y]-mu_F)**2))/y.shape[0]
    return pi_T,pi_F,mu_T,mu_F,sigma

@njit(error_model="numpy")
def sub_LDA_1d_score(x,y,pi_T,pi_F,mu_T,mu_F,sigma):
    f=(mu_T-mu_F)*x-(mu_T-mu_F)*(mu_T+mu_F)/2+sigma*np.log(pi_T/pi_F)
    return np.sum((f>=0)==y)/y.shape[0],0


### QDA
@njit(error_model="numpy")
def QDA_1d(x,y):
    args=sub_QDA_1d_fit(x,y)
    return sub_QDA_1d_score(x,y,*args)

@njit(error_model="numpy")
def sub_QDA_1d_fit(x,y):
    mu_T,mu_F=np.mean(x[y]),np.mean(x[~y])
    pi_T,pi_F=np.sum(y),np.sum(~y)
    sigma_T=np.sum((x[y]-mu_T)**2)/(pi_T-1)
    sigma_F=np.sum((x[~y]-mu_F)**2)/(pi_F-1)        
    value2=(2*np.log(pi_T/pi_F)-np.log(np.abs(sigma_T/sigma_F)))
    return mu_T,mu_F,sigma_T,sigma_F,value2

@njit(error_model="numpy")
def sub_QDA_1d_score(x,y,mu_T,mu_F,sigma_T,sigma_F,value2):
    value=-((x-mu_T)**2/sigma_T)+(((x-mu_F))**2/sigma_F)+value2
    return np.sum((value>0)==y)/y.shape[0],0

### Hull
@njit(error_model="numpy")#,fastmath=True)
def Hull_1d(x,y):
    args=sub_Hull_1d_fit(x,y)
    return sub_Hull_1d_score(x,y,*args)

@njit(error_model="numpy")#,fastmath=True)
def sub_Hull_1d_fit(x,y):
    min_T,max_T=np.min(x[y]),np.max(x[y])
    min_F,max_F=np.min(x[~y]),np.max(x[~y])
    return min_T,max_T,min_F,max_F

@njit(error_model="numpy")#,fastmath=True)
def sub_Hull_1d_score(x,y,min_T,max_T,min_F,max_F):
    TF=np.empty(x.shape[0],dtype="bool")
    TF[y]=min_F>x[y]
    TF[y]|=x[y]>max_F
    TF[~y]=min_T>x[~y]
    TF[~y]|=x[~y]>max_T
    return np.mean(TF),0

### DT
@njit(error_model="numpy")
def DT_1d(x,y):
    args=sub_DT_1d_fit(x,y)
    return sub_DT_1d_score(x,y,*args)

@njit(error_model="numpy")
def sub_DT_1d_fit(x,y):
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
    score_FT=np.sum(~y)+np.max(cumsum)
    score_TF=np.sum(y)-np.min(cumsum)
    sort_x=x[sort_index[TF]]
    if score_FT>score_TF:
        index=np.argmax(cumsum)
        first_area=True
    else:
        index=np.argmin(cumsum)
        first_area=False
    border=(sort_x[index]+sort_x[index+1])/2
    return border,first_area

@njit(error_model="numpy")
def sub_DT_1d_score(x,y,border,first_area):
    return (np.sum(x[y==first_area]<=border)+np.sum(x[y!=first_area]>border))/y.shape[0],0

### make_CV_model
def make_CV_model(sub_func_fit,sub_func_score,k=5,name=None):
    if k<=1:
        raise 
    @njit(error_model="numpy")
    def CrossValidation(x,y):
        index=np.arange(y.shape[0])
        np.random.shuffle(index)
        TF=np.ones(y.shape[0],dtype="bool")
        sum_score1,sum_score2=0,0
        for i in range(k):
            TF[index[i::k]]=False
            args=sub_func_fit(x[TF],y[TF])
            score1,score2=sub_func_score(x[~TF],y[~TF],*args)
            sum_score1+=score1
            sum_score2+=score2
            TF[index[i::k]]=True
        return sum_score1/k,sum_score2/k
    model=CrossValidation
    if not name is None:
        if not isinstance(name,str):
            raise TypeError(f"Expected variable 'name' to be of type str, but got {type(name)}.")
        else:
            model.__name__=name
    return model