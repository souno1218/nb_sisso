import datetime
import numpy as np
from numba_progress import ProgressBar
from numba import njit,prange,set_num_threads,get_thread_id
#https://qiita.com/m1t0/items/06f2d07e626d1c4733fd

@njit(error_model="numpy",fastmath=True)
def jit_cov(X,ddof=1):
    n=X.shape[1]
    X_1=X[0,:]-np.sum(X[0,:])/n
    X_2=X[1,:]-np.sum(X[1,:])/n
    var_1=np.sum(X_1*X_1)/(n-ddof)#不偏分散、sklearnに準拠
    var_2=np.sum(X_2*X_2)/(n-ddof)
    cross_cov=np.sum(X_1*X_2)/(n-ddof)
    return var_1,var_2,cross_cov

@njit(error_model="numpy",fastmath=True)
def QDA_2d(X,Y):#X.shape=(2,-1)
    class1_X=X[:,Y]
    class2_X=X[:,~Y]
    class1_var_1,class1_var_2,class1_cov=jit_cov(class1_X)
    class2_var_1,class2_var_2,class2_cov=jit_cov(class2_X)
    det_cov1=class1_var_1*class1_var_2-class1_cov**2
    det_cov2=class2_var_1*class2_var_2-class2_cov**2    

    target_x_1=(X.T-np.array([np.mean(class1_X[0,:]),np.mean(class1_X[1,:])])).T
    target_x_2=(X.T-np.array([np.mean(class2_X[0,:]),np.mean(class2_X[1,:])])).T
    value1=class1_var_2*target_x_1[0,:]**2+class1_var_1*target_x_1[1,:]**2-2*class1_cov*target_x_1[0,:]*target_x_1[1,:]
    value1/=det_cov1
    value2=class2_var_2*target_x_2[0,:]**2+class2_var_1*target_x_2[1,:]**2-2*class2_cov*target_x_2[0,:]*target_x_2[1,:]
    value2/=det_cov2
    value3=2*(np.log(np.sum(Y))-np.log(np.sum(~Y)))-np.log(det_cov1)+np.log(det_cov2)
    value=(-value1+value2+value3)
    return np.sum((value>0)==Y)/X.shape[1]

@njit(error_model="numpy",fastmath=True)
def LDA_2d(X,Y):#xy.shape=(2,-1)
    pi_0=np.sum(Y)
    pi_1=np.sum(~Y)
    class0_var_0,class0_var_1,class0_cov=jit_cov(X[:,Y],ddof=0)
    class1_var_0,class1_var_1,class1_cov=jit_cov(X[:,~Y],ddof=0)
    var_0=(pi_0*class0_var_0+pi_1*class1_var_0)/(pi_0+pi_1)
    var_1=(pi_0*class0_var_1+pi_1*class1_var_1)/(pi_0+pi_1)
    cov=(pi_0*class0_cov+pi_1*class1_cov)/(pi_0+pi_1)
    mean_0_0=np.mean(X[0,Y])
    mean_0_1=np.mean(X[1,Y])
    mean_1_0=np.mean(X[0,~Y])
    mean_1_1=np.mean(X[1,~Y])
    value=((mean_0_0-mean_1_0)*var_1-(mean_0_1-mean_1_1)*cov)*X[0]
    value+=((mean_0_1-mean_1_1)*var_0-(mean_0_0-mean_1_0)*cov)*X[1]
    value+=(mean_1_0**2-mean_0_0**2)*var_1/2
    value+=(mean_1_1**2-mean_0_1**2)*var_0/2
    value+=(mean_0_0*mean_0_1-mean_1_0*mean_1_1)*cov
    value-=(var_0*var_1-cov**2)*(np.log(pi_1/pi_0))
    return np.sum((value>0)==Y)/X.shape[1]

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

@njit(error_model="numpy")
def DT_1d_with_border(x,y):#x.shape=(N),y=ans
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
        score=score_FT/y.shape[0]
    else:
        index=np.argmin(cumsum)
        score=score_TF/y.shape[0]
    border=(sort_x[index]+sort_x[index+1])/2
    return score,border

@njit(error_model="numpy")
def DT_2d(X,Y):#xy.shape=(2,-1)
    score0,border0=DT_1d_with_border(X[0],Y)
    score1,border1=DT_1d_with_border(X[1],Y)
    arange=np.arange(Y.shape[0])
    score=0
    if score0>score1:
        split_dim,score_dim,border=0,1,border0
    else:
        split_dim,score_dim,border=1,0,border1
    index_0=arange[X[split_dim]>border]
    index_1=arange[X[split_dim]<=border]
    weight0=index_0.shape[0]/Y.shape[0]
    weight1=index_1.shape[0]/Y.shape[0]
    if weight0!=0:
        score+=weight0*DT_1d(X[score_dim][index_0],Y[index_0])
    if weight1!=0:
        score+=weight1*DT_1d(X[score_dim][index_1],Y[index_1])
    return score

@njit(error_model="numpy")
def sub_Hull_2d(base_x,other_x,not_is_in,arange,base_index1,base_index2,base_x_mask):
    if np.any(not_is_in):
        copy_base_x_mask=base_x_mask.copy()
        base_vec=base_x[:,base_index1]-base_x[:,base_index2]
        bec_xy=base_x[:,arange[copy_base_x_mask]]-np.expand_dims(base_x[:,base_index2], axis=1)
        cross=base_vec[0]*bec_xy[1]-base_vec[1]*bec_xy[0]
        if np.any(cross>0):
            next_point=np.argmax(cross)
            next_index=arange[copy_base_x_mask][next_point]
            copy_base_x_mask[arange[copy_base_x_mask][cross<=0]]=False
            copy_base_x_mask[next_index]=False
            use_other_x=other_x[:,not_is_in]-np.expand_dims(base_x[:,base_index2], axis=1)
            num_is_in=np.empty((2,np.sum(not_is_in)),dtype="float")
            num_is_in[0]=base_vec[1]*use_other_x[0]-base_vec[0]*use_other_x[1]
            num_is_in[1]=-bec_xy[1,next_point]*use_other_x[0]+bec_xy[0,next_point]*use_other_x[1]
            num_is_in/=(bec_xy[0,next_point]*base_vec[1]-base_vec[0]*bec_xy[1,next_point])
            not_is_in[not_is_in]=(((num_is_in[0]+num_is_in[1])>1)|(0>num_is_in[0])|(0>num_is_in[1]))
            sub_Hull_2d(base_x,other_x,not_is_in,arange,next_index,base_index2,copy_base_x_mask)
            sub_Hull_2d(base_x,other_x,not_is_in,arange,base_index1,next_index,copy_base_x_mask)

@njit(error_model="numpy")
def Hull_2d(X,Y):#xy.shape=(2,-1)
    index0=np.argsort(X[0])
    index1=np.argsort(X[1])
    if (np.sum(index0==index1)/X.shape[1])>0.9:
        return 0
    if (np.sum(index0==index1[::-1])/X.shape[1])>0.9:
        return 0
    class1_X,class2_X=X[:,Y],X[:,~Y]
    index_x_max=np.argmax(class1_X[0])
    index_x_min=np.argmin(class1_X[0])
    arange=np.arange(class1_X.shape[1])
    class1_X_mask=np.ones(class1_X.shape[1],dtype="bool")
    not_is_in=np.ones(class2_X.shape[1],dtype="bool")
    class1_X_mask[index_x_max]=False
    class1_X_mask[index_x_min]=False
    copy_class1_X_mask=class1_X_mask.copy()
    sub_Hull_2d(class1_X,class2_X,not_is_in,arange,index_x_max,index_x_min,copy_class1_X_mask)
    sub_Hull_2d(class1_X,class2_X,not_is_in,arange,index_x_min,index_x_max,class1_X_mask)
    ans=np.sum(~not_is_in)
    index_x_max=np.argmax(class2_X[0])
    index_x_min=np.argmin(class2_X[0])
    arange=np.arange(class2_X.shape[1])
    class2_X_mask=np.ones(class2_X.shape[1],dtype="bool")
    not_is_in=np.ones(class2_X.shape[1],dtype="bool")
    class2_X_mask[index_x_max]=False
    class2_X_mask[index_x_min]=False
    copy_class2_X_mask=class2_X_mask.copy()
    sub_Hull_2d(class2_X,class1_X,not_is_in,arange,index_x_max,index_x_min,copy_class2_X_mask)
    sub_Hull_2d(class2_X,class1_X,not_is_in,arange,index_x_min,index_x_max,class2_X_mask)
    ans+=np.sum(~not_is_in)
    return 1-(ans/(Y.shape[0]))