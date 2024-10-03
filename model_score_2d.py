import numpy as np
from utils import jit_cov
from numba import njit
#https://qiita.com/m1t0/items/06f2d07e626d1c4733fd

### QDA_2d
@njit(error_model="numpy",fastmath=True)
def QDA_2d(X,Y):
    args=sub_QDA_2d_fit(X,Y)
    return sub_QDA_2d_score(X,Y,*args)

@njit(error_model="numpy",fastmath=True)
def sub_QDA_2d_fit(X,Y):
    classT_X=X[:,Y]
    classF_X=X[:,~Y]
    pi_T,pi_F=classT_X.shape[1]+1e-300,classF_X.shape[1]+1e-300
    classT_var_1,classT_var_2,classT_cov=jit_cov(classT_X)
    det_covT=classT_var_1*classT_var_2-classT_cov**2+1e-300
    
    classF_var_1,classF_var_2,classF_cov=jit_cov(classF_X)
    det_covF=classF_var_1*classF_var_2-classF_cov**2+1e-300
    
    mean_T=np.array([np.mean(classT_X[0,:]),np.mean(classT_X[1,:])])
    mean_F=np.array([np.mean(classF_X[0,:]),np.mean(classF_X[1,:])])
    return pi_T,pi_F,classT_var_1,classT_var_2,classT_cov,classF_var_1,classF_var_2,classF_cov,mean_T,mean_F,det_covT,det_covF
    
@njit(error_model="numpy",fastmath=True)
def sub_QDA_2d_score(X,Y,pi_T,pi_F,classT_var_1,classT_var_2,classT_cov,classF_var_1,classF_var_2,classF_cov,mean_T,mean_F,det_covT,det_covF):
    target_x_1=(X.T-mean_T).T
    target_x_2=(X.T-mean_F).T
    value1=classT_var_2*target_x_1[0,:]**2+classT_var_1*target_x_1[1,:]**2-2*classT_cov*target_x_1[0,:]*target_x_1[1,:]
    value1/=det_covT
    value2=classF_var_2*target_x_2[0,:]**2+classF_var_1*target_x_2[1,:]**2-2*classF_cov*target_x_2[0,:]*target_x_2[1,:]
    value2/=det_covF
    value3=2*np.log(pi_T/pi_F)-np.log(np.abs(det_covT/det_covF)+1e-300)
    value=(-value1+value2+value3)
    score=np.sum((value>0)==Y)/X.shape[1]

    #Kullback-Leibler Divergence , https://sucrose.hatenablog.com/entry/2013/07/20/190146
    kl_d=np.log(np.abs(det_covF/det_covT+1e-300))/2-1
    kl_d+=(classF_var_2*classT_var_1+classF_var_1*classT_var_2-2*classT_cov*classF_cov)/det_covF/2
    dmean=mean_T-mean_F
    kl_d+=(dmean[0]**2*classF_var_2+dmean[1]**2*classF_var_1-2*dmean[0]*dmean[1]*classF_cov)/det_covF/2
    return np.sum((value>0)==Y)/X.shape[1],kl_d

### LDA_2d
@njit(error_model="numpy",fastmath=True)
def LDA_2d(X,Y):
    args=sub_LDA_2d_fit(X,Y)
    return sub_LDA_2d_score(X,Y,*args)
    
@njit(error_model="numpy",fastmath=True)
def sub_LDA_2d_fit(X,Y):
    pi_T=np.sum(Y)+1e-300
    pi_F=np.sum(~Y)+1e-300
    classT_var_0,classT_var_1,classT_cov=jit_cov(X[:,Y],ddof=0)
    classF_var_0,classF_var_1,classF_cov=jit_cov(X[:,~Y],ddof=0)
    var_0=(pi_T*classT_var_0+pi_F*classF_var_0)/(pi_T+pi_F)
    var_1=(pi_T*classT_var_1+pi_F*classF_var_1)/(pi_T+pi_F)
    cov=(pi_T*classT_cov+pi_F*classF_cov)/(pi_T+pi_F)
    mean_T_0,mean_T_1=np.mean(X[0,Y]),np.mean(X[1,Y])
    mean_F_0,mean_F_1=np.mean(X[0,~Y]),np.mean(X[1,~Y])
    dmean_0,dmean_1=mean_T_0-mean_F_0,mean_T_1-mean_F_1
    c=(mean_F_0**2-mean_T_0**2)*var_1/2+(mean_F_1**2-mean_T_1**2)*var_0/2
    c+=(mean_T_0*mean_T_1-mean_F_0*mean_F_1)*cov
    c-=(var_0*var_1-cov**2)*(np.log(pi_F/pi_T))
    return var_0,var_1,cov,dmean_0,dmean_1,c

@njit(error_model="numpy",fastmath=True)
def sub_LDA_2d_score(X,Y,var_0,var_1,cov,dmean_0,dmean_1,c):
    a=(dmean_0*var_1-dmean_1*cov)
    b=(dmean_1*var_0-dmean_0*cov)    
    score=np.sum(((a*X[0]+b*X[1]+c)>0)==Y)/X.shape[1]
    #Kullback-Leibler Divergence , https://sucrose.hatenablog.com/entry/2013/07/20/190146
    kl_d=(dmean_0**2*var_1+dmean_1**2*var_0-2*dmean_0*dmean_1*cov)/(var_0*var_1-cov**2+1e-300)/2
    return score,kl_d

### DT_2d
from model_score_1d import DT_1d,sub_DT_1d_fit,sub_DT_1d_score

@njit(error_model="numpy")
def DT_2d(X,Y):
    args=sub_DT_2d_fit(X,Y)
    return sub_DT_2d_score(X,Y,*args)
    
@njit(error_model="numpy")
def sub_DT_2d_fit(x,y):
    #root node
    border0,first_area_predict0=sub_DT_1d_fit(x[0],y)
    score0,minus_entropy0=sub_DT_1d_score(x[0],y,border0,first_area_predict0)
    border1,first_area_predict1=sub_DT_1d_fit(x[1],y)
    score1,minus_entropy1=sub_DT_1d_score(x[1],y,border1,first_area_predict1)
    if score0>score1:
        root_node=0
        root_border=border0
    elif score0==score1:
        if minus_entropy0>minus_entropy1:
            root_node=0
            root_border=border0
        else:
            root_node=1
            root_border=border1
    else:
        root_node=1
        root_border=border1
    area_0_index=x[root_node]<root_border
    area_1_index=x[root_node]>=root_border
    
    #leaf node  (area_0)
    border0,area_0_predict0=sub_DT_1d_fit(x[0,area_0_index],y[area_0_index])
    score0,minus_entropy0=sub_DT_1d_score(x[0,area_0_index],y[area_0_index],border0,first_area_predict0)
    border1,area_0_predict1=sub_DT_1d_fit(x[1,area_0_index],y[area_0_index])
    score1,minus_entropy1=sub_DT_1d_score(x[1,area_0_index],y[area_0_index],border1,first_area_predict1)
    if score0>score1:
        leaf_0_node=0
        leaf_0_border=border0
        leaf_0_area_0_predict=area_0_predict0
    elif score0==score1:
        if minus_entropy0>minus_entropy1:
            leaf_0_node=0
            leaf_0_border=border0
            leaf_0_area_0_predict=area_0_predict0
        else:
            leaf_0_node=1
            leaf_0_border=border1
            leaf_0_area_0_predict=area_0_predict1
    else:
        leaf_0_node=1
        leaf_0_border=border1
        leaf_0_area_0_predict=area_0_predict1
        
    #leaf node  (area_1)
    border0,area_0_predict0=sub_DT_1d_fit(x[0,area_1_index],y[area_1_index])
    score0,minus_entropy0=sub_DT_1d_score(x[0,area_1_index],y[area_1_index],border0,first_area_predict0)
    border1,area_0_predict1=sub_DT_1d_fit(x[1,area_1_index],y[area_1_index])
    score1,minus_entropy1=sub_DT_1d_score(x[1,area_1_index],y[area_1_index],border1,first_area_predict1)
    if score0>score1:
        leaf_1_node=0
        leaf_1_border=border0
        leaf_1_area_0_predict=area_0_predict0
    elif score0==score1:
        if minus_entropy0>minus_entropy1:
            leaf_1_node=0
            leaf_1_border=border0
            leaf_1_area_0_predict=area_0_predict0
        else:
            leaf_1_node=1
            leaf_1_border=border1
            leaf_1_area_0_predict=area_0_predict1
    else:
        leaf_1_node=1
        leaf_1_border=border1
        leaf_1_area_0_predict=area_0_predict1
    return root_node,root_border,leaf_0_node,leaf_0_border,leaf_0_area_0_predict,leaf_1_node,leaf_1_border,leaf_1_area_0_predict

@njit(error_model="numpy")
def sub_DT_2d_score(x,y,root_node,root_border,leaf_0_node,leaf_0_border,leaf_0_area_0_predict,leaf_1_node,leaf_1_border,leaf_1_area_0_predict):
    area_0_index=x[root_node]<root_border
    n_area_0=np.sum(area_0_index)
    area_1_index=x[root_node]>=root_border
    n_area_1=np.sum(area_1_index)

    score0,minus_entropy0=sub_DT_1d_score(x[leaf_0_node,area_0_index],y[area_0_index],leaf_0_border,leaf_0_area_0_predict)
    score1,minus_entropy1=sub_DT_1d_score(x[leaf_1_node,area_1_index],y[area_1_index],leaf_1_border,leaf_1_area_0_predict)
    
    score=(n_area_0*score0+n_area_1*score1)/(n_area_0+n_area_1)
    minus_entropy=(n_area_0*minus_entropy0+n_area_1*minus_entropy1)/(n_area_0+n_area_1)
    return score,minus_entropy


### Hull_2d
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
    #Spearman rank correlation coefficient
    index0=np.argsort(np.argsort(X[0]))
    index1=np.argsort(np.argsort(X[1]))
    n=index0.shape[0]
    r_R=np.abs(1-6*np.sum((index0-index1)**2)/(n*(n**2-1)))
    if r_R>0.9:
        return 0,0
    classT_X,classF_X=X[:,Y],X[:,~Y]
    index_x_max=np.argmax(classT_X[0])
    index_x_min=np.argmin(classT_X[0])
    arange=np.arange(classT_X.shape[1])
    classT_X_mask=np.ones(classT_X.shape[1],dtype="bool")
    not_is_in=np.ones(classF_X.shape[1],dtype="bool")
    classT_X_mask[index_x_max]=False
    classT_X_mask[index_x_min]=False
    copy_classT_X_mask=classT_X_mask.copy()
    sub_Hull_2d(classT_X,classF_X,not_is_in,arange,index_x_max,index_x_min,copy_classT_X_mask)
    sub_Hull_2d(classT_X,classF_X,not_is_in,arange,index_x_min,index_x_max,classT_X_mask)
    ans=np.sum(~not_is_in)
    index_x_max=np.argmax(classF_X[0])
    index_x_min=np.argmin(classF_X[0])
    arange=np.arange(classF_X.shape[1])
    classF_X_mask=np.ones(classF_X.shape[1],dtype="bool")
    not_is_in=np.ones(classT_X.shape[1],dtype="bool")
    classF_X_mask[index_x_max]=False
    classF_X_mask[index_x_min]=False
    copy_classF_X_mask=classF_X_mask.copy()
    sub_Hull_2d(classF_X,classT_X,not_is_in,arange,index_x_max,index_x_min,copy_classF_X_mask)
    sub_Hull_2d(classF_X,classT_X,not_is_in,arange,index_x_min,index_x_max,classF_X_mask)
    ans+=np.sum(~not_is_in)
    return 1-(ans/(Y.shape[0])),0