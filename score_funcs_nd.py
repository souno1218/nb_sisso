import numpy as np
from utils import jit_cov
from numba import njit

#めっちゃ遅い

@njit(error_model="numpy",fastmath=True)
def QDA_nd(X,y):
    pi_T,pi_F=np.sum(y),np.sum(~y)
    mu_T=np.expand_dims(np.sum(X[:,y],axis=1)/pi_T,1)
    mu_F=np.expand_dims(np.sum(X[:,~y],axis=1)/pi_F,1)
    if X.shape[0]==1:
        sigma_T=np.array([[np.var(X[:,y])]])*pi_T/(pi_T-1)
        sigma_F=np.array([[np.var(X[:,~y])]])*pi_F/(pi_F-1)
    else:
        sigma_T=np.cov(X[:,y],ddof=1)
        sigma_F=np.cov(X[:,~y],ddof=1)
    value=-np.diag(((X-mu_T).T@np.linalg.pinv(sigma_T)@(X-mu_T)))
    value+=np.diag(((X-mu_F).T@np.linalg.pinv(sigma_F)@(X-mu_F)))
    value+=(2*np.log(pi_T/pi_F)-np.log(np.linalg.det(sigma_T)/np.linalg.det(sigma_F)))
    return np.sum((value>0)==y)/y.shape[0],0