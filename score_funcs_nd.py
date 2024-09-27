def QDA_nd(X,y):#X.shape=(n_d,n_samples)
    mu_T,mu_F=np.mean(X[:,y],axis=1),np.mean(X[:,~y],axis=1)
    pi_T,pi_F=np.sum(y),np.sum(~y)
    sigma_T=np.cov(X[:,y],ddof=1)
    sigma_F=np.cov(X[:,~y],ddof=1)
    if X.shape[0]==1:
        sigma_T=np.array([[sigma_T]])
        sigma_F=np.array([[sigma_F]])
    value=-((X.T-mu_T)@np.linalg.inv(sigma_T)@(X.T-mu_T).T).diagonal()
    value+=((X.T-mu_F)@np.linalg.inv(sigma_F)@(X.T-mu_F).T).diagonal()
    value+=(2*np.log(pi_T/pi_F)-np.log(np.linalg.det(sigma_T)/np.linalg.det(sigma_F)))
    return np.sum((value>0)==y)/y.shape[0]