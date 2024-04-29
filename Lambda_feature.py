import numpy as np
from Lambda_map import *
from joblib import Parallel, delayed
# from sklearn.manifold import Isomap
import warnings

warnings.filterwarnings('ignore')

def normalization_partition(data,eta):
    a = data.T
    psi = a.shape[0]
    a = np.expand_dims(a,0).repeat(psi,axis=0)
    tmp_2 = data.T.reshape((psi,1,data.shape[0]))
    M = a-tmp_2
    m = 1/np.sqrt(np.sum(np.exp(-2*eta*M),axis=0))
    m = m.T
    return m

def normalization_vect(vect,eta,psi,t):
    assert vect.shape[1] == (psi*t)
    data_lst =  Parallel(n_jobs=-1)(delayed(normalization_partition)(vect[:,int(i*psi):int((i+1)*psi)],eta) for i in range(t))
    feature_map = np.concatenate(data_lst,axis=1)
    assert feature_map.shape==vect.shape,print(feature_map.shape,vect.shape)
    assert np.all(feature_map<=1)
    assert np.all(feature_map>=0)
    return feature_map/np.sqrt(t)

# def Iso_dm(X,k):
#     embedding = Isomap(n_neighbors=k)
#     embedding.fit(X)
#     return embedding.dist_matrix_

def lambda_feature_infty(distribution,newdata,psi,t=100,P=None,alpha=None):
    # produce feature and distance matrix for X and query_points
    lm = Lambda_map(psi,t)
    lm.fit(distribution,P,alpha)
    dis_map = lm.transform(distribution).toarray()
    new_map = lm.transform(newdata).toarray()
    dis_map = dis_map/np.sqrt(t)
    new_map = new_map/np.sqrt(t)
    dm = 1-np.dot(dis_map,dis_map.T)
    dm[np.where(dm<0)]=0
    dm2 = 1-np.dot(new_map,dis_map.T)
    dm2[np.where(dm2<0)]=0
    return dis_map,new_map,dm,dm2

def lambda_feature_continous(distribution,newdata,eta,psi,t=100,P=None,alpha=None):
    # produce feature and distance matrix for X and query_points
    lm = Lambda_map(psi,t)
    lm.fit(distribution,P,alpha)
    dis_map = lm.transform_continous(distribution).toarray()
    dis_map = normalization_vect(dis_map,eta,psi,t)
    new_map = lm.transform_continous(newdata).toarray()
    new_map = normalization_vect(new_map,eta,psi,t)
    dm = 1-np.dot(dis_map,dis_map.T)
    dm[np.where(dm<0)]=0
    dm2 = 1-np.dot(new_map,dis_map.T)
    dm2[np.where(dm2<0)]=0
    return dis_map,new_map,dm,dm2

def lambda_feature_infty_dm(dm,psi,t=100,P=None,alpha=None):
    lm = Lambda_map(psi,t)
    lm.fit_dm(dm,P,alpha)
    dis_map = lm.transform_dm().toarray()
    print(np.unique(dis_map))
    dis_map = dis_map/np.sqrt(t)
    dm = 1-np.dot(dis_map,dis_map.T)
    dm[np.where(dm<0)]=0
    return dis_map,dm

def lambda_feature_continous_dm(dm,eta,psi,t=100,P=None,alpha=None):
    lm = Lambda_map(psi,t)
    lm.fit_dm(dm,P,alpha)
    dis_map = lm.transform_continous_dm().toarray()
    dis_map = normalization_vect(dis_map,eta,psi,t)
    dm = 1-np.dot(dis_map,dis_map.T)
    dm[np.where(dm<0)]=0
    return dis_map,dm

# def lambda_feature_infty_iso(distribution,k,psi,t=100):
#     dm = Iso_dm(distribution,k)
#     return lambda_feature_infty_dm(dm,psi,t)

# def lambda_feature_continous_iso(distribution,k,eta,psi,t=100):
#     dm = Iso_dm(distribution,k)
#     return lambda_feature_continous_dm(dm,eta,psi,t)

# from sklearn.metrics.pairwise import cosine_similarity as coss

# def d_eta(X,eta,psi,t=100):
#     lm = Lambda_map(psi,t)
#     lm.fit(X)
#     dis_map = lm.transform_continous(X).toarray()
#     l_map = np.exp(-1*eta*dis_map)
#     n,_ = X.shape
#     # Lambda-kernel
#     l_kernel = np.zeros((n,n))
#     for i in range(t):
#         l_kernel = l_kernel + coss(l_map[:,int(i*psi):int((i+1)*psi)],l_map[:,int(i*psi):int((i+1)*psi)])
#     l_kernel = l_kernel/t
#     # Lambda-kernel's Derivate of eta
#     de = np.zeros((n,n))
#     for i in range(t):
#         tmp_dis_map = l_map[:,int(i*psi):int((i+1)*psi)]
#         dis_norm = np.linalg.norm(tmp_dis_map,axis=1)
#         dis_norm = np.reshape(dis_norm,(-1,1))

#         tmp_dis_map_norm_weighted = tmp_dis_map/dis_norm.repeat(psi,axis=1)
#         tmp_dis_map_x_weighted = -1*dis_map[:,int(i*psi):int((i+1)*psi)]*tmp_dis_map/dis_norm.repeat(psi,axis=1)

#         p1 = np.dot(tmp_dis_map_x_weighted,tmp_dis_map_norm_weighted.T)
#         p3 = np.linalg.norm(tmp_dis_map_norm_weighted*tmp_dis_map_norm_weighted*dis_map[:,int(i*psi):int((i+1)*psi)],axis=1,ord=1)
#         p3 = np.reshape(p3,(-1,1)).repeat(n,axis=1)
        
#         tmp_de = p1+p1.T+(p3+p3.T)*coss(l_map[:,int(i*psi):int((i+1)*psi)],l_map[:,int(i*psi):int((i+1)*psi)])
#         de = de+tmp_de
#     de = de/t
#     return l_kernel,de

# def remove_0(X):
#     x = np.min(X[np.where(X>0)])
#     X[np.where(X==0)] = x/100
#     return X