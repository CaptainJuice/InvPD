import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

class Lambda_map:
    data = None
    centroid = []
    def __init__(self, psi, t):
        self.psi = psi
        self.t = t
        self.iso = 0
        self.dm = None

    def fit(self, data,P=None,alpha=None):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]
        self.P = P
        # assert P.shape[0] == sn,'wtf'
        for i in range(self.t):
            if P is None:
                subIndex = sample(range(sn), self.psi)
            else:
                cu_lst = np.unique(P)
                c_num = int(cu_lst.shape[0])
                r1_lst = np.array([np.sum(P==cu)/P.shape[0] for cu in cu_lst])
                r2_lst = np.array([1/c_num]*c_num)
                # print(r1_lst,r2_lst)
                assert r1_lst.shape[0]==r2_lst.shape[0],'ratio match error'
                r_lst = alpha*r1_lst+(1-alpha)*r2_lst
                psi_lst = self.psi*r_lst
                psi_lst = np.array([int(psi) for psi in psi_lst])
                psi_lst[-1] = int(self.psi-np.sum(psi_lst[:-1]))
                assert np.sum(psi_lst)==self.psi,'psi error'
                subIndex = list()
                for idx,cu in enumerate(cu_lst):
                    tmp_idx = sample(list(np.where(P==cu)[0]),psi_lst[idx])
                    subIndex.extend(tmp_idx)

            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]
            tt_dis = cdist(tdata, tdata)
            radius = [] #restore centroids' radius
            for r_idx in range(self.psi):
                r = tt_dis[r_idx]
                r[r<0] = 0
                r = np.delete(r,r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)


    def transform(self, newdata):
            # Lambda=infty, euclidean distance
            assert self.centroid != None, "invoke fit() first!"
            n, _ = newdata.shape
            IDX = np.array([])
            V = []
            for i in range(self.t):
                subIndex = self.centroid[i]
                radius = self.centroids_radius[i]
                tdata = self.data[subIndex, :]
                dis = cdist(tdata, newdata) #-------------------------
                centerIdx = np.argmin(dis, axis=0)
                for j in range(n):
                    # HyperSphere
                    #V.append(int(dis[centerIdx[j], j] <= radius[centerIdx[j]]))
                    # Voronoi diagram
                    V.append(1) 
                IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
            IDR = np.tile(range(n), self.t)
            ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
            return ndata

    def transform_continous(self, newdata):
        # Lambda<infty, euclidean distance
        assert self.centroid != None, "invoke fit() first!"
        n, _ = newdata.shape
        IDX = np.array([])
        V = []
        for i in range(self.t):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            dis = cdist(tdata, newdata)
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                # continous case
                # V.append(np.exp(-1*eta*dis[centerIdx[j], j]))
                V.append(dis[centerIdx[j], j])
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata
    
    def fit_dm(self,dm,P=None,alpha=None):
        self.dm = dm #
        self.centroid = []
        self.centroids_radius = []
        self.iso = 1 #
        sn = self.dm.shape[0]
        self.P = P
        # assert P.shape[0] == sn,'wtf'
        for i in range(self.t):
            if P is None:
                subIndex = sample(range(sn), self.psi)
            else:
                cu_lst = np.unique(P)
                c_num = int(cu_lst.shape[0])
                r1_lst = np.array([np.sum(P==cu)/P.shape[0] for cu in cu_lst])
                r2_lst = np.array([1/c_num]*c_num)
                # print(r1_lst,r2_lst)
                assert r1_lst.shape[0]==r2_lst.shape[0],'ratio match error'
                r_lst = alpha*r1_lst+(1-alpha)*r2_lst
                psi_lst = self.psi*r_lst
                psi_lst = np.array([int(psi) for psi in psi_lst])
                psi_lst[-1] = int(self.psi-np.sum(psi_lst[:-1]))
                assert np.sum(psi_lst)==self.psi,'psi error'
                subIndex = list()
                # print(psi_lst)
                for idx,cu in enumerate(cu_lst):
                    tmp_idx = sample(list(np.where(P==cu)[0]),psi_lst[idx])
                    subIndex.extend(tmp_idx)

            self.centroid.append(subIndex)
            tt_dis = self.dm[subIndex,:][:,subIndex]
            radius = [] #restore centroids' radius
            for r_idx in range(self.psi):
                r = tt_dis[r_idx]
                r[r<0] = 0
                r = np.delete(r,r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)
        # self.psi = self.psi*int(np.unique(P).shape[0])

    def transform_dm(self, newdata=None): 
        # Lambda=infty, geodesic distance
        assert self.centroid != None, "invoke fit_iso() first!"
        #if self.iso == 1:
        #    assert np.all(newdata==self.data),'Match error!'
        n = self.dm.shape[0]
        IDX = np.array([])
        V = []
        for i in range(self.t):
            subIndex = self.centroid[i]
            dis = self.dm[subIndex,:]
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                V.append(1) 
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata

    def transform_continous_dm(self, newdata=None):
        # Lambda<infty, geodesic distance
        assert self.centroid != None, "invoke fit() first!"
        #if self.iso == 1:
        #    assert np.all(newdata==self.data),'Match error!'
        n = self.dm.shape[0]
        IDX = np.array([])
        V = []
        for i in range(self.t):
            subIndex = self.centroid[i]
            dis = self.dm[subIndex,:]
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                V.append(dis[centerIdx[j], j])
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata
    
    

    

