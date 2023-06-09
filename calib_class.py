import numpy as np
from numpy.linalg import inv,svd,pinv
from scipy.linalg import norm
from scipy.spatial.transform import Rotation
from numpy import random
import cv2
import matplotlib.pyplot as plt 
from outlier_detect import outlier_iqr, outlier_zscore

class Calibrator:
    def __init__(self,config,a,b):
        self.config = config
        self.center =  np.array([(config["cam_w"] - 1) / 2, (config["cam_h"] - 1) / 2])
        self.a = np.hstack((a-self.center, np.ones((self.n, 1))))
        self.b = np.hstack((b-self.center,np.ones((self.n,1)))) 
        self.n = len(self.a)
        self.eps = 1e-5 
    
    def get_M(self):
        M = np.zeros((3*(self.n-1), 2*self.n), dtype=a.dtype)
        for i in range(1, self.n):
            s, e = (3*(i-1), 3*i)
            M[s:e, 0] = -self.a[0]
            M[s:e, i] = self.a[i]
            M[s:e, self.n] = self.b[0]
            M[s:e, self.n+i] = -self.b[i]
        return M 
    
    def get_depths(self,M):
        _,_,Vh = np.linalg(M)
        v = Vh[:][-1]
        lm, mu = v[:self.n, np.newaxis], v[self.n:,np.newaxis]

        return lm, mu 
    
    def get_c(self, lm, mu):
        self.c = mu * self.b - lm * self.a
        return self.c

    def get_d(self, lm):
        self.d = lm * self.a - lm[0] * self.a[0] 
        return self.d
    
    def get_f(self): 
        f_n = sum([(ci[0]*di[0] + ci[1]*di[1]) * ci[2]*di[2] for ci, di in zip(c[1:], d[1:])])
        f_d = sum([(ci[2]*di[2])**2 for ci, di in zip(c[1:], d[1:])])
        sqrt = -f_n / (f_d + self.eps) 
        self.f = np.sqrt(sqrt)
        return self.f
    
    #Closed form of estimated r3
    def get_r3(self):
        # Calculate intrinsic matrix K 
        K = np.array([[self.f, 0., 0], [0., self.f, 0], [0., 0., 1.]])
        try:
            kinv = inv(K)
        except np.linalg.LinAlgError:
            return "nan"
        
        kinv_cs = np.zeros((self.n, 3, 1))
        #Update kinv_cs 
        kinv_cs = np.array([(kinv@cc) for cc in self.c]) # N * (3x1) 
        #Transform 3*1 vector to 3*3 matrix which is equivalnt to cross product  
        crs_kinv_cs =[]

        #Power that matrix 
        crs_kinv_cs_pow = np.array([crs_kinv_c @ crs_kinv_c for crs_kinv_c in crs_kinv_cs])
        
        # sum all N matrices , like A1 + A2 + A3.... 
        A = np.zeros_like(crs_kinv_cs[0])
        for element in crs_kinv_cs_pow:
            A = A + element   

        # r3 can be computed as the eigenvector of this formula(A) corresponding to the smallest eigenvalue 
        try:
            _,_,Vh = svd(-A)
        except np.linalg.LinAlgError: return "nan"
        self.r3 =Vh[:][-1]

        return self.r3 
    

        

    def get_theta(self):
        #TODO
    def get_phi(self):
        #TODO
    def get_rvec(self, theta, phi, rodrigues = False):


