import numpy as np
from numpy.linalg import inv,svd,pinv
from scipy.linalg import norm
from scipy.spatial.transform import Rotation
from numpy import random
import cv2
import matplotlib.pyplot as plt 
from outlier_detect import outlier_iqr, outlier_zscore

class Calibrator:
    def __init__(self,a,b,**config):
        self.config = config
        self.center =  np.array([(config["cam_w"] - 1) / 2, (config["cam_h"] - 1) / 2])
        self.a = np.hstack((a-self.center, np.ones((self.n, 1))))
        self.b = np.hstack((b-self.center,np.ones((self.n,1)))) 
        self.n = len(self.a)
        self.eps = 1e-5 
        self.line_height = config["l"]
    
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
        self.lm, self.mu = v[:self.n, np.newaxis], v[self.n:,np.newaxis]

        return self.lm, self.mu 
    
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
            kinv = np.linalg.inv(K)
            self.kinv = kinv
        except np.linalg.LinAlgError:
            return "nan"

        # Update kinv_cs 
        kinv_cs = np.array([(kinv @ cc) for cc in self.c]) # N * (3x1) 
        
        # store self.kinv_cs (estimate_l used this)
        self.kinv_cs = kinv_cs
        
        # Represent cross product matrix as a three by three skew-symmetric matrices for matrix multiplication
        zeros = np.zeros_like(kinv_cs[:, 0])
        crs_kinv_cs = np.array([
            [zeros, -kinv_cs[:, 2], kinv_cs[:, 1]],
            [kinv_cs[:, 2], zeros, -kinv_cs[:, 0]],
            [-kinv_cs[:, 1], kinv_cs[:, 0], zeros]
        ]).transpose(2, 0, 1)

        # Power that matrix 
        crs_kinv_cs_pow = np.einsum('bij,bjk->bik', crs_kinv_cs, crs_kinv_cs)

        # Sum all N matrices , like A1 + A2 + A3.... 
        A = np.sum(crs_kinv_cs_pow, axis=0)

        # r3 can be computed as the eigenvector of this formula(A) corresponding to the smallest eigenvalue 
        try:
            _,_,Vh = np.linalg.svd(-A)
        except np.linalg.LinAlgError: 
            return "nan"
        
        self.r3 = Vh[:][-1]
        return self.r3 

    

        

    def get_theta(self):
        self.theta = np.arccos(self.r3[2])
        return self.theta
    def get_phi(self):
        self.phi = np.arctan(-self.r3[0]/self.r3[1])
        return self.phi
    
    def get_rvec(self, rodrigues = False):
        Rx = np.array([[1., 0, 0], 
                       [0, np.cos(self.theta), -np.sin(self.theta)], 
                       [0, np.sin(self.theta), np.cos(self.theta)]])
        Rz = np.array([[np.cos(self.phi), -np.sin(self.phi), 0], 
                       [np.sin(self.phi), np.cos(self.phi), 0], 
                       [0, 0, 1.]])
        rvec = Rz@Rx

        if rodrigues:
            _, rvec = cv2.Rodrigues(rvec)
        
        self.rvec = rvec
        return self.rvec

    # Estimate the length of the line segment to deal with scale ambiguity 
    def estimate_l(self):
        self.l = self.r3 @ np.sum(self.kinv_cs)/self.n
        return self.l
    
    #Calculate p and q to get a position value 
    def get_pq(self):
        Q = np.diag([1,1,-1])
        lm_a = (self.lm[:, np.newaxis] * self.a).T
        mu_b = (self.mu[:, np.newaxis] * self.b).T
        pq = Q @ self.rvec.T @ self.kinv @ np.hstack((lm_a, mu_b))
        self.p = pq[:, :, 0]
        self.q = pq[:, :, 1]

        return self.p, self.q 
    
    def get_height(self):
        p_3 = self.p[:,2]
        q_3 = self.q[:,2]
        height = np.sum(p_3+q_3)/(2*self.n) + self.l/2 
        
        #Scaling to get absolute size 
        height = height*(self.line_heigth/self.l)
    
        #Ignore that positive or negative 
        height = abs(height)
        self.height = height

        return self.height
    
    # Show results in dictionaty type 
    def show_result(self):
        result = dict()
        result["f"] = self.f
        result["theta"] = self.theta
        result["phi"] = self.phi
        result["height"] = self.height

        return result 
        

