import numpy as np
from numpy.linalg import inv,svd,pinv
from scipy.linalg import norm
from scipy.spatial.transform import Rotation
from numpy import random
import cv2
import matplotlib.pyplot as plt 
from outlier_detect import outlier_iqr, outlier_zscore




def gaussian_noise(x, mu, std):
    x_noisy = [x[i] + random.normal(mu,std,size=x[0].shape) for i in range(len(x))]
    return x_noisy

def calib_camera_vanilla(a, b, line_heigth, **config):
    
    #assert len(a) == len(b) and len(a) > 0
    
    # Prepare 'a' and 'b' w.r.t. the paper
    n = len(a)
    center = np.array([(config["cam_w"] - 1) / 2, (config["cam_h"] - 1) / 2])
    a = a - center
    b = b - center
    a = np.hstack((a, np.ones((n, 1)))) # To homogeneous notatione
    b = np.hstack((b, np.ones((n, 1)))) #    (n x 2) to (n x 3)

    # Solve 'M @ v = 0 such that v > 0' in Equation (21)
    M = np.zeros((3*(n-1), 2*n), dtype=a.dtype)
    for i in range(1, n):
        s, e = (3*(i-1), 3*i)
        M[s:e, 0] = -a[0]
        M[s:e, i] = a[i]
        M[s:e, n] = b[0]
        M[s:e, n+i] = -b[i]
    
    #Method 1) Using SVD
    _, _, Vh = np.linalg.svd(M)
    v = Vh[:][-1]
    lm, mu = v[:n, np.newaxis], v[n:, np.newaxis]
    
    #assert (v > 0).all()
      
    #Calculate 'f' using Equation (24)
  
    c = mu * b - lm * a
    d = lm * a - lm[0] * a[0]
    f_n = sum([(ci[0]*di[0] + ci[1]*di[1]) * ci[2]*di[2] for ci, di in zip(c[1:], d[1:])])
    f_d = sum([(ci[2]*di[2])**2 for ci, di in zip(c[1:], d[1:])])
    #assert f_d > 0
    #assert f_n < 0
    
    eps = 1e-5
    sqrt = -f_n / (f_d + eps) 
    f = np.sqrt(sqrt)
    k= np.array([[f, 0., 0], [0., f, 0], [0., 0., 1.]]) 
    
    #Closed form of estimated r3
    try:
        kinv = inv(k)
    except np.linalg.LinAlgError:
        return "nan"
    #Initialize kinv_cs
    kinv_cs = np.zeros((len(c), 3, 1))
    #Update kinv_cs 
    kinv_cs = np.array([(kinv@cc) for cc in c]) # N * (3x1) 
    #Transform 3*1 vector to 3*3 matrix which is equivalnt to cross product  
    crs_kinv_cs =[]
    
    for kinv_c in kinv_cs:
        #Represent cross product matrix as a three by three skew-symmetric matrices for matrix multiplication
        crossed = np.array([[0,-kinv_c[2],kinv_c[1]],
                        [kinv_c[2],0,-kinv_c[0]],
                        [-kinv_c[1],kinv_c[0],0]])
        crs_kinv_cs.append(crossed)
    
    #Power that matrix 
    crs_kinv_cs_2 = np.array([crs_kinv_c @ crs_kinv_c for crs_kinv_c in crs_kinv_cs])
    
    # sum all N matrices , like A1 + A2 + A3.... 
    A = np.zeros_like(crs_kinv_cs[0])
    for element in crs_kinv_cs_2:
        A = A + element   

    # r3 can be computed as the eigenvector of this formula(A) corresponding to the smallest eigenvalue 
    try:
        _,_,Vh = svd(-A)
    except np.linalg.LinAlgError: return "nan"
    r3 =Vh[:][-1]
    
    #Solve rvec
    theta = np.arccos(r3[2])
    phi = np.arctan(-r3[0]/r3[1])
    Rx = np.array([[1., 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1.]])
    R = Rz@Rx
    #Closed form of estimated length of line segment
    l = r3.T@sum(kinv_cs)/n

    #Calculate p and q to get a position value 
    Q =np.diag([1,1,-1]) 
    pq = []
    for i in range(n):
        lma = lm[i]*a[i]
        lma = lma[:,np.newaxis]
        mub = mu[i]*b[i]
        mub = mub[:,np.newaxis]
        ij = Q @ R.T @ kinv @ np.hstack((lma,mub))
        pq.append(ij)
    p = [p[:,0]  for p in pq]
    q = [q[:,1]  for q in pq]
    
    #Calculate position
    x = [0.5*(p[i][0]+q[i][0]) for i in range(n)]
    y = [0.5*(p[i][1]+q[i][1]) for i in range(n)]
    p_3 =[ p[i][2] for i in range(n)]
    q_3 =[ q[i][2] for i in range(n)]
    height = sum(p_3+q_3)/(2*n) + l/2
    
    
    #Scaling to get absolute size 
    height = height*(line_heigth/l)
    
    #Ignore that positive or negative 
    height = abs(height)

    # result 
    result ={'f': f, 
             'theta': theta, 
             'phi' : phi, 
             'height': height}
    return result
    

def calib_camera_ransac(a, b,
                        iqr =True,
                        line_height=None,
                        r_iter = 100,
                        trsh = 3,
                        **config):
    assert len(a) == len(b) and len(a) > 0

    # Prepare 'a' and 'b' w.r.t. the paper
    n = len(a)
    ransac_trial, ransac_tresh= r_iter, trsh # same as the number of line segment  n 

    center = np.array([(config["cam_w"] - 1) / 2, (config["cam_h"]- 1) / 2])
    a = a - center
    b = b - center
    a = np.hstack((a, np.ones((n, 1)))) # To homogeneous notation
    b = np.hstack((b, np.ones((n, 1)))) #    (n x 2) to (n x 3)
    
    M = np.zeros((3*(n-1), 2*n), dtype=a.dtype)
    
    for i in range(1, n):
        s, e = (3*(i-1), 3*i)
        M[s:e, 0] = -a[0]
        M[s:e, i] = a[i]
        M[s:e, n] = b[0]
        M[s:e, n+i] = -b[i]

    _, _, Vh = np.linalg.svd(M)
    v = Vh[:][-1]
    lm, mu = v[:n, np.newaxis], v[n:, np.newaxis]
    lm_mu = lm/mu
    # Delete Outliers 
    if iqr: 
        outlier_index= outlier_iqr(lm_mu)
    else: 
        outlier_index= outlier_zscore(lm_mu)
    try:
        lm = np.array([lm[i] for i in range(n) if i not in outlier_index])
        mu = np.array([mu[i] for i in range(n) if i not in outlier_index])
        a = np.array([a[i] for i in range(n) if i not in outlier_index])
        b = np.array([b[i] for i in range(n) if i not in outlier_index])
    except :breakpoint()
    
    # Recalculate the number of inliers
    n = len(a)
    best_score = -1 
    for i in range(ransac_trial): 

        # Select two points randomly
        indices = []
        pts = []
        while len(indices) != 2: 
            idx = np.random.randint(0,n)     
            if idx not in pts:
                indices.append(idx)
            else: continue 

        for index in indices:
            pts.append([lm[index],mu[index]])
        
        # Make a line 
        slope =  pts[1][1]-pts[0][1] / pts[1][0]-pts[0][0]
        y_int =  -slope*pts[0][0] + pts[0][1]

        line = np.array([-slope,1,-y_int])
        score  = 0 

        for i in range(n): 
            err = np.fabs(line[0] * lm[i] + line[1] * mu[i] + line[2])
            if err < ransac_tresh:
                score += 1 
            
            if best_score < score: 
                best_score = score 
                best_pts = pts
                best_idx = indices


    # New a,b and lm,mu 

    lm = [best_pts[0][0],best_pts[1][0]]
    mu = [best_pts[0][1],best_pts[1][1]]

    a = [a[best_idx[0]],a[best_idx[1]]]
    b = [b[best_idx[0]],b[best_idx[1]]]
    
    # Solve focal length
    c = mu[0] * b[0] - lm[0] * a[0]
    d = lm[1] * a[1] - lm[0] * a[0]
    c, d = c.ravel(), d.ravel()
    f_n = (c[0]*d[0] + c[1]*d[1])
    f_d = (c[2]*d[2])
    n_d = -f_n/f_d
    f = np.sqrt(abs(n_d))
    k= np.array([[f, 0., 0], [0., f, 0], [0., 0., 1.]])
    try:
        kinv = inv(k)
    except np.linalg.LinAlgError:
        return "nan"
    
    #Solve theta and phif
    try:
        length = norm(kinv@c)
    except (np.linalg.LinAlgError, ValueError):
        return "nan"
    r3 =(kinv@c)/length
    theta = np.arccos(r3[2])
    phi = np.arctan(-r3[0]/r3[1])
    
    #Sovle the height of camera and position of two lines
    Q=np.diag((1,1,-1))
    
    Rx = np.array([[1., 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1.]])
    R = Rz@Rx
    
    # Estimate the height of the camera : cam_pos
    lma1 = lm[0] * a[0]
    lma1 = lma1[:, np.newaxis]
    lma2 = lm[1] * a[1]
    lma2 = lma2[:, np.newaxis]
    P = Q @ R.T @ kinv @ np.hstack((lma1,lma2))
    
    #Scaling
    height = P[2][0] * (line_height / length)  
    
    # result 
    result ={'f': f, 
             'theta': theta, 
             'phi' : phi, 
             'height': height}
    return result


#Calibartion Algorithm using n line segments 
def calib_camera_nlines_ransac(a,b,
                            iqr =True,
                            line_height=None,
                            r_iter = 100,
                            trsh = 3,
                            **config
                            ):
    assert len(a) == len(b) and len(a) > 0
    
    # Prepare 'a' and 'b' w.r.t. the paper
    n = len(a)
    center = np.array([(config["cam_w"] - 1) / 2, (config["cam_h"] - 1) / 2])
    a = a - center
    b = b - center
    a = np.hstack((a, np.ones((n, 1)))) # To homogeneous notation
    b = np.hstack((b, np.ones((n, 1)))) #    (n x 2) to (n x 3)

    # Solve 'M @ v = 0 such that v > 0' in Equation (21)
    M = np.zeros((3*(n-1), 2*n), dtype=a.dtype)
    for i in range(1, n):
        s, e = (3*(i-1), 3*i)
        M[s:e, 0] = -a[0]
        M[s:e, i] = a[i]
        M[s:e, n] = b[0]
        M[s:e, n+i] = -b[i]
        
    #Method 1) Using SVD
    _, _, Vh = np.linalg.svd(M)
    v = Vh[:][-1]
    lm, mu = v[:n, np.newaxis], v[n:, np.newaxis]
    lm_mu = lm/mu
    if iqr: 
        outlier_index= outlier_iqr(lm_mu)
    else: 
        outlier_index= outlier_zscore(lm_mu)
    # Delete Outliers 
    lm = np.array([lm[i] for i in range(n) if i not in outlier_index])
    mu = np.array([mu[i] for i in range(n) if i not in outlier_index])
    a = np.array([a[i] for i in range(n) if i not in outlier_index])
    b = np.array([b[i] for i in range(n) if i not in outlier_index])

    # Recalculate the number of inliers
    n = len(a)

    # Using RANSAC delete Outlier 
    ransac_trial, ransac_tresh, best_score = r_iter ,trsh ,-1 
    for i in range(ransac_trial): 
        # Select a points 
        indices = []
        pts = []
        while len(indices) != 2: 
            idx = np.random.randint(0,n) 
            if idx not in pts:
                indices.append(idx)
            else: continue
        for index in indices:
            pts.append([lm[index],mu[index]])

        # Make a line 
        slope =  pts[1][1]-pts[0][1]/pts[1][0]-pts[0][0]
        y_int =  -slope*pts[0][0] + pts[0][1]

        line = np.array([-slope,1,-y_int])
        score  = 0 
        for i in range(n): 
            err = np.fabs(line[0]*lm[i]+line[1]*mu[i]+line[2])
            if err < ransac_tresh:
                score += 1

            if best_score < score: 
                best_score = score
                best_line = line 
    
    erase = []
    for i in range(n):
        err = np.fabs(best_line[0]*lm[i]+best_line[1]*mu[i]+best_line[2])
        if err > ransac_tresh:
            erase.append(i)
    
    # Delete outlier a
    for e in erase:
        a.remove(e)
        b.remove(e)
        lm.remove(e)
        mu.remove(e)
    #Calculate 'f' using Equation (24)
    c = mu * b - lm * a
    d = lm * a - lm[0] * a[0]
    f_n = sum([(ci[0]*di[0] + ci[1]*di[1]) * ci[2]*di[2] for ci, di in zip(c[1:], d[1:])])
    f_d = sum([(ci[2]*di[2])**2 for ci, di in zip(c[1:], d[1:])])
    #assert f_d > 0
    #assert f_n < 0
    eps = 1e-6
    sqrt = -f_n / (f_d + eps)
    f = np.sqrt(sqrt)
    k= np.array([[f, 0., 0], [0., f, 0], [0., 0., 1.]]) 
    
    #Closed form of estimated r3
    try:
        kinv = inv(k)
    except np.linalg.LinAlgError:
        return "nan"
    #Initialize kinv_cs
    kinv_cs = np.zeros((len(c), 3, 1))
    #Update kinv_cs 
    kinv_cs = np.array([(kinv@cc) for cc in c]) # N * (3x1) 
    #Transform 3*1 vector to 3*3 matrix which is equivalnt to cross product  
    crs_kinv_cs =[]
    
    for kinv_c in kinv_cs:
        #Represent cross product matrix as a three by three skew-symmetric matrices for matrix multiplication
        crossed = np.array([[0,-kinv_c[2],kinv_c[1]],
                        [kinv_c[2],0,-kinv_c[0]],
                        [-kinv_c[1],kinv_c[0],0]])
        crs_kinv_cs.append(crossed)
    
    #Power that matrix 
    crs_kinv_cs_2 = np.array([crs_kinv_c @ crs_kinv_c for crs_kinv_c in crs_kinv_cs])
    
    # sum all N matrices , like A1 + A2 + A3.... 
    A = np.zeros_like(crs_kinv_cs[0])
    for element in crs_kinv_cs_2:
        A = A + element   

    # r3 can be computed as the eigenvector of this formula(A) corresponding to the smallest eigenvalue
    try:
        _,_,Vh = svd(-A)
        r3 =Vh[:][-1]
    except np.linalg.LinAlgError: return "nan"

    #Solve rvec
    theta = np.arccos(r3[2])
    phi = np.arctan(-r3[0]/r3[1])
    Rx = np.array([[1., 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1.]])
    R = Rz@Rx

    #Closed form of estimated length of line segment
    l = r3.T@sum(kinv_cs)/n

    #Calculate p and q to get a position value 
    Q =np.diag([1,1,-1]) 
    pq = []
    for i in range(n):
        lma = lm[i]*a[i]
        lma = lma[:,np.newaxis]
        mub = mu[i]*b[i]
        mub = mub[:,np.newaxis]
        ij = Q @ R.T @ kinv @ np.hstack((lma,mub))
        pq.append(ij)
    p = [p[:,0]  for p in pq]
    q = [q[:,1]  for q in pq]
    
    # Calculate the heigth
    p_3 =[ p[i][2] for i in range(n)]
    q_3 =[ q[i][2] for i in range(n)]
    height = sum(p_3+q_3)/(2*n) + l/2
    
    #Scaling to get absolute size 
    height = height*(line_height/l)
    
    #Ignore that positive or negative 
    height = abs(height)

    # result 
    result = dict()
    result['f'] = f
    result['theta'] =theta
    result['phi'] = phi
    result['height'] = height
    result['viz'] =v
   
    return result




def calib_camera_stat(a, b, iqr = True, line_height=None, **config):
    assert len(a) == len(b) and len(a) > 0
    
    # Prepare 'a' and 'b' w.r.t. the paper
    n = len(a)
    center = np.array([(config["cam_w"] - 1) / 2, (config["cam_h"] - 1) / 2])
    a = a - center
    b = b - center
    a = np.hstack((a, np.ones((n, 1)))) # To homogeneous notation
    b = np.hstack((b, np.ones((n, 1)))) #    (n x 2) to (n x 3)
    
    # Solve 'M @ v = 0 such that v > 0' in Equation (21)
    M = np.zeros((3*(n-1), 2*n), dtype=a.dtype)
    for i in range(1, n):
        s, e = (3*(i-1), 3*i)
        M[s:e, 0] = -a[0]
        M[s:e, i] = a[i]
        M[s:e, n] = b[0]
        M[s:e, n+i] = -b[i]
        
    # Solve Mv = 0 by SVD 
    _, _, Vh = np.linalg.svd(M)
    v = Vh[:][-1]
    lm, mu = v[:n, np.newaxis], v[n:, np.newaxis]

    # To detect outlier, divide both 
    lm_mu = lm / mu 
    if iqr: 
        outlier_index= outlier_iqr(lm_mu)
    else: 
        outlier_index= outlier_zscore(lm_mu)
   
    # Delete Outliers 
    lm = np.array([lm[i] for i in range(n) if i not in outlier_index])
    mu = np.array([mu[i] for i in range(n) if i not in outlier_index])
    a = np.array([a[i] for i in range(n) if i not in outlier_index])
    b = np.array([b[i] for i in range(n) if i not in outlier_index])

    # Recalculate the number of inliers
    n = len(a)

    #Calculate 'f' using Equation (24) 
    c = mu * b - lm * a
    d = lm * a - lm[0] * a[0]
    f_n = sum([(ci[0]*di[0] + ci[1]*di[1]) * ci[2]*di[2] for ci, di in zip(c[1:], d[1:])])
    f_d = sum([(ci[2]*di[2])**2 for ci, di in zip(c[1:], d[1:])])
    #assert f_d > 0
    #assert f_n < 0

    eps = 1e-6
    sqrt = -f_n / (f_d + eps)
    f = np.sqrt(sqrt)
    k= np.array([[f, 0., 0], [0., f, 0], [0., 0., 1.]]) 
    
    #Closed form of estimated r3
    try:
        kinv = inv(k)
    except:
        return "nan"
    #Initialize kinv_cs
    kinv_cs = np.zeros((len(c), 3, 1))
    #Update kinv_cs 
    kinv_cs = np.array([(kinv@cc) for cc in c]) # N * (3x1) 
    #Transform 3*1 vector to 3*3 matrix which is equivalnt to cross product  
    crs_kinv_cs =[]
    
    for kinv_c in kinv_cs:
        #Represent cross product matrix as a three by three skew-symmetric matrices for matrix multiplication
        crossed = np.array([[0,-kinv_c[2],kinv_c[1]],
                        [kinv_c[2],0,-kinv_c[0]],
                        [-kinv_c[1],kinv_c[0],0]])
        crs_kinv_cs.append(crossed)
    
    #Power that matrix 
    crs_kinv_cs_2 = np.array([crs_kinv_c @ crs_kinv_c for crs_kinv_c in crs_kinv_cs])
    
    # sum all N matrices , like A1 + A2 + A3.... 
    A = np.zeros_like(crs_kinv_cs[0])
    for element in crs_kinv_cs_2:
        A = A + element   

    # r3 can be computed as the eigenvector of this formula(A) corresponding to the smallest eigenvalue 
    try:    
        _,_,Vh = svd(-A)
        r3 =Vh[:][-1]
    except np.linalg.LinAlgError: return "nan"
    #Solve rvec
    theta = np.arccos(r3[2])
    phi = np.arctan(-r3[0]/r3[1])
    Rx = np.array([[1., 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1.]])
    R = Rz@Rx

    #Closed form of estimated length of line segment
    l = r3.T@sum(kinv_cs)/n

    #Calculate p and q to get a position value 
    Q =np.diag([1,1,-1]) 
    pq = []
    for i in range(n):
        lma = lm[i]*a[i]
        lma = lma[:,np.newaxis]
        mub = mu[i]*b[i]
        mub = mub[:,np.newaxis]
        ij = Q @ R.T @ kinv @ np.hstack((lma,mub))
        pq.append(ij)
    p = [p[:,0]  for p in pq]
    q = [q[:,1]  for q in pq]
    
    # Calculate the heigth
    p_3 =[ p[i][2] for i in range(n)]
    q_3 =[ q[i][2] for i in range(n)]
    height = sum(p_3+q_3)/(2*n) + l/2
  
    #Scaling to get absolute size 
    height = height*(line_height/l)
    
    #Ignore that positive or negative 
    height = abs(height)

    # result 
    result = dict()
    result['f'] = f
    result['theta'] =theta
    result['phi'] = phi
    result['height'] = height
 
    return result






if __name__ == "__main__":
   
    # Define the test configuration
    cam_f = 1000
    l = 1.  # length of lines [m]
    h = 3   #[m]
    theta_gt = np.deg2rad(20+90) # Camera 좌표계에서 바라보는 각도
    phi_gt = np.deg2rad(0) # Camera 좌표계에서 바라보는 각도 
    cam_pos = [0,0,h] # world 좌표계에서 카메라의 position 을 관찰 했을때 
    cam_w, cam_h =(1920, 1080)
    n = 300
    noise_mean = 0
    noise_std = 6
    
    lines =[]
    for i in range(n):
        x = np.random.uniform(0,10)
        y = np.random.uniform(0,10)
        lines.append(np.array([[x,y,0],
                                [x,y,l]]))
   
    # Project two lines
    
    Rx = np.array([[1., 0, 0], [0, np.cos(theta_gt), -np.sin(theta_gt)], [0, np.sin(theta_gt), np.cos(theta_gt)]])
    Rz = np.array([[np.cos(phi_gt), -np.sin(phi_gt), 0], [np.sin(phi_gt), np.cos(phi_gt), 0], [0, 0, 1.]])
    R_gt = Rz@Rx #Camera 좌표계에서 관찰하는 Rotation Matrix 
    r_gt = Rotation.from_matrix(R_gt)
    tvec_gt = -R_gt @ cam_pos
    
    #Project to 2d 
    # cx,cy 값을 0으로 두어 principal point 에 원점에 향하도록 
    cam_mat =np.array([[cam_f, 0., cam_w/2], [0., cam_f, cam_h/2], [0., 0., 1.]]) 
    #project lines
    p_lines= []
    for line in lines: #cv.projectpoints
        x,_= cv2.projectPoints(line, r_gt.as_rotvec(), tvec_gt.ravel(), cam_mat, np.zeros(4))
        p_lines.append(x.squeeze(1))

    p_lines = gaussian_noise(p_lines, noise_mean, noise_std)
    ### Function for implement papers
    
    #visualize data
    # plt.figure()
    # for x in p_lines:
    #     plt.plot(x[:,0], x[:,1]) 
    # plt.xlim([0,cam_w])
    # plt.ylim([cam_h,0])
    # plt.legend()
    # plt.show()
    # #bottom point a  
    a =[aa[0] for aa in p_lines]
    #head point b 
    b =[bb[1] for bb in p_lines]
   
    # calibration Result 

    ret  = calib_camera_nlines_ransac(a, 
                                b,
                                line_height=l,
                                r_iter = 150, trsh =3,
                                cam_w = cam_w,
                                cam_h = cam_h)
    
    focal_length, theta, phi, height,viz = ret['f'],ret['theta'],ret['phi'],ret['height'],ret['viz']
    

    lm,mu = viz[:n,np.newaxis], viz [n:,np.newaxis]
    lm_mu = lm/mu
    # Visualize Outliers
    x = outlier_iqr(lm_mu)
    plt.scatter(np.arange(n),lm_mu)
    plt.scatter(x, lm_mu[x], color = 'r')
    plt.xlabel("Index")
    plt.ylabel("lambda divided mu")
    plt.show()

    print("1.define calibration :\n")
    print(f"논문 Camera Calibration Using Parallel Line Segments 구현 중\n"
            f"2 line segment 에 의한 camera calbiration 방식")
    print('-----------------------------------')
    print("2.describe project situation :\n")
    print("f:3차원 상에 존재하는 선분의 bottom point (a)와 head point (b)"
            f"를 image plane에 투영하여 이미지 상의 픽셀 값을 논문상의 수식전개를 통해"
            f"focal length, rvec, tvec 에 대해서 계산")
    print('-----------------------------------\n')
    print(f'noise description: noise mean:{noise_mean} , noise std: {noise_std}')
    print(f'focal length: {round(focal_length,4)} [groundtruth:{cam_f }]')
    print(f'relative Error of focal length: {np.fabs(cam_f-round(focal_length,4))*100/cam_f}% ')
    print(f'theta: {np.rad2deg(theta)} [groundtruth:{np.rad2deg(theta_gt)}]')
    print(f'phi: {np.rad2deg(phi)} [groundtruth:{np.rad2deg(phi_gt)}]')
    print(f"height:{round(height,3)} [heigth:{h}]\n")
    print('-----------------------------------\n')
