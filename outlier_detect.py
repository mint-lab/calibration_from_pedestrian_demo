import numpy as np 

def outlier_iqr(data, w= 1.5):
    sorted_data = sorted(data)
    q_25 = np.percentile(sorted_data, 25)
    q_75 = np.percentile(sorted_data, 75)

    iqr = q_75 - q_25
    iqr_w = iqr * w

    lb = q_25 - iqr_w
    ub = q_75 + iqr_w
    
    condition = (data < lb)|(ub < data)
    outlier_index, _ = np.where(condition)
    outlier_index = outlier_index.tolist()
    return outlier_index 

    
def outlier_zscore(data): 
    m = np.mean(data)
    s = np.std(data)
    z = (data-m)/ s 
    outlier_index,_= np.where(np.fabs(z) > 3)
    return outlier_index