import numpy as np 
class RANSAC:
    def __init__(self, trials, threshold):
        self.trials = trials
        self.treshold = threshold
        
    @staticmethod
    def calculate_iterations(p = 0.7 , e = 0.3, s = 2):
        """
        p: 원하는 정확도
        e: outlier의 비율
        s: 모델을 추정하기 위한 무작위로 선택한 데이터 포인트의 수
        """
        N = np.log(1 - p) / np.log(1 - (1 - e) ** s)
        return np.ceil(N)  # 반복 횟수는 항상 올림하여 정수로 표현
    
    @staticmethod
    def calculate_threshold(residuals, scale_factor = 1.4826):
        """
        residuals: 모델에서 계산된 잔차
        scale_factor: 스케일 계수, 표준정규분포를 가정할 경우 1.4826 사용
        """
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        threshold = scale_factor * mad
        return threshold

    def do_ransac(self, n, lm, mu):

        best_score = -1 
        for r_t in range(self.trials): 

            # Select two index randomly
            indices = np.random.choice(n, 2, replace=False)
            pts = [[self.lm[i], self.mu[i]] for i in indices]

            # Make a line 
            (x1, y1), (x2, y2) = pts
            slope =  (y2 - y1) / (x2 - x1)
            y_int =  y1 - slope * x1
            line = np.array([-slope,1,-y_int])

            errs = []
            # Explore all datas 
            for i in range(n): 
                
                err = np.fabs(line[0] * lm[i] + line[1] * mu[i] + line[2])
                errs.append(err)
                if err < self.treshold:
                    score += 1 
                
                if best_score < score: 
                    best_score = score 
                    best_pts = pts
                    best_idx = indices
            
            # For debugging...
            errs = np.array(errs)
        return best_idx
    