# mewma.py
import numpy as np

class MEWMA:
    def __init__(self, lambda_=0.2, ARL0=200):
        self.lambda_ = lambda_
        self.ARL0 = ARL0
        self.mean = None # بردار میانگین IC (mu_0)
        self.cov = None  # ماتریس کوواریانس IC (Sigma_0)
        self.inv_cov_Z = None # معکوس کوواریانس آماره Z (Sigma_Z^(-1))
        self.Z_prev = None # بردار MEWMA قبلی (Z_{t-1})
        self.h = None      # حد کنترل (Control Limit)

    def fit(self, X):
        """X: (n_samples, p) - Score Vectors"""
        p = X.shape[1]
        self.mean = np.mean(X, axis=0)
        centered = X - self.mean
        cov = np.cov(centered, rowvar=False)
        # افزودن مقدار کوچک به قطر برای اطمینان از مثبت معین بودن ماتریس
        cov = cov + np.eye(cov.shape[0]) * 1e-6
        self.cov = cov
        
        # محاسبه ماتریس کوواریانس آماره MEWMA (Sigma_Z)
        Sigma_Z = (self.lambda_ / (2 - self.lambda_)) * cov
        self.inv_cov_Z = np.linalg.pinv(Sigma_Z)
        
        # تعیین حد کنترل (h) - این مقدار باید از جدول‌های مقاله یا با شبیه‌سازی مونت‌کارلو 
        # (برای ARL0 مشخص) تعیین شود. مقدار زیر یک تخمین خام است.
        # *برای بازتولید دقیق مقاله، باید مقدار hard-coded مقاله را از جدول‌ها استخراج کنید.*
        # مثلاً، برای lambda=0.2 و ARL0=200، h نزدیک به 12-14 است.
        self.h = 13.0 + 0.002 * p # *مقدار تقریبی*

    def update(self, x):
        """x: (p,) - Score Vector جدید"""
        x_c = x - self.mean
        if self.Z_prev is None:
            self.Z_prev = np.zeros_like(x_c)
            
        # محاسبه بردار MEWMA فعلی (Z_t)
        Z = self.lambda_ * x_c + (1-self.lambda_)*self.Z_prev
        
        # محاسبه آماره هاتلینگ T^2
        # T^2 = Z_t.T * Sigma_Z^(-1) * Z_t
        T2 = float(Z @ self.inv_cov_Z @ Z)
        self.Z_prev = Z
        
        # بررسی خروج از کنترل
        return T2 > self.h, T2
