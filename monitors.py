# monitors.py
import numpy as np
from mewma import MEWMA
from mpca_tucker import mpca_fit_tucker, mpca_transform_tucker

class FusedMEWMA:
    """
    روش پیشنهادی مقاله: MPCA (تاکر) برای ادغام و کاهش بعد، سپس MEWMA.
    """
    def __init__(self, lambda_=0.2):
        self.mewma = MEWMA(lambda_=lambda_)
        self.mean_tensor = None
        self.factors = None # [U_N, U_H, U_W, U_C]

    def fit(self, img1_ic, img2_ic, ranks=(200, 10, 10, 2)):
        """img1_ic, img2_ic: (N, H, W)"""
        # گام ۱: MPCA/تاکر برای کاهش بعد و استخراج Scores
        scores, mean_tensor, factors = mpca_fit_tucker(img1_ic, img2_ic, ranks=ranks)
        self.mean_tensor = mean_tensor
        self.factors = factors
        
        # گام ۲: Fit کردن MEWMA بر روی Scores
        self.mewma.fit(scores)
        print(f"Fused MEWMA fitted. Score dimension: {scores.shape[1]}. Control Limit (h): {self.mewma.h:.3f}")


    def monitor(self, img1, img2):
        """مانیتورینگ یک جفت تصویر جدید"""
        # گام ۱: تبدیل تصاویر به فضای Score
        sc = mpca_transform_tucker(img1, img2, self.mean_tensor, self.factors)
        
        # گام ۲: به‌روزرسانی MEWMA و بررسی خروج از کنترل
        return self.mewma.update(sc)

class CombinedMEWMA:
    """
    روش جایگزین: دو MEWMA جداگانه (یکی برای هر تصویر) بر روی وکتورهای پیکسلی.
    """
    def __init__(self, lambda_=0.2):
        self.mewma1 = MEWMA(lambda_=lambda_)
        self.mewma2 = MEWMA(lambda_=lambda_)

    def fit(self, img1_ic, img2_ic):
        # Flatten کردن تصاویر (Vectorization)
        X1 = img1_ic.reshape(len(img1_ic), -1).astype(np.float64)
        X2 = img2_ic.reshape(len(img2_ic), -1).astype(np.float64)
        
        # Fit کردن MEWMA برای هر وکتور به صورت جداگانه
        self.mewma1.fit(X1)
        self.mewma2.fit(X2)
        print(f"Combined MEWMA 1 fitted. Pixel dimension: {X1.shape[1]}. Control Limit (h): {self.mewma1.h:.3f}")
        print(f"Combined MEWMA 2 fitted. Pixel dimension: {X2.shape[1]}. Control Limit (h): {self.mewma2.h:.3f}")

    def monitor(self, img1, img2):
        """مانیتورینگ یک جفت تصویر جدید"""
        # Flatten کردن
        x1 = img1.flatten().astype(np.float64)
        x2 = img2.flatten().astype(np.float64)
        
        # بررسی خروج از کنترل؛ اگر هر کدام سیگنال بدهد، خروج از کنترل اعلام می‌شود (OR rule)
        s1, t2_1 = self.mewma1.update(x1)
        s2, t2_2 = self.mewma2.update(x2)
        return s1 or s2
