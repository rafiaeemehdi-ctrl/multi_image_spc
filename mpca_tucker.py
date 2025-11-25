# mpca_tucker.py
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker

tl.set_backend('numpy') # مطمئن شوید که tensorly با numpy کار می‌کند

def mpca_fit_tucker(img1_array, img2_array, ranks=(10, 10, 10, 2)):
    """
    تجزیه تاکر (MPCA) برای کاهش بعد.
    img1_array, img2_array: (N, H, W)
    ranks: (R_N, R_H, R_W, R_C)
    """
    N, H, W = img1_array.shape
    # ساخت تنسور داده: (N, H, W, 2)
    X = np.stack((img1_array, img2_array), axis=-1).astype(np.float64)
    
    # مرکزدهی داده: از میانگین فاز I کم می‌شود
    mean_tensor = X.mean(axis=0)
    Xc = X - mean_tensor
    
    # اجرای تجزیه تاکر
    # توجه: ranks[0] معمولاً N است، اما در اینجا برای کاهش بعد نمونه‌ها نیز از آن استفاده می‌کنیم.
    # به دلیل حجم محاسباتی، ranks باید کوچک انتخاب شوند. مقاله از ranks=(40,40,1) برای (128,128,2) استفاده کرده.
    # ما از ranks=(N, R_H, R_W, R_C) برای محاسبه مؤلفه‌ها استفاده می‌کنیم.
    
    # محاسبه مؤلفه‌ها (فاکتورها)
    core, factors = tucker(Xc, ranks=ranks, init='random', n_iter_max=50, verbose=False)
    
    # scores: تنسور هسته (Core Tensor) در واقع معادل امتیازات (Scores) است.
    # Scores = X_centered x_H U_H.T x_W U_W.T x_C U_C.T
    
    # مؤلفه‌های نهایی برای مانیتورینگ: مؤلفه‌هایی که نویز را فیلتر می‌کنند
    # در مقاله، آنها از مؤلفه‌های حالت‌های H, W, و C برای کاهش بعد استفاده می‌کنند.
    
    # بازسازی (Approximation): Xc_hat = core x_H U_H x_W U_W x_C U_C
    # ما فقط به فضای سیگنال نیاز داریم، بنابراین از مؤلفه‌های H, W, C استفاده می‌کنیم.
    
    # کاهش ابعاد به شکل (N, R_H * R_W * R_C)
    # ما تنسور هسته (core) را به عنوان scores استفاده می‌کنیم.
    # ابعاد Scores: (N, R_H, R_W, R_C)
    
    # تبدیل Scores از (N, R_H, R_W, R_C) به (N, p_monitor)
    scores = core.reshape(N, -1)
    
    # برگرداندن پارامترهای مدل:
    # 1. scores (برای fit کردن MEWMA): (N, p_monitor)
    # 2. mean_tensor (برای مرکزدهی داده جدید): (H, W, 2)
    # 3. factors (برای تبدیل داده جدید): [U_N, U_H, U_W, U_C]
    return scores, mean_tensor, factors

def mpca_transform_tucker(new_img1, new_img2, mean_tensor, factors):
    """
    تبدیل نمونه‌های جدید به فضای کاهش‌یافته (Scores)
    """
    H, W = new_img1.shape
    x = np.stack((new_img1, new_img2), axis=-1).astype(np.float64) # (H, W, 2)
    x_c = x - mean_tensor
    
    # فاکتورها: factors[0]=U_N، factors[1]=U_H، factors[2]=U_W، factors[3]=U_C
    U_H = factors[1]
    U_W = factors[2]
    U_C = factors[3]
    
    # اعمال تبدیل تاکر برای به‌دست‌آوردن تنسور هسته (Scores) برای یک نمونه جدید
    # score_tensor = x_c x_H U_H.T x_W U_W.T x_C U_C.T
    # tucker_mode_product یک نمونه (rank-3 tensor) را در مؤلفه‌های عوامل ضرب می‌کند.
    
    # ضرب مُد-۲ (H)
    core_H = tl.mode_dot(x_c, U_H.T, mode=0) 
    # ضرب مُد-۳ (W)
    core_HW = tl.mode_dot(core_H, U_W.T, mode=1)
    # ضرب مُد-۴ (C)
    score_tensor = tl.mode_dot(core_HW, U_C.T, mode=2)
    
    # خروجی: وکتور امتیازات (score vector): (R_H * R_W * R_C)
    return score_tensor.reshape(-1)
