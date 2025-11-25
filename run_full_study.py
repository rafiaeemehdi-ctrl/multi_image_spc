# run_full_study.py
from simulation import generate_dataset, generate_shifted_image
from monitors import FusedMEWMA, CombinedMEWMA
import numpy as np
import time
from tqdm import tqdm

def single_run(monitor, shifted_gen_func, max_rl=2000):
    """اجرای یک Run Length (RL)"""
    # ریست کردن بردارهای Z (بردار MEWMA) برای شروع یک دنباله جدید
    if isinstance(monitor, FusedMEWMA):
        monitor.mewma.Z_prev = None
    else:
        monitor.mewma1.Z_prev = None
        monitor.mewma2.Z_prev = None
    
    for t in range(max_rl):
        img1, img2 = shifted_gen_func()
        is_ooc = monitor.monitor(img1, img2)
        if is_ooc:
            return t+1
    return max_rl

def compute_arl(monitor, shifted_gen_func, n_rep=300):
    """محاسبه میانگین Run Length (ARL)"""
    rls = []
    # استفاده از tqdm برای نمایش پیشرفت
    for _ in tqdm(range(n_rep), desc=f"Computing ARL for {monitor.__class__.__name__}"):
        rls.append(single_run(monitor, shifted_gen_func))
    return np.mean(rls), np.std(rls)

if __name__ == "__main__":
    # --- پارامترهای مقاله (مثال کوچک‌تر برای تست محلی) ---
    IMG_SIZE=(64,64)
    N_IC_SAMPLES=200 # تعداد نمونه‌های فاز I
    LAMBDA=0.2       # پارامتر هموارسازی MEWMA
    # برای MPCA تاکر (R_N, R_H, R_W, R_C). R_N در اینجا N_IC_SAMPLES است.
    # مقاله از رتبه‌های کوچک برای H, W استفاده می‌کند. (مثلاً 10، 10)
    TUCKER_RANKS=(N_IC_SAMPLES, 10, 10, 2) 
    N_REPLICATIONS = 300 # تعداد تکرارها برای ARL (مقاله از 10000+ استفاده می‌کند)
    SHIFT_MAGNITUDE = 20 # شدت شیفت فاز II
    
    print("--- Phase I: Fitting IC Models ---")
    
    # فاز I: تولید مجموعه داده IC
    img1_ic, img2_ic = generate_dataset(n_samples=N_IC_SAMPLES, size=IMG_SIZE, rho_cross=0.9, smooth_sigma=2.0)
    
    # الف) Fused MEWMA (با MPCA/تاکر)
    fused = FusedMEWMA(lambda_=LAMBDA)
    fused.fit(img1_ic, img2_ic, ranks=TUCKER_RANKS)
    
    # ب) Combined MEWMA (دو MEWMA وکتوری)
    combined = CombinedMEWMA(lambda_=LAMBDA)
    combined.fit(img1_ic, img2_ic)

    # --- Phase II: ARL Computation (OOC) ---
    print("\n--- Phase II: ARL Computation ---")
    
    # تولید کننده داده شیفت‌یافته (OOC Generator)
    def shifted_gen():
        return generate_shifted_image(size=IMG_SIZE, rho_cross=0.9, smooth_sigma=2.0, magnitude=SHIFT_MAGNITUDE)

    t0 = time.time()
    
    # محاسبه ARL برای Fused MEWMA
    arl_fused, std_fused = compute_arl(fused, shifted_gen, n_rep=N_REPLICATIONS)
    
    # محاسبه ARL برای Combined MEWMA
    arl_combined, std_combined = compute_arl(combined, shifted_gen, n_rep=N_REPLICATIONS)
    
    t1 = time.time()
    
    # --- Results ---
    print("\n--- Results ---")
    print(f"Parameters: Image Size={IMG_SIZE}, Samples IC={N_IC_SAMPLES}, Lambda={LAMBDA}, Shift Mag.={SHIFT_MAGNITUDE}")
    print(f"Fused MEWMA ARL (MPCA-Tucker) ≈ {arl_fused:.2f} (SD: {std_fused:.2f})")
    print(f"Combined MEWMA ARL (Vector) ≈ {arl_combined:.2f} (SD: {std_combined:.2f})")
    print(f"Elapsed Time for ARL Calculation (s): {t1-t0:.2f}")
