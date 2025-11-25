# run_full_study.py
import sys
import os

# 🚨🚨 FIX: رفع خطای ModuleNotFoundError با افزودن پوشه src به مسیر پایتون 🚨🚨
current_dir = os.path.dirname(os.path.abspath(__file__))
# اطمینان حاصل می‌کند که ماژول‌های داخل src (مانند simulation.py و utils.py) پیدا شوند.
sys.path.append(os.path.join(current_dir, 'src')) 

# پس از اصلاح مسیر، دیگر نیازی به پیشوند 'src.' نیست.
# تمامی ماژول‌ها مستقیماً از src وارد می‌شوند.
from simulation import generate_dataset 
from monitors import FusedMEWMA, CombinedMEWMA
from utils import load_paired_image_dataset, dummy_ooc_generator
import numpy as np
import time
from tqdm import tqdm
# import os (قبلا در بالا وارد شده است)

# ----------------------------------------------------------------------
# توابع کمکی (ARL Calculation) - بدون تغییر
# ----------------------------------------------------------------------
def single_run(monitor, shifted_gen_func, max_rl=2000):
    """اجرای یک Run Length (RL)"""
    if isinstance(monitor, FusedMEWMA):
        monitor.mewma.Z_prev = None
    else:
        monitor.mewma1.Z_prev = None
        monitor.mewma2.Z_prev = None
    
    for t in range(max_rl):
        img1, img2 = shifted_gen_func()
        is_ooc, _ = monitor.monitor(img1, img2) 
        if is_ooc:
            return t+1
    return max_rl

def compute_arl(monitor, shifted_gen_func, n_rep=300):
    """محاسبه میانگین Run Length (ARL)"""
    rls = []
    for _ in tqdm(range(n_rep), desc=f"Computing ARL for {monitor.__class__.__name__}"):
        rls.append(single_run(monitor, shifted_gen_func))
    return np.mean(rls), np.std(rls)

# ----------------------------------------------------------------------
# اجرای اصلی ورک‌فلو
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- تنظیمات برای بازتولید مقاله ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 👈 تعریف مسیرهای داده ورودی شما
    LEFT_DATA_PATH = os.path.join(BASE_DIR, "auto_cropped")
    RIGHT_DATA_PATH = os.path.join(BASE_DIR, "auto_cropped_right")
    
    # پارامترهای اصلی
    IMG_SIZE=(128, 128)        # اندازه تصویر
    LAMBDA=0.2               
    TUCKER_RANKS_MAX=(500, 40, 40, 2) # (N, R_H, R_W, R_C)
    N_REPLICATIONS = 1000      # برای نتایج دقیق، این را به 10000+ افزایش دهید.
    SHIFT_MAGNITUDE = 20     

    print(f"Project Base Directory: {BASE_DIR}")
    print("\n--- Phase I: Loading Real IC Data and Fitting Models ---")

    # گام ۱: بارگذاری تصاویر واقعی از دو پوشه
    img1_ic, img2_ic = load_paired_image_dataset(LEFT_DATA_PATH, RIGHT_DATA_PATH, size=IMG_SIZE)
    
    if len(img1_ic) == 0:
        # این بخش به دلیل بررسی‌های گیت‌هاب اکشنز نباید در CI اجرا شود، 
        # اما برای اجرای محلی یا تست باقی می‌ماند.
        print("\n!!! WARNING: No real paired IC data loaded. Using synthetic data for demonstration. !!!")
        img1_ic, img2_ic = generate_dataset(n_samples=TUCKER_RANKS_MAX[0], size=IMG_SIZE, rho_cross=0.9, smooth_sigma=2.0)
    
    N_IC_SAMPLES = len(img1_ic)
    
    # به‌روزرسانی رنک‌های تاکر
    actual_tucker_ranks = (N_IC_SAMPLES, TUCKER_RANKS_MAX[1], TUCKER_RANKS_MAX[2], TUCKER_RANKS_MAX[3])
    
    print(f"Loaded {N_IC_SAMPLES} IC samples. Fitting models...")

    # A) Fused MEWMA (با MPCA/Tucker)
    fused = FusedMEWMA(lambda_=LAMBDA)
    fused.fit(img1_ic, img2_ic, ranks=actual_tucker_ranks)
    
    # B) Combined MEWMA (دو MEWMA وکتوری)
    combined = CombinedMEWMA(lambda_=LAMBDA)
    combined.fit(img1_ic, img2_ic)

    # --- Phase II: ARL Computation ---
    print("\n--- Phase II: ARL Computation (Simulated OOC) ---")
    
    # تولید کننده داده شیفت‌یافته (OOC Generator)
    def shifted_gen():
        return dummy_ooc_generator(img1_ic, img2_ic, magnitude=SHIFT_MAGNITUDE)

    t0 = time.time()
    
    # محاسبه ARL
    arl_fused, std_fused = compute_arl(fused, shifted_gen, n_rep=N_REPLICATIONS)
    arl_combined, std_combined = compute_arl(combined, shifted_gen, n_rep=N_REPLICATIONS)
    
    t1 = time.time()
    
    # --- Results ---
    print("\n--- Final Results (ARL) ---")
    print(f"Total IC Samples Used: {N_IC_SAMPLES}")
    print(f"Fused MEWMA ARL (MPCA-Tucker) ≈ {arl_fused:.2f} (SD: {std_fused:.2f})")
    print(f"Combined MEWMA ARL (Vector) ≈ {arl_combined:.2f} (SD: {std_combined:.2f})")
    print(f"Elapsed Time for ARL Calculation (s): {t1-t0:.2f}")
