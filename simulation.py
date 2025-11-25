# simulation.py
import numpy as np
from scipy.ndimage import gaussian_filter

def simulate_pair_smooth(size=(64,64), rho_cross=0.9, smooth_sigma=2.0, seed=None):
    """
    تولید سریع دو تصویر همبسته فضایی برای شبیه‌سازی.
    روش: یک میدان پایه گاوسی تولید، فیلتر گاوسی می‌شود (جهت ایجاد همبستگی فضایی).
    """
    if seed is not None:
        np.random.seed(seed)
    h,w = size
    base = np.random.randn(h,w)
    # فیلتر کردن برای ایجاد همبستگی فضایی
    base = gaussian_filter(base, sigma=smooth_sigma)
    noise2 = np.random.randn(h,w)
    
    img1 = base
    # همبستگی بین-تصویری (cross-correlation)
    img2 = rho_cross * base + np.sqrt(max(0,1-rho_cross**2)) * noise2
    img2 = gaussian_filter(img2, sigma=smooth_sigma) # فیلتر کردن تصویر دوم
    
    # نرمال‌سازی به بازه [0, 255] و تبدیل به uint8
    def norm(x):
        x = x - x.min()
        x = x / (x.ptp() + 1e-12)
        return (x*255).astype(np.uint8)
    return norm(img1), norm(img2)

def generate_dataset(n_samples=200, **kwargs):
    """تولید مجموعه داده فاز I (In-Control)"""
    imgs1, imgs2 = [], []
    for i in range(n_samples):
        a,b = simulate_pair_smooth(seed=i, **kwargs)
        imgs1.append(a); imgs2.append(b)
    # خروجی به شکل (N, H, W)
    return np.array(imgs1), np.array(imgs2)

def generate_shifted_image(size=(64,64), rho_cross=0.9, smooth_sigma=2.0, magnitude=20):
    """
    تولید یک جفت تصویر خارج از کنترل (Out-of-Control)
    (شبیه‌سازی یک شیفت موضعی، همانند مثال run_full_study.py شما)
    """
    img1, img2 = simulate_pair_smooth(size=size, rho_cross=rho_cross, smooth_sigma=smooth_sigma)
    center=(size[0]//2, size[1]//2)
    radius=10
    ys, xs = np.ogrid[:size[0], :size[1]]
    mask = (ys-center[0])**2 + (xs-center[1])**2 <= radius*radius
    
    im1 = img1.astype(int)
    # اعمال شیفت در یک ناحیه دایره‌ای (تنها بر روی img1 برای تست)
    im1[mask] = np.clip(im1[mask] + magnitude, 0, 255)
    
    return im1.astype(np.uint8), img2.astype(np.uint8)
