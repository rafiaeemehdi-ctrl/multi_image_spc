# simulation.py
import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import gaussian_filter  # اگر smooth_sigma استفاده می‌کنی، اضافه کن

def simulate_cross_correlated_pair(size=(128,128), rho_auto=0.95, rho_cross=0.7, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    h, w = size
    H = 2 * h
    W = 2 * w
    
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt((X - X.T)**2 + (Y - Y.T)**2)
    
    range_param = 20.0
    gamma = np.exp(-dist / range_param)
    
    C11 = rho_auto * gamma
    C22 = rho_auto * gamma
    C12 = rho_cross * gamma
    
    C_circ = np.zeros((H, W))
    C_circ[:h, :w] = C11
    C_circ[:h, W-w:] = C11[:, ::-1]
    C_circ[H-h:, :w] = C11[::-1, :]
    C_circ[H-h:, W-w:] = C11[::-1, ::-1]
    
    eigenvalues = np.real(fftn(C_circ))
    eigenvalues = np.maximum(eigenvalues, 0)
    
    Z = np.random.randn(H, W)
    field = np.real(ifftn(np.sqrt(eigenvalues) * fftn(Z)))
    
    img1 = field[:h, :w]
    img2 = field[h:, :w]
    
    def normalize(img):
        img = img - img.min()
        ptp_val = np.ptp(img)  # فیکس: np.ptp(img) به جای img.ptp()
        img = img / (ptp_val + 1e-8)
        return (img * 255).astype(np.uint8)
        
    return normalize(img1), normalize(img2)

def generate_dataset(n_samples=10000, **kwargs):
    img1_list, img2_list = [], []
    for i in range(n_samples):
        im1, im2 = simulate_cross_correlated_pair(seed=i, **kwargs)
        img1_list.append(im1)
        img2_list.append(im2)
    return np.array(img1_list), np.array(img2_list)
