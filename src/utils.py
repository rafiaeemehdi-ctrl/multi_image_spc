# src/utils.py
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

def load_paired_image_dataset(dir_left, dir_right, size=(128, 128)):
    """
    بارگذاری جفت تصاویر (Img1: چپ، Img2: راست) از دو دایرکتوری جداگانه.
    تصاویر باید نام فایل مشترک داشته باشند (مثلاً '20250905_031521.jpg' در هر دو پوشه).

    خروجی: تاپل (آرایه Img1ها, آرایه Img2ها) با شکل (N_samples, H, W)
    """
    if not os.path.isdir(dir_left) or not os.path.isdir(dir_right):
        print("Error: One or both data directories not found.")
        return np.array([]), np.array([])
        
    # لیست فایل‌های موجود در هر دو دایرکتوری
    supported_formats = ('.png', '.jpg', '.jpeg', '.tif')
    files_left = set([f for f in os.listdir(dir_left) if f.lower().endswith(supported_formats)])
    files_right = set([f for f in os.listdir(dir_right) if f.lower().endswith(supported_formats)])
    
    # فایل‌هایی که در هر دو پوشه مشترک هستند (جفت‌ها)
    common_files = sorted(list(files_left.intersection(files_right)))

    if not common_files:
        print("Warning: Could not find common file names (pairs) in the two directories.")
        return np.array([]), np.array([])
        
    imgs1_left = [] # Img1 (چپ)
    imgs2_right = [] # Img2 (راست)
    
    for filename in tqdm(common_files, desc=f"Loading {len(common_files)} paired images"):
        path1 = os.path.join(dir_left, filename)
        path2 = os.path.join(dir_right, filename)
        
        try:
            # خواندن به صورت Grayscale
            img1_raw = imread(path1, as_gray=True) 
            img2_raw = imread(path2, as_gray=True)
            
            # تغییر اندازه (Resizing)
            img1 = resize(img1_raw, size, anti_aliasing=True)
            img2 = resize(img2_raw, size, anti_aliasing=True)
            
            # مقیاس‌بندی به 0-255 و تبدیل به uint8
            if img1.max() <= 1.0: img1 = (img1 * 255)
            if img2.max() <= 1.0: img2 = (img2 * 255)
            
            imgs1_left.append(img1.astype(np.uint8))
            imgs2_right.append(img2.astype(np.uint8))
        except Exception as e:
            print(f"Error loading or processing file {filename}: {e}")
            continue
        
    return np.array(imgs1_left), np.array(imgs2_right)


def dummy_ooc_generator(img1_ic_list, img2_ic_list, magnitude=20):
    """
    تولید یک نمونه OOC مصنوعی (برای شبیه‌سازی فاز II) بر اساس یک تصویر IC پایه.
    """
    if len(img1_ic_list) == 0:
         # Fallback: در صورتی که بارگذاری داده واقعی شکست خورد، از داده مصنوعی استفاده کند.
         from src.simulation import simulate_pair_smooth
         img1, img2 = simulate_pair_smooth(size=(128, 128))
    else:
        # استفاده از اولین نمونه IC موجود برای اعمال شیفت مصنوعی OOC
        img1, img2 = img1_ic_list[0].copy(), img2_ic_list[0].copy() 

    IMG_SIZE = img1.shape
    
    # اعمال شیفت موضعی (شبیه‌سازی عیب)
    center=(IMG_SIZE[0]//2, IMG_SIZE[1]//2)
    radius=10
    ys, xs = np.ogrid[:IMG_SIZE[0], :IMG_SIZE[1]]
    mask = (ys-center[0])**2 + (xs-center[1])**2 <= radius*radius
    
    im1 = img1.astype(int)
    im1[mask] = np.clip(im1[mask] + magnitude, 0, 255)
    
    return im1.astype(np.uint8), img2.astype(np.uint8)
