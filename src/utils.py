# src/utils.py
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

def load_image_dataset(base_dir, size=(64, 64)):
    """
    بارگذاری جفت تصاویر (Img1, Img2) از یک دایرکتوری.
    فرض بر این است که تصاویر به صورت 'ID_1.ext' و 'ID_2.ext' نام‌گذاری شده‌اند.

    خروجی: تاپل (آرایه Img1ها, آرایه Img2ها) با شکل (N_samples, H, W)
    """
    if not os.path.isdir(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return np.array([]), np.array([])
        
    file_list = sorted([f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    
    # شناسایی IDهای مشترک برای جفت‌ها
    sample_ids = set()
    extensions = {}
    for f in file_list:
        parts = f.rsplit('_', 1)
        if len(parts) == 2 and parts[1][0] in ('1', '2'):
            root_name, suffix_ext = parts
            ext = os.path.splitext(suffix_ext)[1]
            sample_ids.add(root_name)
            extensions[root_name] = ext

    if not sample_ids:
        print(f"Warning: Could not find paired images in {base_dir}. Ensure naming convention is 'ID_1.ext' and 'ID_2.ext'.")
        return np.array([]), np.array([])
        
    imgs1 = []
    imgs2 = []
    
    for sample_id in tqdm(sorted(list(sample_ids)), desc=f"Loading images from {os.path.basename(base_dir)}"):
        ext = extensions.get(sample_id, '.png')
        path1 = os.path.join(base_dir, f"{sample_id}_1{ext}")
        path2 = os.path.join(base_dir, f"{sample_id}_2{ext}")
        
        if os.path.exists(path1) and os.path.exists(path2):
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
                
                imgs1.append(img1.astype(np.uint8))
                imgs2.append(img2.astype(np.uint8))
            except Exception as e:
                print(f"Error loading or processing sample {sample_id}: {e}")
                continue
        
    return np.array(imgs1), np.array(imgs2)

def dummy_ooc_generator(img1_ic, img2_ic, magnitude=20):
    """
    تولید یک نمونه OOC مصنوعی (برای شبیه‌سازی فاز II) بر اساس یک تصویر IC پایه.
    """
    if len(img1_ic) == 0:
         # Fallback: در صورتی که بارگذاری داده واقعی شکست خورد، از داده مصنوعی استفاده کند.
         from src.simulation import simulate_pair_smooth
         img1, img2 = simulate_pair_smooth(size=(64, 64))
    else:
        img1, img2 = img1_ic[0], img2_ic[0] 

    IMG_SIZE = img1.shape
    
    # اعمال شیفت موضعی (شبیه‌سازی عیب)
    center=(IMG_SIZE[0]//2, IMG_SIZE[1]//2)
    radius=10
    ys, xs = np.ogrid[:IMG_SIZE[0], :IMG_SIZE[1]]
    mask = (ys-center[0])**2 + (xs-center[1])**2 <= radius*radius
    
    im1 = img1.astype(int)
    im1[mask] = np.clip(im1[mask] + magnitude, 0, 255)
    
    return im1.astype(np.uint8), img2.astype(np.uint8)
