# run_full_study.py

# ... (وارد کردن توابع)
from src.utils import load_paired_image_dataset # 👈 باید در utils.py این تابع جدید را ایجاد کنید

if __name__ == "__main__":
    # --- Configuration ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 👈 مسیرهای جدید ورودی داده واقعی شما
    LEFT_DATA_PATH = os.path.join(BASE_DIR, "auto_cropped")
    RIGHT_DATA_PATH = os.path.join(BASE_DIR, "auto_cropped_right")
    
    # ... (سایر پارامترها) ...

    print("\n--- Phase I: Loading Real IC Data and Fitting Models ---")

    # 👈 فراخوانی تابع جدید با دو مسیر
    img1_ic, img2_ic = load_paired_image_dataset(LEFT_DATA_PATH, RIGHT_DATA_PATH, size=IMG_SIZE)
    
    # ... (ادامه فاز I و فاز II) ...
