# Multi-Image-SPC-MEWMA (بازتولید مقاله نظارت چند تصویری بر فرآیند آماری)

این پروژه پیاده‌سازی روش **Fused MEWMA** (با استفاده از Multi-way PCA یا Tucker Decomposition) و مقایسه آن با روش **Combined MEWMA** را برای نظارت بر فرآیندهای صنعتی مبتنی بر تصاویر ارائه می‌دهد.

## ۱. گام‌های راه‌اندازی و اجرا ⚙️

### ۱.۱. پیش‌نیازها

* Python 3.8+
* GPU (اختیاری، اما برای سرعت بی‌نظیر شبیه‌سازی‌های بزرگ توصیه می‌شود).

### ۱.۲. نصب محیط و وابستگی‌ها

```bash
# ساخت و فعال‌سازی محیط مجازی
python -m venv venv
source venv/bin/activate    # Linux/Mac

# نصب وابستگی‌ها
pip install -r requirements.txt

# برای فعال‌سازی GPU (جهت سرعت بالا)، نیاز است که tensorly و torch را با هم نصب کنید:
# pip install tensorly[pytorch] torch
