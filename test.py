import cv2
import numpy as np
import pyautogui
import os
from datetime import datetime

def save_debug_image(img, name):
    os.makedirs("logs", exist_ok=True)
    path = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.png"
    cv2.imwrite(path, img)
    print(f"[+] Saved: {path}")

def find_image_smart(image_path, scales=np.arange(0.4, 1.5, 0.01).tolist(), min_confidence=0.75, upscale_template=2.0):
    if not os.path.exists(image_path):
        print("❌ Image not found:", image_path)
        return None

    # خواندن و بزرگ‌کردن تصویر مرجع
    template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.resize(template, None, fx=upscale_template, fy=upscale_template, interpolation=cv2.INTER_CUBIC)

    # اسکرین‌شات
    screenshot = pyautogui.screenshot()
    screen_img_rgb = np.array(screenshot)
    screen_img = cv2.cvtColor(screen_img_rgb, cv2.COLOR_RGB2GRAY)

    # ذخیره نسخه اولیه
    save_debug_image(template, "template_scaled")
    save_debug_image(screen_img, "screenshot_gray")

    best_val = 0
    best_rect = None

    for scale in scales:
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screen_img, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        print(f"Scale {scale:.2f} → Score: {max_val:.4f}")
        if max_val > best_val:
            h, w = resized_template.shape
            best_val = max_val
            best_rect = (max_loc[0], max_loc[1], w, h)

    if best_val >= min_confidence:
        x, y, w, h = best_rect
        print(f"✅ Match at ({x},{y}) size=({w}x{h}) | confidence={best_val:.3f}")
        cv2.rectangle(screen_img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        save_debug_image(cv2.cvtColor(screen_img_rgb, cv2.COLOR_RGB2BGR), "match_result")
        return (x, y, w, h)
    else:
        print(f"❌ No good match found. Best confidence: {best_val:.3f}")
        return None

# تست مستقل
if __name__ == "__main__":
    path = "assets/chatGPT.png"  # مسیر تصویر مرجع
    # path = "assets/cpgpt.png"  # مسیر تصویر مرجع
    result = find_image_smart(path)

    if result:
        print("📍 Location:", result)
    else:
        print("🔎 Match not found.")
