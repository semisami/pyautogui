import logging
import os
import sys
import time
import webbrowser
import pyautogui
import pyperclip
import psutil
import cv2
import numpy as np
# from tkinter import Tk, Button, Label, Entry, messagebox
from tkinter import Tk, Button, Label, Text, Scrollbar, messagebox, END, RIGHT, Y, LEFT, BOTH, Frame
from tkinter import font as tkFont
from pynput import mouse, keyboard
from datetime import datetime

# تنظیمات لاگینگ
logging.basicConfig(
    filename='automation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# کانفیگ‌ها
CONFIG = {
    'confidence': 0.7,
    'image_timeout': 10,
    'browser_process_name': 'chrome',
    'mouse_move_threshold': 1000,
    'image_paths': {
        'chatgpt': 'assets/chatGPT.png',
        'tools': 'assets/tools.png',
        'sendgpt': 'assets/sendgpt.png',
        'msg': 'assets/msg.png',
        'instagram': 'assets/instagram.png',
        '3dot': 'assets/3dot.png',
        'cpmsg': 'assets/cpmsg.png',
        'cpgpt': 'assets/cpgpt.png',
        'sendinsta': 'assets/sendinsta.png'
    },
    'image_configs': {
        'chatgpt': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': np.arange(0.8, 1.2, 0.1).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'tools': {'default_scale': 1.0, 'default_confidence': 0.6, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'sendgpt': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'msg': {'default_scale': 1.0, 'default_confidence': 0.6, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'instagram': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        '3dot': {'default_scale': 1.0, 'default_confidence': 0.8, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'cpmsg': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'cpgpt': {'default_scale': 1.0, 'default_confidence': 0.4, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'sendinsta': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]}
    },
    'color_dict': {
        'newmsg': (38, 38, 38),
        'black': (0, 0, 0),
        'typing': (48, 48, 48)
    },
    'offsets': {
        'tools_y': -50,
        'msg_x': -40,
        'msg_y': -90,
        'msg_relative_x': -30,
        'msg_relative_y': -65
    },
    'loop_count': 1000,
    'default_instagram_url': 'https://www.instagram.com/direct/t/17842735848513381/',
    'default_prompt': 'سلام، لطفا جواب های کوتاه و ساده بده...'
}

# متغیرهای جهانی
is_running = False
last_mouse_pos = None
stop_reason = None
screen_width, screen_height = pyautogui.size()
scale_factor = 1.0

def resource_path(relative_path):
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        full_path = os.path.join(base_path, relative_path)
        logger.info(f"Attempting to access file: {full_path}")
        if not os.path.exists(full_path):
            logger.error(f"File {full_path} does not exist")
        return full_path
    except Exception as e:
        logger.error(f"Error in resource_path: {str(e)}")
        return relative_path

def save_debug_image(img, name):
    os.makedirs("logs", exist_ok=True)
    path = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.png"
    cv2.imwrite(path, img)
    print(f"[+] Saved: {path}")

def find_image_smart(image_path, image_key, scales=None, min_confidence=None, region=None):
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        print(f"❌ Image {image_key} not found: {image_path}")
        return None

    try:
        # خواندن تصویر قالب به‌صورت BGR (فرمت پیش‌فرض OpenCV)
        template = cv2.imread(image_path)
        if template is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        if scales is None:
            scales = CONFIG['image_configs'][image_key]['scale_range']
        if min_confidence is None:
            min_confidence = CONFIG['image_configs'][image_key]['default_confidence']
        default_scale = CONFIG['image_configs'][image_key]['default_scale']

        scales = [s * scale_factor for s in scales]

        # گرفتن اسکرین‌شات که به‌صورت RGB است
        screenshot = pyautogui.screenshot()
        screen_img_rgb = np.array(screenshot)
        # تبدیل اسکرین‌شات از RGB به BGR برای سازگاری با OpenCV
        screen_img_bgr = cv2.cvtColor(screen_img_rgb, cv2.COLOR_RGB2BGR)

        if region:
            x, y, w, h = region
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                logger.error(f"Invalid region: {region}")
                return None
            screen_img_bgr = screen_img_bgr[y:y+h, x:x+w]

        # ذخیره تصاویر برای دیباگ (در فضای BGR)
        save_debug_image(template, f"template_{image_key}")
        save_debug_image(screen_img_bgr, f"screenshot_{image_key}")

        best_val = 0
        best_rect = None
        best_scale = default_scale

        # تغییر اندازه تصویر قالب با مقیاس پیش‌فرض
        resized_template = cv2.resize(template, None, fx=default_scale, fy=default_scale, interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screen_img_bgr, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        print(f"Scale {default_scale:.2f} → Score: {max_val:.4f}")
        if max_val >= min_confidence:
            h, w = resized_template.shape[:2]
            best_val = max_val
            if region:
                best_rect = (max_loc[0] + x, max_loc[1] + y, w, h)
            else:
                best_rect = (max_loc[0], max_loc[1], w, h)
            logger.info(f"Found {image_key} with default scale={default_scale:.2f}, confidence={max_val:.3f}")

        if not best_rect:
            for scale in scales:
                if abs(scale - default_scale) < 0.01:
                    continue
                resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                result = cv2.matchTemplate(screen_img_bgr, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_val:
                    h, w = resized_template.shape[:2]
                    best_val = max_val
                    best_scale = scale
                    if region:
                        best_rect = (max_loc[0] + x, max_loc[1] + y, w, h)
                    else:
                        best_rect = (max_loc[0], max_loc[1], w, h)

        if best_val >= min_confidence:
            x, y, w, h = best_rect
            CONFIG['image_configs'][image_key]['default_scale'] = best_scale
            logger.info(f"Updated config for {image_key}: scale={best_scale:.2f}, confidence={best_val:.3f}")
            print(f"✅ Match for {image_key} at ({x},{y}) size=({w}x{h}) | confidence={best_val:.3f}")
            # رسم مستطیل روی تصویر (در فضای BGR)
            cv2.rectangle(screen_img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            save_debug_image(screen_img_bgr, f"match_result_{image_key}")
            return (x, y, w, h)
        else:
            logger.warning(f"No good match for {image_key}. Best confidence: {best_val:.3f}")
            print(f"❌ No good match found for {image_key}. Best confidence: {best_val:.3f}")
            return None

    except Exception as e:
        logger.error(f"Error in find_image_smart for {image_key}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
def wait_for_image(image_path, image_key, timeout=CONFIG['image_timeout'], region=None):
    try:
        timeout = float(timeout)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if find_image_smart(image_path, image_key, region=region):
                logger.info(f"Image {image_key} found within {timeout} seconds")
                return True
            time.sleep(0.1)
        logger.warning(f"Image {image_key} not found within {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"Error in wait_for_image: {str(e)}")
        return False

def click_on_image(image_path, image_key, confidence=None, region=None):
    try:
        min_conf = confidence if confidence is not None else CONFIG['image_configs'][image_key]['default_confidence']
        if wait_for_image(image_path, image_key, timeout=CONFIG['image_timeout'], region=region):
            location = find_image_smart(image_path, image_key, min_confidence=min_conf, region=region)
            if location:
                center_x = location[0] + location[2] // 2
                center_y = location[1] + location[3] // 2
                pyautogui.click(center_x, center_y)
                logger.info(f"Clicked on {image_key} at ({center_x}, {center_y})")
                return True
        logger.warning(f"Image {image_key} not found for click")
        return False
    except Exception as e:
        logger.error(f"Error in click_on_image: {str(e)}")
        return False

def click_relative_to_image(image_path, image_key, offset_x=0, offset_y=0, region=None):
    try:
        if wait_for_image(image_path, image_key, timeout=CONFIG['image_timeout'], region=region):
            location = find_image_smart(image_path, image_key, region=region)
            if location:
                center_x = location[0] + location[2] // 2
                center_y = location[1] + location[3] // 2
                target_x = center_x + int(offset_x * scale_factor)
                target_y = center_y + int(offset_y * scale_factor)
                if not (0 <= target_x < screen_width and 0 <= target_y < screen_height):
                    logger.error(f"Coordinates ({target_x}, {target_y}) are outside screen bounds")
                    return False
                pyautogui.click(target_x, target_y)
                logger.info(f"Clicked at ({target_x}, {target_y}) relative to image {image_key}")
                return True
        logger.warning(f"Image {image_key} not found for relative click")
        return False
    except Exception as e:
        logger.error(f"Error in click_relative_to_image: {str(e)}")
        return False

def move_relative_to_image(image_path, image_key, offset_x=0, offset_y=0, region=None):
    try:
        if wait_for_image(image_path, image_key, timeout=CONFIG['image_timeout'], region=region):
            location = find_image_smart(image_path, image_key, region=region)
            if location:
                center_x = location[0] + location[2] // 2
                center_y = location[1] + location[3] // 2
                target_x = center_x + int(offset_x * scale_factor)
                target_y = center_y + int(offset_y * scale_factor)
                if not (0 <= target_x < screen_width and 0 <= target_y < screen_height):
                    logger.error(f"Coordinates ({target_x}, {target_y}) are outside screen bounds")
                    return False
                pyautogui.moveTo(target_x, target_y)
                logger.info(f"Mouse moved to ({target_x}, {target_y})")
                return True
        logger.warning(f"Image {image_key} not found for relative move")
        return False
    except Exception as e:
        logger.error(f"Error in move_relative_to_image: {str(e)}")
        return False

def paste_at_cursor():
    try:
        time.sleep(0.5)
        clipboard_content = pyperclip.paste()
        logger.info(f"Clipboard content before paste: {clipboard_content[:50]}...")
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.5)
        logger.info("Pasted at current cursor position")
        return True
    except Exception as e:
        logger.error(f"Error in paste_at_cursor: {str(e)}")
        return False

def capture_image_offset(image_path, image_key, offset_x=0, offset_y=0, region_size=(20, 30), region=None):
    try:
        if wait_for_image(image_path, image_key, timeout=CONFIG['image_timeout'], region=region):
            location = find_image_smart(image_path, image_key, region=region)
            if location:
                center_x = location[0] + location[2] // 2
                center_y = location[1] + location[3] // 2
                target_x = center_x + int(offset_x * scale_factor)
                target_y = center_y + int(offset_y * scale_factor)
                
                # بررسی معتبر بودن مختصات
                if not (0 <= target_x < screen_width and 0 <= target_y < screen_height):
                    logger.error(f"Coordinates ({target_x}, {target_y}) are outside screen bounds")
                    return None
                
                # بررسی معتبر بودن محدوده برش
                region_width, region_height = region_size
                if (target_x + region_width > screen_width or 
                    target_y + region_height > screen_height or 
                    region_width <= 0 or region_height <= 0):
                    logger.error(f"Invalid capture region: start=({target_x}, {target_y}), size={region_size}")
                    return None
                
                # گرفتن اسکرین‌شات
                screenshot = pyautogui.screenshot(region=(target_x, target_y, region_width, region_height))
                screenshot_np = np.array(screenshot)  # به‌صورت RGB
                screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # تبدیل به BGR
                
                # ذخیره تصویر برای دیباگ
                logger.info(f"Screenshot {region_size} captured at ({target_x}, {target_y})")
                save_debug_image(screenshot_bgr, f"captured_offset_{image_key}")
                
                return screenshot_bgr  # بازگشت تصویر در فضای BGR
        logger.warning(f"Image {image_key} not found for capture")
        return None
    except Exception as e:
        logger.error(f"Error in capture_image_offset for {image_key}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
def find_closest_color(image, color_dict):
    try:
        if image is None:
            logger.error("No image provided to find_closest_color")
            return None
        # تصویر به صورت آرایه NumPy (RGB) است
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.error(f"Invalid image format: shape={image.shape}")
            return None
        
        # تبدیل آرایه به لیست پیکسل‌ها
        pixels = image.reshape(-1, image.shape[2])
        color_count = {}
        for pixel in pixels:
            pixel_tuple = tuple(pixel)
            color_count[pixel_tuple] = color_count.get(pixel_tuple, 0) + 1
        
        # پیدا کردن پرتکرارترین رنگ
        most_common_color = max(color_count, key=color_count.get)
        
        # پیدا کردن نزدیک‌ترین رنگ در color_dict
        min_distance = float('inf')
        closest_color_key = None
        for key, rgb in color_dict.items():
            distance = sum((most_common_color[i] - rgb[i]) ** 2 for i in range(3)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_color_key = key
        
        logger.info(f"Closest color found: {closest_color_key} (color={most_common_color}, distance={min_distance:.2f})")
        return closest_color_key
    except Exception as e:
        logger.error(f"Error in find_closest_color: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def check_browser():
    try:
        for proc in psutil.process_iter(['name']):
            if proc.info['name'].lower().startswith(CONFIG['browser_process_name']):
                return True
        logger.warning("Browser process not found")
        messagebox.showwarning("Warning", "Closing the browser will disrupt the automation process!")
        return False
    except Exception as e:
        logger.error(f"Error in check_browser: {str(e)}")
        return True

def on_move(x, y):
    global last_mouse_pos, is_running, stop_reason
    try:
        if not is_running:
            return
        if last_mouse_pos:
            distance = ((x - last_mouse_pos[0])**2 + (y - last_mouse_pos[1])**2)**0.5
            if distance > CONFIG['mouse_move_threshold']:
                logger.info(f"User intervention detected: Mouse moved {distance} pixels")
                is_running = False
                stop_reason = "Mouse movement detected"
        last_mouse_pos = (x, y)
    except Exception as e:
        logger.error(f"Error in on_move: {str(e)}")

def on_key_press(key):
    global is_running, stop_reason
    try:
        if is_running:
            logger.info(f"User intervention detected: Key {key} pressed")
            is_running = False
            stop_reason = "Keyboard input detected"
    except Exception as e:
        logger.error(f"Error in on_key_press: {str(e)}")

def automation_loop(chatgpt_prompt, instagram_dm_url):
    global is_running, stop_reason
    try:
        logger.info(f"Starting automation loop. Screen: {screen_width}x{screen_height}")
        webbrowser.open(instagram_dm_url)
        webbrowser.open("https://chat.openai.com/")
        time.sleep(6)
        if not wait_for_image(resource_path(CONFIG['image_paths']['chatgpt']), 'chatgpt'):
            logger.error("Failed to load ChatGPT page")
            return

        if not click_on_image(resource_path(CONFIG['image_paths']['chatgpt']), 'chatgpt'):
            return
        if not click_relative_to_image(
            resource_path(CONFIG['image_paths']['tools']),
            'tools',
            offset_y=CONFIG['offsets']['tools_y']
        ):
            return
        time.sleep(0.1)
        pyperclip.copy(chatgpt_prompt)
        logger.info(f"Copied prompt to clipboard: {chatgpt_prompt[:50]}...")
        if not paste_at_cursor():
            return
        if not click_on_image(resource_path(CONFIG['image_paths']['sendgpt']), 'sendgpt'):
            return

        for i in range(CONFIG['loop_count']):
            if not is_running or not check_browser():
                logger.info(f"Automation stopped: {stop_reason or 'Unknown reason'}")
                break
            captured_image = capture_image_offset(
                resource_path(CONFIG['image_paths']['msg']),
                'msg',
                offset_x=CONFIG['offsets']['msg_x'],
                offset_y=CONFIG['offsets']['msg_y']
            )
            if captured_image is not None:
                closest_color = find_closest_color(captured_image, CONFIG['color_dict'])
                print(closest_color)
                if closest_color == "newmsg":
                    if not click_on_image(resource_path(CONFIG['image_paths']['instagram']), 'instagram'):
                        continue
                    if not move_relative_to_image(
                        resource_path(CONFIG['image_paths']['msg']),
                        'msg',
                        offset_x=CONFIG['offsets']['msg_relative_x'],
                        offset_y=CONFIG['offsets']['msg_relative_y']
                    ):
                        continue
                    msg_location = find_image_smart(resource_path(CONFIG['image_paths']['msg']), 'msg')
                    if msg_location:
                        msg_x, msg_y, msg_w, msg_h = msg_location
                        search_region = (
                            0,
                            max(0, msg_y - 100),
                            screen_width,
                            min(screen_height - (msg_y - 100), 150)
                        )
                        logger.info(f"Searching for 3dot in region: {search_region}")
                        try:
                            region_screenshot = pyautogui.screenshot(region=search_region)
                            region_img = np.array(region_screenshot)
                            region_img_bgr = cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR)
                            save_debug_image(region_img_bgr, f"search_region_3dot")
                            logger.info(f"Saved screenshot of search_region: {search_region}")
                        except Exception as e:
                            logger.error(f"Error capturing search_region screenshot: {str(e)}")
                        if not click_on_image(
                            resource_path(CONFIG['image_paths']['3dot']),
                            '3dot'
                            # region=search_region
                        ):
                            continue
                    else:
                        logger.warning("Could not find 'msg' image to define search region for '3dot'")
                        continue
                    if not click_on_image(resource_path(CONFIG['image_paths']['cpmsg']), 'cpmsg'):
                        continue
                    if not click_on_image(resource_path(CONFIG['image_paths']['chatgpt']), 'chatgpt'):
                        continue
                    if not click_relative_to_image(
                        resource_path(CONFIG['image_paths']['tools']),
                        'tools',
                        offset_y=CONFIG['offsets']['tools_y']
                    ):
                        continue
                    time.sleep(0.1)
                    if not paste_at_cursor():
                        continue
                    if not click_on_image(resource_path(CONFIG['image_paths']['sendgpt']), 'sendgpt'):
                        continue
                    time.sleep(5)
                    if wait_for_image(resource_path(CONFIG['image_paths']['cpgpt']), 'cpgpt', timeout=15):
                        for _ in range(1):
                            if not click_on_image(resource_path(CONFIG['image_paths']['cpgpt']), 'cpgpt', confidence=0.6):
                                break
                        if not click_on_image(resource_path(CONFIG['image_paths']['instagram']), 'instagram'):
                            continue
                        if not click_on_image(resource_path(CONFIG['image_paths']['msg']), 'msg', confidence=0.6):
                            continue
                        time.sleep(0.1)
                        if not paste_at_cursor():
                            continue
                        if not click_on_image(resource_path(CONFIG['image_paths']['sendinsta']), 'sendinsta'):
                            continue
            else:
                if not click_on_image(resource_path(CONFIG['image_paths']['instagram']), 'instagram'):
                    continue
            time.sleep(0.1)
        logger.info("Automation loop finished")
    except Exception as e:
        logger.error(f"Unexpected error in automation_loop: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        is_running = False
        stop_reason = stop_reason or "Loop completed"

def toggle_automation():
    global is_running, stop_reason
    try:
        if not is_running:
            dm_url = instagram_entry.get("1.0", END).strip()
            prompt_text = prompt_entry.get("1.0", END).strip()

            if not dm_url or not prompt_text:
                messagebox.showerror("Error", "Please enter both Instagram URL and ChatGPT prompt")
                logger.error("Empty Instagram URL or prompt")
                return
            is_running = True
            stop_reason = None
            start_button.config(text="Stop Automation")
            logger.info("Automation started")
            automation_loop(prompt_text, dm_url)
            start_button.config(text="Start Automation")
            if stop_reason:
                messagebox.showinfo("Info", f"Automation stopped: {stop_reason}")
        else:
            is_running = False
            stop_reason = "User stopped manually"
            start_button.config(text="Start Automation")
            logger.info("Automation stopped by user")
    except Exception as e:
        logger.error(f"Error in toggle_automation: {str(e)}")
        messagebox.showerror("Error", "An unexpected error occurred. Check automation.log for details.")

if __name__ == "__main__":
    try:
        pyautogui.FAILSAFE = True
        last_mouse_pos = pyautogui.position()

        try:
            import pygetwindow as gw
            window = gw.getWindowsWithTitle('Chrome')[0]
            scale_factor = screen_width / window.width if window.width else 1.0
            logger.info(f"Detected scale factor: {scale_factor}")
        except Exception as e:
            logger.warning(f"Could not detect browser window: {str(e)}")
            scale_factor = 1.0

        mouse_listener = mouse.Listener(on_move=on_move)
        keyboard_listener = keyboard.Listener(on_key_press=on_key_press)
        mouse_listener.start()
        keyboard_listener.start()

        root = Tk()
        root.title("🔥 Automation Controller 🔧")
        root.geometry("600x500")
        root.configure(bg="#1e1e1e")

        style_font = ("Segoe UI", 10)
        label_fg = "#ffffff"
        entry_bg = "#2d2d2d"
        entry_fg = "#ffffff"
        btn_bg = "#007acc"
        btn_fg = "#ffffff"

        # --- Instagram URL ---
        Label(root, text="📥 Instagram DM URL:", font=style_font, fg=label_fg, bg=root["bg"]).pack(pady=(15, 0))
        insta_frame = Frame(root, bg=root["bg"])
        insta_frame.pack(pady=5, fill=BOTH, padx=10)

        insta_scroll = Scrollbar(insta_frame)
        insta_scroll.pack(side=RIGHT, fill=Y)

        instagram_entry = Text(insta_frame, height=3, font=style_font, bg=entry_bg, fg=entry_fg, insertbackground="white", relief="flat", wrap="word", yscrollcommand=insta_scroll.set)
        instagram_entry.pack(side=LEFT, fill=BOTH, expand=True)
        insta_scroll.config(command=instagram_entry.yview)
        instagram_entry.insert(END, CONFIG['default_instagram_url'])

        # --- ChatGPT Prompt ---
        Label(root, text="💬 ChatGPT Prompt:", font=style_font, fg=label_fg, bg=root["bg"]).pack(pady=(10, 0))
        prompt_frame = Frame(root, bg=root["bg"])
        prompt_frame.pack(pady=5, fill=BOTH, padx=10)

        prompt_scroll = Scrollbar(prompt_frame)
        prompt_scroll.pack(side=RIGHT, fill=Y)

        prompt_entry = Text(prompt_frame, height=3, font=style_font, bg=entry_bg, fg=entry_fg, insertbackground="white", relief="flat", wrap="word", yscrollcommand=prompt_scroll.set)
        prompt_entry.pack(side=LEFT, fill=BOTH, expand=True)
        prompt_scroll.config(command=prompt_entry.yview)
        prompt_entry.insert(END, CONFIG['default_prompt'])

        # --- Buttons ---
        start_button = Button(root, text="▶ Start Automation", font=style_font, bg=btn_bg, fg=btn_fg, width=30, height=2, relief="flat", command=toggle_automation)
        start_button.pack(pady=(15, 5))

        Button(root, text="❌ Exit", font=style_font, bg="#cc0000", fg="white", width=30, height=2, relief="flat", command=root.quit).pack()

        # --- Static Info Text ---
        Label(root,
            text="ℹ اگر موس رو تکون بدین، برنامه متوقف میشه\nو اینکه قبل از اجرا زبان کیبورد رو انگلیسی کنید",
            font=("Segoe UI", 9, "italic"),
            fg="#cccccc",
            bg=root["bg"],
            justify="center").pack(pady=(15, 10))

        root.mainloop()



    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
    finally:
        mouse_listener.stop()
        keyboard_listener.stop()