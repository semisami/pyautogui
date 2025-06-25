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
from tkinter import Tk, Button, Label, Text, Scrollbar, messagebox, END, RIGHT, Y, LEFT, BOTH, Frame
from tkinter import font as tkFont
from pynput import mouse, keyboard
from datetime import datetime
import json

# Load configuration from config file
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    default_config = {
        'logging': {
            'enabled': True,
            'filename': 'automation.log',
            'level': 'INFO',
            'max_size_bytes': 10485760,  # 10MB
            'backup_count': 3
        },
        'screenshots': {
            'enabled': False,
            'save_directory': 'logs'
        },
        'automation': {
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
                'chatgpt': {'default_scale': 1.0, 'default_confidence': 0.8, 'scale_range': np.arange(0.9, 1.1, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
                'tools': {'default_scale': 1.0, 'default_confidence': 0.6, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
                'sendgpt': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
                'msg': {'default_scale': 1.0, 'default_confidence': 0.6, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
                'instagram': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
                '3dot': {'default_scale': 1.0, 'default_confidence': 0.65, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.65, 0.7, 0.75, 0.8]},
                'cpmsg': {'default_scale': 1.0, 'default_confidence': 0.65, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.65, 0.7, 0.8, 0.9]},
                'cpgpt': {'default_scale': 1.0, 'default_confidence': 0.4, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.4, 0.5, 0.6, 0.7]},
                'sendinsta': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': np.arange(0.8, 1.2, 0.01).tolist(), 'confidence_range': [0.6, 0.7, 0.8, 0.9]}
            },
            'templates': [
                {'name': 'msg', 'path': 'assets/msg.png'},
                {'name': 'heart', 'path': 'assets/heart.png'}
            ],
            'strip_width': 20,
            'background_threshold': 10,
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
    }
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # Merge loaded config with default to ensure all keys exist
                def merge_dicts(default, loaded):
                    for key, value in default.items():
                        if key not in loaded:
                            loaded[key] = value
                        elif isinstance(value, dict):
                            merge_dicts(value, loaded[key])
                merge_dicts(default_config, loaded_config)
                return loaded_config
        return default_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return default_config

CONFIG = load_config()

# Setup logging with rotation
if CONFIG['logging']['enabled']:
    from logging.handlers import RotatingFileHandler
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, CONFIG['logging']['level']))
    handler = RotatingFileHandler(
        CONFIG['logging']['filename'],
        maxBytes=CONFIG['logging']['max_size_bytes'],
        backupCount=CONFIG['logging']['backup_count'],
        encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
else:
    logger = logging.getLogger()
    logger.addHandler(logging.NullHandler())

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
    if not CONFIG['screenshots']['enabled']:
        return
    try:
        os.makedirs(CONFIG['screenshots']['save_directory'], exist_ok=True)
        path = f"{CONFIG['screenshots']['save_directory']}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.png"
        cv2.imwrite(path, img)
        logger.info(f"[+] Saved: {path}")
    except Exception as e:
        logger.error(f"Error saving debug image: {str(e)}")

def find_template(template_cfg):
    try:
        screenshot = pyautogui.screenshot()
        screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        tpl = cv2.imread(resource_path(template_cfg['path']))
        if tpl is None:
            logger.error(f"Failed to load template: {template_cfg['path']}")
            return None
        res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= 0.7:
            h, w = tpl.shape[:2]
            x, y = max_loc
            logger.info(f"Found {template_cfg['name']} at {x,y} conf={max_val:.2f}")
            return (x, y, w, h)
        else:
            logger.warning(f"Template {template_cfg['name']} not found, best={max_val:.2f}")
            return None
    except Exception as e:
        logger.error(f"Error in find_template: {e}")
        return None

def capture_strip(template_loc, name):
    x, y, w, h = template_loc
    strip_x = x + w//2 - CONFIG['automation']['strip_width']//2
    strip_y = 0
    strip_h = y - 30
    if strip_h <= 0:
        logger.error(f"Invalid strip height for {name}: {strip_h}")
        return None
    shot = pyautogui.screenshot(region=(strip_x, strip_y, CONFIG['automation']['strip_width'], strip_h))
    strip = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    save_debug_image(strip, f"vertical_strip_{name}")
    return strip

def analyze_strip(strip, name):
    gray = cv2.GaussianBlur(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    means = np.mean(gray, axis=1)
    msg_centers = []
    in_bg, start = False, None
    H = len(means)
    for i in range(H-1, -1, -1):
        m = means[i]
        if m <= CONFIG['automation']['background_threshold'] and not in_bg:
            if start is not None:
                msg_centers.append((i + start) // 2)
            in_bg, start = True, i
        elif m > CONFIG['automation']['background_threshold'] and in_bg:
            in_bg, start = False, i
    if start is not None and not in_bg:
        msg_centers.append(start // 2)
    dbg = strip.copy()
    for c in msg_centers:
        cv2.circle(dbg, (strip.shape[1]//2, c), 5, (0, 0, 255), -1)
    save_debug_image(dbg, f"{name}_message_centers")
    return msg_centers

def detect_messages():
    msg_targets = []
    heart_bottom_y = 0
    for tpl in CONFIG['automation']['templates']:
        loc = find_template(tpl)
        if not loc:
            continue
        strip = capture_strip(loc, tpl['name'])
        if strip is None:
            continue
        centers = analyze_strip(strip, tpl['name'])
        x_center = loc[0] + loc[2] // 2
        coords = [(x_center, cy) for cy in centers]
        if tpl['name'] == 'heart':
            if centers:
                heart_bottom_y = max(centers)
        else:
            msg_targets.extend(coords)
    filtered_msg_targets = [(x, y) for x, y in msg_targets if y > heart_bottom_y]
    filtered_msg_targets.sort(key=lambda t: t[1])
    return filtered_msg_targets

def find_image_smart(image_path, image_key, scales=None, min_confidence=None, region=None):
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None
    try:
        template = cv2.imread(image_path)
        if template is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        if scales is None:
            scales = CONFIG['automation']['image_configs'][image_key]['scale_range']
        if min_confidence is None:
            min_confidence = CONFIG['automation']['image_configs'][image_key]['default_confidence']
        default_scale = CONFIG['automation']['image_configs'][image_key]['default_scale']
        scales = [s * scale_factor for s in scales]
        screenshot = pyautogui.screenshot()
        screen_img_rgb = np.array(screenshot)
        screen_img_bgr = cv2.cvtColor(screen_img_rgb, cv2.COLOR_RGB2BGR)
        if region:
            x, y, w, h = region
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                logger.error(f"Invalid region: {region}")
                return None
            screen_img_bgr = screen_img_bgr[y:y+h, x:x+w]
        save_debug_image(screen_img_bgr, f"screenshot_{image_key}")
        save_debug_image(template, f"template_{image_key}")
        best_val = 0
        best_rect = None
        best_scale = default_scale
        resized_template = cv2.resize(template, None, fx=default_scale, fy=default_scale, interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screen_img_bgr, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
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
            CONFIG['automation']['image_configs'][image_key]['default_scale'] = best_scale
            logger.info(f"Updated config for {image_key}: scale={best_scale:.2f}, confidence={best_val:.3f}")
            return (x, y, w, h)
        else:
            logger.warning(f"No good match for {image_key}. Best confidence: {best_val:.3f}")
            return None
    except Exception as e:
        logger.error(f"Error in find_image_smart for {image_key}: {str(e)}")
        return None

def wait_for_image(image_path, image_key, timeout=CONFIG['automation']['image_timeout'], region=None):
    try:
        timeout = float(timeout)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if find_image_smart(image_path, image_key, region=region):
                logger.info(f"Image found {image_key} found within {timeout} seconds")
                return True
            time.sleep(0.1)
        logger.warning(f"Image not {image_key} not found within {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"Error in wait_for_image: {str(e)}")
        return False

def smooth_move(x, y):
    pyautogui.moveTo(x, y, duration=0.2, tween=pyautogui.easeInOutQuad)

def smooth_click(x, y):
    smooth_move(x, y)
    pyautogui.click()

def click_on_image(image_path, image_key, confidence=None, region=None):
    try:
        min_conf = confidence if confidence is not None else CONFIG['automation']['image_configs'][image_key]['default_confidence']
        if wait_for_image(image_path, image_key, timeout=CONFIG['automation']['image_timeout'], region=region):
            location = find_image_smart(image_path, image_key, min_confidence=min_conf, region=region)
            if location:
                center_x = location[0] + location[2] // 2
                center_y = location[1] + location[3] // 2
                smooth_click(center_x, center_y)
                logger.info(f"Clicked on {image_key} at ({center_x}, {center_y})")
                return True
        logger.warning(f"Image {image_key} not found for click")
        return False
    except Exception as e:
        logger.error(f"Error in click_on_image: {str(e)}")
        return False

def click_relative_to_image(image_path, image_key, offset_x=0, offset_y=0, region=None):
    try:
        if wait_for_image(image_path, image_key, timeout=CONFIG['automation']['image_timeout'], region=region):
            location = find_image_smart(image_path, image_key, region=region)
            if location:
                center_x = location[0] + location[2] // 2
                center_y = location[1] + location[3] // 2
                target_x = center_x + int(offset_x * scale_factor)
                target_y = center_y + int(offset_y * scale_factor)
                if not (0 <= target_x < screen_width and 0 <= target_y < screen_height):
                    logger.error(f"Coordinates ({target_x}, {target_y}) are outside screen bounds")
                    return False
                smooth_click(target_x, target_y)
                logger.info(f"Clicked at ({target_x}, {target_y}) relative to image {image_key}")
                return True
        logger.warning(f"Image {image_key} not found for relative click")
        return False
    except Exception as e:
        logger.error(f"Error in click_relative_to_image: {str(e)}")
        return False

def hover_on_message(x, y):
    try:
        smooth_move(x, y)
        time.sleep(0.5)  # زمان برای ظاهر شدن آیکون سه نقطه
        logger.info(f"Hovered on message at ({x}, {y})")
        return True
    except Exception as e:
        logger.error(f"Error in hover_on_message: {str(e)}")
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

def find_cpmsg_for_message(y, height=400):
    region = (0, max(0, y - height//2), screen_width, height)
    try:
        region_screenshot = pyautogui.screenshot(region=region)
        region_img = np.array(region_screenshot)
        region_img_bgr = cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR)
        save_debug_image(region_img_bgr, f"search_region_cpmsg_y_{y}")
    except Exception as e:
        logger.error(f"Error capturing cpmsg search_region screenshot: {str(e)}")
    return find_image_smart(resource_path(CONFIG['automation']['image_paths']['cpmsg']), 'cpmsg', region=region, min_confidence=0.65)

def check_browser():
    try:
        for proc in psutil.process_iter(['name']):
            if proc.info['name'].lower().startswith(CONFIG['automation']['browser_process_name']):
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
            if distance > CONFIG['automation']['mouse_move_threshold']:
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
        time.sleep(10)
        if not wait_for_image(resource_path(CONFIG['automation']['image_paths']['chatgpt']), 'chatgpt'):
            logger.error("Failed to load ChatGPT page")
            return
        if not click_on_image(resource_path(CONFIG['automation']['image_paths']['chatgpt']), 'chatgpt'):
            return
        if not click_relative_to_image(
            resource_path(CONFIG['automation']['image_paths']['tools']),
            'tools',
            offset_y=CONFIG['automation']['offsets']['tools_y']
        ):
            return
        time.sleep(0.3)
        pyperclip.copy(chatgpt_prompt)
        logger.info(f"Copied prompt to clipboard: {chatgpt_prompt[:50]}...")
        if not paste_at_cursor():
            return
        if not click_on_image(resource_path(CONFIG['automation']['image_paths']['sendgpt']), 'sendgpt'):
            return

        for i in range(CONFIG['automation']['loop_count']):
            if not is_running or not check_browser():
                logger.info(f"Automation stopped: {stop_reason or 'Unknown reason'}")
                break

            if not click_on_image(resource_path(CONFIG['automation']['image_paths']['instagram']), 'instagram'):
                logger.warning("Failed to click on Instagram, continuing loop")
                continue

            # استخراج پیام‌های جدید
            messages = detect_messages()
            if messages:
                logger.info(f"تشخیص {len(messages)} پیام جدید، شروع بررسی تغییرات")
                # ذخیره تصویر اولیه محدوده پیام‌ها
                message_region = (0, 0, screen_width, screen_height)  # کل صفحه، در صورت نیاز محدوده را تنظیم کنید
                initial_image = pyautogui.screenshot(region=message_region)
                initial_image = cv2.cvtColor(np.array(initial_image), cv2.COLOR_RGB2BGR)
                
                while True:
                    time.sleep(5)
                    current_image = pyautogui.screenshot(region=message_region)
                    current_image = cv2.cvtColor(np.array(current_image), cv2.COLOR_RGB2BGR)
                    
                    # محاسبه اختلاف پیکسلی
                    diff = cv2.absdiff(initial_image, current_image)
                    non_zero_count = np.count_nonzero(diff)
                    
                    if non_zero_count == 0:
                        # تصاویر یکسان هستند، ادامه پردازش
                        logger.info("پیام‌ها ثابت شدند، ادامه پردازش")
                        break
                    else:
                        # تصاویر متفاوت هستند، به‌روزرسانی تصویر اولیه و ادامه صبر
                        initial_image = current_image
                        logger.info("پیام‌ها در حال تغییر هستند، صبر ۵ ثانیه دیگر")

                messages = detect_messages()
                logger.info(f"بعد از ثبات، {len(messages)} پیام تشخیص داده شد")

            temp_messages = []
            for msg_x, msg_y in messages:
                # هاور روی پیام برای ظاهر شدن سه نقطه
                if not hover_on_message(msg_x, msg_y):
                    logger.warning(f"Failed to hover on message at ({msg_x}, {msg_y})")
                    continue
                # تعریف منطقه جستجو برای سه نقطه
                search_region = (0, max(0, msg_y - 50), screen_width, 100)
                logger.info(f"Searching for 3dot in region: {search_region}")
                try:
                    region_screenshot = pyautogui.screenshot(region=search_region)
                    region_img = np.array(region_screenshot)
                    region_img_bgr = cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR)
                    save_debug_image(region_img_bgr, f"search_region_3dot_msg_y_{msg_y}")
                except Exception as e:
                    logger.error(f"Error capturing 3dot search_region screenshot: {str(e)}")
                if not click_on_image(
                    resource_path(CONFIG['automation']['image_paths']['3dot']),
                    '3dot',
                    region=search_region,
                    confidence=0.65
                ):
                    logger.warning(f"Could not find '3dot' for message at y={msg_y}")
                    continue
                time.sleep(0.3)  # تأخیر برای باز شدن منوی کپی
                cpmsg_loc = find_cpmsg_for_message(msg_y)
                if cpmsg_loc:
                    smooth_click(cpmsg_loc[0] + cpmsg_loc[2]//2, cpmsg_loc[1] + cpmsg_loc[3]//2)
                    time.sleep(0.5)
                    copied_message = pyperclip.paste()
                    if copied_message:
                        temp_messages.append(copied_message)
                        logger.info(f"Copied message: {copied_message[:50]}...")
                    else:
                        logger.warning(f"No text copied for message at y={msg_y}")
                else:
                    logger.warning(f"Could not find 'cpmsg' for message at y={msg_y}")

            # ارسال پیام‌ها به ChatGPT
            if temp_messages:
                logger.info(f"Processing {len(temp_messages)} messages in ChatGPT")
                if not click_on_image(resource_path(CONFIG['automation']['image_paths']['chatgpt']), 'chatgpt'):
                    logger.warning("Failed to click on ChatGPT, continuing loop")
                    continue
                if not click_relative_to_image(
                    resource_path(CONFIG['automation']['image_paths']['tools']),
                    'tools',
                    offset_y=CONFIG['automation']['offsets']['tools_y']
                ):
                    logger.warning("Failed to click on tools, continuing loop")
                    continue
                time.sleep(0.3)
                combined_messages = "\n".join(temp_messages)
                pyperclip.copy(combined_messages)
                logger.info(f"Copied combined messages to clipboard: {combined_messages[:50]}...")
                if not paste_at_cursor():
                    logger.warning("Failed to paste messages in ChatGPT, continuing loop")
                    continue
                if not click_on_image(resource_path(CONFIG['automation']['image_paths']['sendgpt']), 'sendgpt'):
                    logger.warning("Failed to click sendgpt, continuing loop")
                    continue
                time.sleep(5)
                if wait_for_image(resource_path(CONFIG['automation']['image_paths']['cpgpt']), 'cpgpt', timeout=25):
                    for _ in range(1):
                        if not click_on_image(resource_path(CONFIG['automation']['image_paths']['cpgpt']), 'cpgpt', confidence=0.6):
                            logger.warning("Failed to click cpgpt, breaking inner loop")
                            break
                    if not click_on_image(resource_path(CONFIG['automation']['image_paths']['instagram']), 'instagram'):
                        logger.warning("Failed to click Instagram after ChatGPT, continuing loop")
                        continue
                    if not click_on_image(resource_path(CONFIG['automation']['image_paths']['msg']), 'msg', confidence=0.6):
                        logger.warning("Failed to click msg, continuing loop")
                        continue
                    time.sleep(0.3)
                    if not paste_at_cursor():
                        logger.warning("Failed to paste response in Instagram, continuing loop")
                        continue
                    if not click_on_image(resource_path(CONFIG['automation']['image_paths']['sendinsta']), 'sendinsta'):
                        logger.warning("Failed to click sendinsta, continuing loop")
                        continue
                else:
                    logger.warning("Could not find cpgpt, continuing loop")
            else:
                logger.info("No new messages found, continuing loop")
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
        root.title("Instagram Admin Assistant")
        root.geometry("600x500")
        root.configure(bg="#1e1e1e")
        style_font = ("Segoe UI", 10)
        label_fg = "#ffffff"
        entry_bg = "#2d2d2d"
        entry_fg = "#ffffff"
        btn_bg = "#007acc"
        btn_fg = "#ffffff"
        Label(root, text="📥 Instagram DM URL:", font=style_font, fg=label_fg, bg=root["bg"]).pack(pady=(15, 0))
        insta_frame = Frame(root, bg=root["bg"])
        insta_frame.pack(pady=5, fill=BOTH, padx=10)
        insta_scroll = Scrollbar(insta_frame)
        insta_scroll.pack(side=RIGHT, fill=Y)
        instagram_entry = Text(insta_frame, height=3, font=style_font, bg=entry_bg, fg=entry_fg, insertbackground="white", relief="flat", wrap="word", yscrollcommand=insta_scroll.set)
        instagram_entry.pack(side=LEFT, fill=BOTH, expand=True)
        insta_scroll.config(command=instagram_entry.yview)
        instagram_entry.insert(END, CONFIG['automation']['default_instagram_url'])
        Label(root, text="💬 ChatGPT Prompt:", font=style_font, fg=label_fg, bg=root["bg"]).pack(pady=(10, 0))
        prompt_frame = Frame(root, bg=root["bg"])
        prompt_frame.pack(pady=5, fill=BOTH, padx=10)
        prompt_scroll = Scrollbar(prompt_frame)
        prompt_scroll.pack(side=RIGHT, fill=Y)
        prompt_entry = Text(prompt_frame, height=3, font=style_font, bg=entry_bg, fg=entry_fg, insertbackground="white", relief="flat", wrap="word", yscrollcommand=prompt_scroll.set)
        prompt_entry.pack(side=LEFT, fill=BOTH, expand=True)
        prompt_scroll.config(command=prompt_entry.yview)
        prompt_entry.insert(END, CONFIG['automation']['default_prompt'])
        start_button = Button(root, text="▶ Start Automation", font=style_font, bg=btn_bg, fg=btn_fg, width=30, height=2, relief="flat", command=toggle_automation)
        start_button.pack(pady=(15, 5))
        Button(root, text="❌ Exit", font=style_font, bg="#cc0000", fg="white", width=30, height=2, relief="flat", command=root.quit).pack()
        Label(root,
            text="Moving the mouse will stop the program.\nPlease ensure the keyboard language is set to English before running.",
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