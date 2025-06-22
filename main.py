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
import json
from tkinter import Tk, Button, Label, Entry, messagebox
from pynput import mouse, keyboard
from PIL import Image

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
    'mouse_move_threshold': 50,
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
        'chatgpt': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0], 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'tools': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0], 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'sendgpt': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0], 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'msg': {'default_scale': 1.0, 'default_confidence': 0.9, 'scale_range': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0], 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'instagram': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0], 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        '3dot': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0], 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'cpmsg': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0], 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'cpgpt': {'default_scale': 1.0, 'default_confidence': 0.6, 'scale_range': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0], 'confidence_range': [0.6, 0.7, 0.8, 0.9]},
        'sendinsta': {'default_scale': 1.0, 'default_confidence': 0.7, 'scale_range': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0], 'confidence_range': [0.6, 0.7, 0.8, 0.9]}
    },
    'color_dict': {
        'newmsg': (38, 38, 38),
        'black': (0, 0, 0),
        'typing': (48, 48, 48)
    },
    'offsets': {
        'tools_y': -50,
        'msg_x': -40,
        'msg_y': -75,
        'msg_relative_x': -30,
        'msg_relative_y': -50
    },
    'loop_count': 100,
    'default_instagram_url': 'https://www.instagram.com/direct/t/17842735848513381/',
    'default_prompt': 'سلام، لطفا جواب های کوتاه  و ساده بده...',
    'config_file': 'image_configs.json'
}

# متغیرهای جهانی
is_running = False
last_mouse_pos = None
stop_reason = None
screen_width, screen_height = pyautogui.size()
scale_factor = 1.0

def load_image_configs():
    try:
        if os.path.exists(CONFIG['config_file']):
            with open(CONFIG['config_file'], 'r') as f:
                loaded_configs = json.load(f)
            for key in CONFIG['image_configs']:
                if key in loaded_configs:
                    CONFIG['image_configs'][key]['default_scale'] = loaded_configs[key].get('default_scale', 1.0)
                    CONFIG['image_configs'][key]['default_confidence'] = loaded_configs[key].get('default_confidence', CONFIG['image_configs'][key]['default_confidence'])
            logger.info("Loaded image configurations from JSON")
    except Exception as e:
        logger.error(f"Error loading image configs: {str(e)}")

def save_image_configs():
    try:
        configs_to_save = {
            key: {
                'default_scale': CONFIG['image_configs'][key]['default_scale'],
                'default_confidence': CONFIG['image_configs'][key]['default_confidence']
            } for key in CONFIG['image_configs']
        }
        with open(CONFIG['config_file'], 'w') as f:
            json.dump(configs_to_save, f, indent=4)
        logger.info("Saved image configurations to JSON")
    except Exception as e:
        logger.error(f"Error saving image configs: {str(e)}")

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

def find_image(image_path, image_key):
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file {image_path} does not exist")
            return None
        template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            logger.error(f"Failed to load image {image_path}")
            return None
        
        screenshot = pyautogui.screenshot()
        screen_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        
        best_score = -1
        best_location = None
        best_scale = CONFIG['image_configs'][image_key]['default_scale']
        best_confidence = CONFIG['image_configs'][image_key]['default_confidence']
        
        # تست مقیاس و confidence پیش‌فرض
        scaled_template = cv2.resize(template, None, fx=best_scale, fy=best_scale, interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screen_img, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= best_confidence:
            h, w = scaled_template.shape
            best_score = max_val
            best_location = (max_loc[0], max_loc[1], w, h)
            logger.info(f"Image {image_path} found at {max_loc} with default scale {best_scale}, confidence {best_confidence}, score {max_val}")
        
        # Brute-force روی رنج مقیاس‌ها و confidence‌ها
        for scale in CONFIG['image_configs'][image_key]['scale_range']:
            for confidence in CONFIG['image_configs'][image_key]['confidence_range']:
                if scale == best_scale and confidence == best_confidence:
                    continue
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                result = cv2.matchTemplate(screen_img, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val >= confidence and max_val > best_score:
                    h, w = scaled_template.shape
                    best_score = max_val
                    best_location = (max_loc[0], max_loc[1], w, h)
                    best_scale = scale
                    best_confidence = confidence
                    logger.info(f"Image {image_path} found at {max_loc} with scale {scale}, confidence {confidence}, score {max_val}")
        
        if best_location:
            CONFIG['image_configs'][image_key]['default_scale'] = best_scale
            CONFIG['image_configs'][image_key]['default_confidence'] = best_confidence
            save_image_configs()
            return best_location
        logger.info(f"Image {image_path} not found")
        return None
    except Exception as e:
        logger.error(f"Error in find_image: {str(e)}")
        return None

def wait_for_image(image_path, image_key, timeout=CONFIG['image_timeout']):
    try:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if find_image(image_path, image_key):
                logger.info(f"Image {image_path} found within {timeout} seconds")
                return True
            time.sleep(0.5)
        logger.warning(f"Image {image_path} not found within {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"Error in wait_for_image: {str(e)}")
        return False

def click_on_image(image_path, image_key, confidence=None):
    try:
        # استفاده از confidence خاص اگه تعریف شده باشه
        conf = confidence if confidence is not None else CONFIG['image_configs'][image_key]['default_confidence']
        if wait_for_image(image_path, image_key):
            location = find_image(image_path, image_key)
            if location:
                center_x = location[0] + location[2] // 2
                center_y = location[1] + location[3] // 2
                pyautogui.click(center_x, center_y)
                logger.info(f"Clicked on image {image_path} at ({center_x}, {center_y})")
                return True
        logger.warning(f"Image {image_path} not found for click")
        return False
    except Exception as e:
        logger.error(f"Error in click_on_image: {str(e)}")
        return False

def click_relative_to_image(image_path, image_key, offset_x=0, offset_y=0):
    try:
        if wait_for_image(image_path, image_key):
            location = find_image(image_path, image_key)
            if location:
                center_x = location[0] + location[2] // 2
                center_y = location[1] + location[3] // 2
                target_x = center_x + int(offset_x * scale_factor)
                target_y = center_y + int(offset_y * scale_factor)
                if not (0 <= target_x < screen_width and 0 <= target_y < screen_height):
                    logger.error(f"Coordinates ({target_x}, {target_y}) are outside screen bounds")
                    return False
                pyautogui.click(target_x, target_y)
                logger.info(f"Clicked at ({target_x}, {target_y}) relative to image {image_path}")
                return True
        logger.warning(f"Image {image_path} not found for relative click")
        return False
    except Exception as e:
        logger.error(f"Error in click_relative_to_image: {str(e)}")
        return False

def move_relative_to_image(image_path, image_key, offset_x=0, offset_y=0):
    try:
        if wait_for_image(image_path, image_key):
            location = find_image(image_path, image_key)
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
        logger.warning(f"Image {image_path} not found for relative move")
        return False
    except Exception as e:
        logger.error(f"Error in move_relative_to_image: {str(e)}")
        return False

def paste_at_cursor():
    try:
        # تأخیر برای اطمینان از آماده بودن کلیپ‌بورد
        time.sleep(0.5)
        # لاگ کردن محتوای کلیپ‌بورد
        clipboard_content = pyperclip.paste()
        logger.info(f"Clipboard content before paste: {clipboard_content[:50]}...")
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.5)  # تأخیر بعد از جایگذاری
        logger.info("Pasted at current cursor position")
        return True
    except Exception as e:
        logger.error(f"Error in paste_at_cursor: {str(e)}")
        return False

def capture_image_offset(image_path, image_key, offset_x=0, offset_y=0, region_size=(20, 30)):
    try:
        if wait_for_image(image_path, image_key):
            location = find_image(image_path, image_key)
            if location:
                center_x = location[0] + location[2] // 2
                center_y = location[1] + location[3] // 2
                target_x = center_x + int(offset_x * scale_factor)
                target_y = center_y + int(offset_y * scale_factor)
                if not (0 <= target_x < screen_width and 0 <= target_y < screen_height):
                    logger.error(f"Coordinates ({target_x}, {target_y}) are outside screen bounds")
                    return None
                screenshot = pyautogui.screenshot(region=(target_x, target_y, *region_size))
                logger.info(f"Screenshot {region_size} captured at ({target_x}, {target_y})")
                return screenshot
        logger.warning(f"Image {image_path} not found for capture")
        return None
    except Exception as e:
        logger.error(f"Error in capture_image_offset: {str(e)}")
        return None

def find_closest_color(image, color_dict):
    try:
        if not image:
            return None
        pixels = list(image.getdata())
        color_count = {}
        for pixel in pixels:
            if pixel in color_count:
                color_count[pixel] += 1
            else:
                color_count[pixel] = 1
        most_common_color = max(color_count, key=color_count.get)
        min_total_distance = float('inf')
        closest_color_key = None
        for key, rgb in color_dict.items():
            total_distance = sum(abs(most_common_color[i] - rgb[i]) for i in range(3))
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                closest_color_key = key
        logger.info(f"Closest color found: {closest_color_key}")
        return closest_color_key
    except Exception as e:
        logger.error(f"Error in find_closest_color: {str(e)}")
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
        if not wait_for_image(resource_path(CONFIG['image_paths']['chatgpt']), 'chatgpt'):
            logger.error("Failed to load ChatGPT page")
            return

        if not click_on_image(resource_path(CONFIG['image_paths']['chatgpt']), 'chatgpt'):
            return
        # کلیک روی فیلد متنی ChatGPT
        if not click_relative_to_image(
            resource_path(CONFIG['image_paths']['tools']), 'tools',
            offset_y=CONFIG['offsets']['tools_y']
        ):
            return
        time.sleep(0.5)  # تأخیر برای فوکوس
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
                resource_path(CONFIG['image_paths']['msg']), 'msg',
                offset_x=CONFIG['offsets']['msg_x'],
                offset_y=CONFIG['offsets']['msg_y']
            )
            if captured_image:
                closest_color = find_closest_color(captured_image, CONFIG['color_dict'])
                if closest_color == "newmsg":
                    if not click_on_image(resource_path(CONFIG['image_paths']['instagram']), 'instagram'):
                        continue
                    if not move_relative_to_image(
                        resource_path(CONFIG['image_paths']['msg']), 'msg',
                        offset_x=CONFIG['offsets']['msg_relative_x'],
                        offset_y=CONFIG['offsets']['msg_relative_y']
                    ):
                        continue
                    if not click_on_image(resource_path(CONFIG['image_paths']['3dot']), '3dot'):
                        continue
                    if not click_on_image(resource_path(CONFIG['image_paths']['cpmsg']), 'cpmsg'):
                        continue
                    if not click_on_image(resource_path(CONFIG['image_paths']['chatgpt']), 'chatgpt'):
                        continue
                    # کلیک دوباره روی فیلد متنی ChatGPT
                    if not click_relative_to_image(
                        resource_path(CONFIG['image_paths']['tools']), 'tools',
                        offset_y=CONFIG['offsets']['tools_y']
                    ):
                        continue
                    time.sleep(0.5)  # تأخیر برای فوکوس
                    if not paste_at_cursor():
                        continue
                    if not click_on_image(resource_path(CONFIG['image_paths']['sendgpt']), 'sendgpt'):
                        continue
                    if wait_for_image(resource_path(CONFIG['image_paths']['cpgpt']), 'cpgpt', timeout=15):
                        for _ in range(1):
                            if not click_on_image(resource_path(CONFIG['image_paths']['cpgpt']), 'cpgpt', confidence=0.6):
                                break
                        if not click_on_image(resource_path(CONFIG['image_paths']['instagram']), 'instagram'):
                            continue
                        # کلیک روی فیلد متنی اینستاگرام
                        if not click_on_image(resource_path(CONFIG['image_paths']['msg']), 'msg', confidence=0.6):
                            continue
                        time.sleep(0.5)  # تأخیر برای فوکوس
                        if not paste_at_cursor():
                            continue
                        if not click_on_image(resource_path(CONFIG['image_paths']['sendinsta']), 'sendinsta'):
                            continue
            else:
                if not click_on_image(resource_path(CONFIG['image_paths']['instagram']), 'instagram'):
                    continue
            time.sleep(0.5)
        logger.info("Automation loop finished")
    except Exception as e:
        logger.error(f"Unexpected error in automation_loop: {str(e)}")
    finally:
        is_running = False
        stop_reason = stop_reason or "Loop completed"

def toggle_automation():
    global is_running, stop_reason
    try:
        if not is_running:
            dm_url = instagram_entry.get()
            prompt_text = prompt_entry.get()
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
        
        # بارگذاری تنظیمات مقیاس‌ها و confidence‌ها
        load_image_configs()
        
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
        root.title("Automation Controller")
        root.geometry("400x250")

        Label(root, text="Instagram DM URL:", font=("Arial", 10)).pack(pady=(10, 0))
        instagram_entry = Entry(root, width=50)
        instagram_entry.pack(pady=5)
        instagram_entry.insert(0, CONFIG['default_instagram_url'])

        Label(root, text="ChatGPT Prompt:", font=("Arial", 10)).pack(pady=(10, 0))
        prompt_entry = Entry(root, width=50)
        prompt_entry.pack(pady=5)
        prompt_entry.insert(0, CONFIG['default_prompt'])

        start_button = Button(root, text="Start Automation", command=toggle_automation, width=25, height=2)
        start_button.pack(pady=15)

        Button(root, text="Exit", command=root.quit, width=25, height=2).pack(pady=5)

        root.mainloop()
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
    finally:
        mouse_listener.stop()
        keyboard_listener.stop()
        save_image_configs()