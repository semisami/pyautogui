from PIL import Image
import pyautogui
import time
import sys
import os

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    full_path = os.path.join(base_path, relative_path)
    print(f"Attempting to access file: {full_path}")
    if not os.path.exists(full_path):
        print(f"Error: File {full_path} does not exist")
    return full_path

def capture_image_offset(image_path, offset_x=0, offset_y=0, confidence=0.8):
    location = find_image(image_path, confidence)
    if location:
        center_x, center_y = pyautogui.center(location)
        target_x = int(center_x + offset_x)
        target_y = int(center_y + offset_y)
        if target_x < 0 or target_y < 0:
            print(f"Error: Coordinates ({target_x}, {target_y}) are outside the screen")
            return None
        screenshot = pyautogui.screenshot(region=(target_x, target_y, 20, 30))
        pil_image = screenshot
        print(f"Screenshot 20x30 captured at ({target_x}, {target_y})")
        return pil_image
    else:
        print(f"Image {image_path} not found")
        return None

def find_image(image_path, confidence=0.8):
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} does not exist")
            return None
        location = pyautogui.locateOnScreen(image_path, confidence=confidence)
        if location:
            print(f"Image {image_path} found at {location}")
        return location
    except pyautogui.ImageNotFoundException:
        print(f"Image {image_path} not found on screen")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def move_above_image(image_path, offset=10, confidence=0.8):
    location = find_image(image_path, confidence)
    if location:
        x, y = pyautogui.center(location)
        pyautogui.moveTo(x, y - offset)
        print(f"Mouse moved to {x}, {y - offset} (10 pixels above image)")
    else:
        print(f"Image {image_path} not found")

def click_on_image(image_path, confidence=0.8):
    location = find_image(image_path, confidence)
    if location:
        pyautogui.click(location)
        print(f"Clicked on image {image_path}")
    else:
        print(f"Image {image_path} not found")

def wait_for_image(image_path, timeout=10, confidence=0.8):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if find_image(image_path, confidence):
            print(f"Image {image_path} found within {timeout} seconds")
            return True
        time.sleep(0.5)
    print(f"Image {image_path} not found within {timeout} seconds")
    return False

def paste_at_cursor():
    pyautogui.hotkey('ctrl', 'v')
    print("Pasted at current cursor position")

def move_relative_to_image(image_path, offset_x=0, offset_y=0, confidence=0.8):
    location = find_image(image_path, confidence)
    if location:
        center_x, center_y = pyautogui.center(location)
        target_x = center_x + offset_x
        target_y = center_y + offset_y
        pyautogui.moveTo(target_x, target_y)
        print(f"Mouse moved to ({target_x}, {target_y})")
    else:
        print(f"Image {image_path} not found")

def check_image_in_relative_region(main_image_path, target_image_path, region_offset_x=0, region_offset_y=0, region_width=100, region_height=100, confidence=0.8):
    main_location = find_image(main_image_path, confidence)
    if main_location:
        center_x, center_y = pyautogui.center(main_location)
        region_left = center_x + region_offset_x
        region_top = center_y + region_offset_y
        region = (region_left, region_top, region_width, region_height)
        try:
            target_location = pyautogui.locateOnScreen(target_image_path, region=region, confidence=confidence)
            if target_location:
                print(f"Image {target_image_path} found in relative region")
                return True
            else:
                print(f"Image {target_image_path} not found in relative region")
                return False
        except pyautogui.ImageNotFoundException:
            return False
    else:
        print(f"Main image {main_image_path} not found")
        return False

def find_closest_color(image, color_dict):
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
    return closest_color_key

if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    assets_url = "assets/"
    color_dict = {
        "newmsg": (38, 38, 38),
        "black": (0, 0, 0),
        "typing": (48, 48, 48)
    }
    for i in range(100):
        captured_image = capture_image_offset(resource_path("assets/msg.png"), offset_x=-40, offset_y=-75, confidence=0.9)
        if captured_image:
            closest_color = find_closest_color(captured_image, color_dict)
            print(f"Closest color: {closest_color}")
            # captured_image.save("captured_region.png")
            # print("Image saved as 'captured_region.png'")
            if closest_color == "newmsg":
                move_relative_to_image(resource_path("assets/msg.png"), offset_x=-30, offset_y=-50, confidence=0.8)
                click_on_image(resource_path("assets/3dot.png"), confidence=0.9)
                time.sleep(1)
                click_on_image(resource_path("assets/cpmsg.png"), confidence=0.9)
                time.sleep(1)
                # click_on_image(resource_path("assets/chatgpt.png"), confidence=0.8)
                click_on_image(resource_path("assets/gptmsg.png"), confidence=0.9)
                paste_at_cursor()
                click_on_image(resource_path("assets/sendgpt.png"), confidence=0.9)
                time.sleep(3)
                if wait_for_image(resource_path("assets/cpgpt.png"), timeout=15, confidence=0.8):
                    click_on_image(resource_path("assets/cpgpt.png"), confidence=0.9)
                    click_on_image(resource_path("assets/cpgpt.png"), confidence=0.9)
                    click_on_image(resource_path("assets/cpgpt.png"), confidence=0.9)
                    click_on_image(resource_path("assets/msg.png"), confidence=0.8)
                    paste_at_cursor()
                    click_on_image(resource_path("assets/sendinsta.png"), confidence=0.9)
        time.sleep(2)
    input("Press Enter to exit...")