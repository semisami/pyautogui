import ctypes
import win32api

# گرفتن هندل پنجره فعال
hwnd = win32api.GetForegroundWindow()

# گرفتن layout کیبورد
thread_id = ctypes.windll.user32.GetWindowThreadProcessId(hwnd, 0)
layout_id = ctypes.windll.user32.GetKeyboardLayout(thread_id)

# استخراج زبان (زبان انگلیسی = 0x409)
lang_id = layout_id & 0xFFFF
print(hex(lang_id))
