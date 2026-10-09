import cv2
import mss
import numpy as np
import pytesseract
import time
import keyboard
import pyautogui
import pythoncom
import win32com.client

# 💥 ตั้งค่า Path Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ================= ระบบเสียง =================
pythoncom.CoInitialize()
spk = win32com.client.Dispatch("SAPI.SpVoice")
spk.Rate = 2

def speak_now(text):
    print(f"🗣️ ระบบพูดว่า: {text}")
    spk.Speak(text, 3) 

def clean_text(raw_text):
    lines = raw_text.split('\n')
    clean_lines = []
    for line in lines:
        clean_line = "".join([c for c in line if c.isalnum() or c.isspace() or c in ":'-+"]).strip()
        if len(clean_line) > 2:
            clean_lines.append(clean_line)
    return " ".join(clean_lines)

def process_hotbar_image(image):
    img_scaled = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    return thresh

def process_inv_image(image):
    """สูตรสำหรับตัวหนังสือในกระเป๋า + ระบบทำลายตาราง (Grid Destroyer)"""
    img_scaled = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # --- เริ่มระบบกรองตาราง ---
    # 1. กลับสีให้เส้นและตัวหนังสือเป็นสีขาว (เพื่อให้ OpenCV มองหาวัตถุเจอ)
    thresh_inv = cv2.bitwise_not(thresh)
    
    # 2. ค้นหารูปร่างทั้งหมดในภาพ (Contours)
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 3. ตัวอักษรปกติจะไม่กว้างหรือสูงเกิน 60-80 พิกเซล
        # ถ้ารูปร่างไหนกว้างหรือสูงเกิน 100 แปลว่าเป็น "เส้นตารางช่องเก็บของ" แน่นอน
        if w > 100 or h > 100:
            # ถมสีดำทับเส้นตารางทิ้งไปเลย
            cv2.drawContours(thresh_inv, [cnt], -1, 0, -1)
            
    # 4. กลับสีคืนให้ตัวหนังสือเป็นสีดำ พื้นขาว เหมือนเดิม
    final_thresh = cv2.bitwise_not(thresh_inv)
    return final_thresh

print("🔍 ระบบ OCR (เวอร์ชันลบตาราง Inventory) พร้อมใช้งาน!")
print("👉 สลับไอเทมบนมือ / ชี้ของใน Inventory แล้วกด 'X' / กด 'ESC' เพื่อออก")

last_hand_text = ""
last_inv_text = ""

config_hand = r'--psm 7 -c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :\'"'
config_inv = r'--psm 11 -c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :\'"'

with mss.MSS() as sct:
    monitor = sct.monitors[1]
    w, h = monitor["width"], monitor["height"]
    
    hotbar_zone = {
        "top": int(h * 0.70),   
        "left": int(w * 0.20),
        "width": int(w * 0.60),
        "height": int(h * 0.15)
    }

    while True:
        # ================= 1. โหมดสลับของในมือ =================
        img_hand = np.array(sct.grab(hotbar_zone))
        processed_hand = process_hotbar_image(img_hand)
        
        raw_text = pytesseract.image_to_string(processed_hand, config=config_hand)
        current_text = clean_text(raw_text)
        
        if current_text and current_text != last_hand_text:
            speak_now(current_text)
            last_hand_text = current_text
            cv2.imwrite("debug_ocr_hand.jpg", processed_hand)
            
        if not current_text:
            last_hand_text = ""

        # ================= 2. โหมดอ่านในกระเป๋า (กด X) =================
        if keyboard.is_pressed('x'):
            mx, my = pyautogui.position()
            
            inv_zone = {
                "top": max(0, my - 60), 
                "left": min(w, mx + 15),
                "width": min(w - (mx + 15), 600),  
                "height": min(h - (my - 60), 500)
            }

            img_inv = np.array(sct.grab(inv_zone))
            processed_inv = process_inv_image(img_inv)
            
            # บังคับเซฟภาพและโชว์ Live Debug
            cv2.imwrite("debug_ocr_inventory.jpg", processed_inv)
            cv2.imshow("AI Vision (Press X to scan)", processed_inv)
            cv2.waitKey(1) 
            
            raw_inv_text = pytesseract.image_to_string(processed_inv, config=config_inv)
            inv_text = clean_text(raw_inv_text)
            
            if inv_text and inv_text != last_inv_text:
                speak_now(inv_text)
                last_inv_text = inv_text
                
            time.sleep(0.2)
        else:
            last_inv_text = "" 
            try:
                cv2.destroyWindow("AI Vision (Press X to scan)")
            except:
                pass

        time.sleep(0.05)
        
        if keyboard.is_pressed('f2'):
            print("ปิดระบบ OCR")
            break