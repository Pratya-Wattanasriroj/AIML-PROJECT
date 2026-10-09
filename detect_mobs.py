import cv2
import mss
import numpy as np
import time
from ultralytics import YOLO
import pythoncom
import win32com.client
import threading
import queue
import ctypes
import math
import keyboard

# ================= การตั้งค่า Aimbot =================
SENSITIVITY_MULTIPLIER = 0.4  
AIMBOT_HOTKEY = 'v'           
# 💥 เพิ่มเกณฑ์ความมั่นใจขั้นต่ำ (แนะนำ 0.70 - 0.75 ถ้ายิ่งล็อกมั่ว ให้ปรับตัวเลขนี้สูงขึ้น)
MIN_CONFIDENCE = 0.70  

def move_mouse_relative(x, y):
    ctypes.windll.user32.mouse_event(0x0001, int(x), int(y), 0, 0)

def auto_attack():
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.02) 
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)

# ================= ระบบเสียง (SAPI5) =================
q = queue.Queue()
def tts():
    pythoncom.CoInitialize()
    spk = win32com.client.Dispatch("SAPI.SpVoice")
    spk.Rate = 2
    while True:
        txt = q.get()
        if txt: spk.Speak(txt)
        q.task_done()
threading.Thread(target=tts, daemon=True).start()

# ================= โหลดโมเดล =================
print("⏳ กำลังโหลดสมอง AI ทั้ง 2 ระบบ...")
model_animals = YOLO(r'runs\detect\animal_detector_v1-3\weights\best.pt')
model_mobs = YOLO(r'runs\detect\mob_radar\weights\best.pt')

print(f"🚨 ระบบฟาร์ม & เรดาร์ (โหมดป้องกันล็อกอากาศ) ทำงาน!")
print(f"👉 กดค้าง '{AIMBOT_HOTKEY}' เพื่อล็อกเป้า (ESC เพื่อออก)")

last_alert = 0
last_attack = 0 

with mss.mss() as sct:
    monitor = sct.monitors[1] 
    while True:
        frame = np.array(sct.grab(monitor))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        h, w, _ = frame_rgb.shape
        
        SCREEN_CENTER_X = w // 2
        SCREEN_CENTER_Y = h // 2
        
        # ค้นหาด้วยเกณฑ์เริ่มต้นที่ 0.50 ไปก่อน แล้วเดี๋ยวเรามากรองแบบละเอียดข้างในอีกที
        results_mobs = model_mobs.predict(frame_rgb, conf=0.50, imgsz=416, verbose=False)[0]
        results_animals = model_animals.predict(frame_rgb, conf=0.50, imgsz=416, verbose=False)[0]
        
        threats = []
        closest_target = None
        min_distance = float('inf')

        cv2.circle(frame_rgb, (SCREEN_CENTER_X, SCREEN_CENTER_Y), 5, (0, 0, 255), -1)

        # --- 1. สแกนหามอนสเตอร์ ---
        if len(results_mobs.boxes) > 0:
            for box in results_mobs.boxes:
                conf = float(box.conf[0])
                if conf < MIN_CONFIDENCE: continue 
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_w, box_h = x2 - x1, y2 - y1
                
                # กรองขนาดกล่อง (ป้องกันล็อกเป้าอากาศ/เศษดิน)
                if box_w < 20 or box_h < 20 or box_w > w * 0.8 or box_h > h * 0.8:
                    continue

                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                name = model_mobs.names[int(box.cls[0])]
                
                dir_txt = "left" if xc < w*0.35 else "right" if xc > w*0.65 else "ahead"
                threats.append((f"Warning, {name}", dir_txt)) 
                
                dist = math.hypot(xc - SCREEN_CENTER_X, yc - SCREEN_CENTER_Y)
                if (dist - 150) < min_distance:
                    min_distance = dist - 150
                    closest_target = (xc, yc)
                    
                cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame_rgb, f"{name} {conf*100:.0f}%", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- 2. สแกนหาสัตว์ในฟาร์ม ---
        if len(results_animals.boxes) > 0:
            for box in results_animals.boxes:
                conf = float(box.conf[0])
                if conf < MIN_CONFIDENCE: continue 

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_w, box_h = x2 - x1, y2 - y1
                
               
                if box_w < 20 or box_h < 20 or box_w > w * 0.8 or box_h > h * 0.8:
                    continue
                    
                # กรองตัดมือผู้เล่นออก 
                # (ถ้าขอบล่างของกล่อง y2 ทะลุเกิน 85% ของความสูงหน้าจอ ถือว่าเป็นมือเรา)
                if y2 > h * 0.85:
                    continue

                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                name = model_animals.names[int(box.cls[0])]
                
                dir_txt = "left" if xc < w*0.35 else "right" if xc > w*0.65 else "ahead"
                threats.append((name, dir_txt))
                
                dist = math.hypot(xc - SCREEN_CENTER_X, yc - SCREEN_CENTER_Y)
                if dist < min_distance:
                    min_distance = dist
                    closest_target = (xc, yc)
                    
                cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"{name} {conf*100:.0f}%", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- แจ้งเตือนด้วยเสียง ---
        if threats and (time.time() - last_alert) > 5.0:
            alert_msg = f"{threats[0][0]} {threats[0][1]}" 
            q.put(alert_msg)
            last_alert = time.time()
            
        # --- Aimbot & โจมตี ---
        if keyboard.is_pressed(AIMBOT_HOTKEY) and closest_target is not None:
            target_x, target_y = closest_target
            delta_x = target_x - SCREEN_CENTER_X
            delta_y = target_y - SCREEN_CENTER_Y
            
            move_mouse_relative(delta_x * SENSITIVITY_MULTIPLIER, delta_y * SENSITIVITY_MULTIPLIER)
            
            current_time = time.time()
            if current_time - last_attack > 0.4:
                auto_attack()
                last_attack = current_time
            
        cv2.imwrite("debug_full_radar.jpg", frame_rgb)
        
        if keyboard.is_pressed('f2'):
            print("ปิดระบบ")
            break