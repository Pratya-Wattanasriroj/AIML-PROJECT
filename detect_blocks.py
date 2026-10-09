import cv2
import mss
import numpy as np
import keyboard
import tensorflow as tf
import time
import pythoncom
import win32com.client
import threading
import queue

# --- 1. ระบบเสียงสุดเสถียร (จากโค้ดไบโอม) ---
q = queue.Queue()
def tts():
    pythoncom.CoInitialize()
    spk = win32com.client.Dispatch("SAPI.SpVoice")
    spk.Rate = 2 # ปรับความเร็วเสียง (1-10)
    while True:
        txt = q.get()
        if txt: 
            spk.Speak(txt)
        q.task_done()

# สั่งรันเสียงพูดเบื้องหลัง
threading.Thread(target=tts, daemon=True).start()

# --- 2. โหลดสมอง AI (CNN) ---
model = tf.keras.models.load_model("models/minecraft_cnn_best.keras")
with open("models/class_names.txt", "r") as f:
    class_names = f.read().splitlines()

print("🟢 ระบบ AI สแกนบล็อก พร้อมทำงาน!")
print("👉 กดปุ่ม 'C' เพื่อสแกน (กด ESC เพื่อออก)")

# ตั้งค่าจุดครอปภาพ 64x64 ใต้เป้ากากบาท
center_x, center_y = 1920 // 2, 1080 // 2
monitor = {
    "top": center_y + 20, 
    "left": center_x - 32,
    "width": 64,
    "height": 64
}

with mss.mss() as sct:
    while True:
        if keyboard.is_pressed('c'):
            # แคปจอและแปลงสี
            screenshot = np.array(sct.grab(monitor))
            frame_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
            
            # ทายผล
            img_array = tf.expand_dims(frame_rgb, 0)
            predictions = model.predict(img_array, verbose=0)
            score = predictions[0]
            class_id = np.argmax(score)
            conf = score[class_id]
            
            name = class_names[class_id]

            if conf > 0.60: 
                print(f"✅ ตรวจพบ: {name} (มั่นใจ {conf*100:.1f}%)")
                
                # โยนชื่อบล็อกลงคิวให้ระบบเสียงทำงาน
                q.put(name)
            else:
                print(f"❌ ไม่แน่ใจ (เดาว่า {name} แต่มั่นใจแค่ {conf*100:.1f}%)")

            cv2.imwrite("debug_cnn_vision.jpg", cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR))
            
            # ดักรอจนกว่าจะปล่อยปุ่ม C
            while keyboard.is_pressed('c'):
                time.sleep(0.05)

        if keyboard.is_pressed('f2'):
            print("🔴 ปิดระบบสแกน")
            break