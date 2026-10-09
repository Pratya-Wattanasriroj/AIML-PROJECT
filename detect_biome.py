import cv2
import mss
import numpy as np
import time
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import pythoncom
import win32com.client
import threading
import queue

# --- ระบบเสียง ---
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

# --- โหลดโมเดล ---
model = load_model('minecraft_biome_custom.h5')
with open('class_names.pkl', 'rb') as f: classes = pickle.load(f)

print("🌍 ระบบแยกไบโอม ทำงาน...")
memory_biome = None

with mss.MSS() as sct:
    monitor = sct.monitors[1]
    while True:
        frame = np.array(sct.grab(monitor))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        h, w, _ = frame_rgb.shape
        
        roi = frame_rgb[int(h * 0.4) : int(h * 0.85), 0 : w]
        arr = np.expand_dims(img_to_array(Image.fromarray(roi).resize((128, 128))), axis=0)
        preds = model.predict(arr, verbose=0)
        
        name = classes[np.argmax(preds[0])].lower()
        conf = np.max(preds[0])
        
        th = 0.85
        if name == "desert": th = 0.20
        elif name == "snowy_tundra": th = 0.99
        elif name == "taiga": th = 0.90
        
        if conf > th and name != memory_biome:
            display = name.replace('_', ' ').title()
            print(f"🌍 ไบโอม: {display} ({conf*100:.1f}%)")
            q.put(f"Entering {display}")
            memory_biome = name
            
        time.sleep(0.5)