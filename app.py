import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import pytesseract
import subprocess
import os
import sys

# 💥 ตั้งค่า Path Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Minecraft AI Assistant", page_icon="🎮", layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #1E1E1E; color: white; }
    .stButton>button { background-color: #4ADE80; color: black; font-weight: bold; border-radius: 5px; border: none; }
    .stButton>button:hover { background-color: #2D5A34; color: white; }
    h1, h2, h3 { color: #4ADE80; }
    div.row-widget.stRadio > div { background-color: #2b2b2b; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🎮 Minecraft AI Assistant for Visually Impaired")
st.markdown("*ทลายข้อจำกัดการมองเห็น สู่โลกจินตนาการไร้ขอบเขต ด้วยเทคโนโลยี AI*")

tab1, tab2 = st.tabs(["🖼️ Web Demo (Test Image)", "🚀 Live Game Control (Run Locally)"])

with tab1:
    st.header("Web Demo: Test AI on Images")
    
    ai_category = st.radio(
        "เลือกหมวดหมู่ที่ต้องการทดสอบ:",
        ["⚔️ สแกนมอนสเตอร์/สัตว์ (YOLOv8)", "🌍 สแกนไบโอม (CNN)", "🧱 สแกนบล็อก (CNN)", "📖 สแกนชื่อไอเทม (OCR)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # 📌 ส่วนเลือกรูปตัวอย่าง หรือ อัพโหลดเอง
    sample_images_dir = "sample_images"
    image_options = ["อัพโหลดรูปภาพของคุณเอง..."]
    
    # ตรวจสอบว่ามีโฟลเดอร์ sample_images และดึงมาเฉพาะไฟล์ .png
    if os.path.exists(sample_images_dir):
        sample_files = [f for f in os.listdir(sample_images_dir) if f.lower().endswith('.png')]
        image_options.extend(sample_files)
        
    selected_image_option = st.selectbox("เลือกรูปภาพตัวอย่างเพื่อทดสอบ:", image_options)
    
    img_array = None
    image = None
    
    if selected_image_option == "อัพโหลดรูปภาพของคุณเอง...":
        uploaded_file = st.file_uploader("หรืออัพโหลดไฟล์ภาพ (PNG, JPG)", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
    else:
        # โหลดรูปภาพ .png จากโฟลเดอร์ sample_images
        img_path = os.path.join(sample_images_dir, selected_image_option)
        image = Image.open(img_path)
        img_array = np.array(image)
    
    if img_array is not None:
        # ป้องกันปัญหาไฟล์ PNG มี 4 Channels (RGBA) แปลงให้เหลือ 3 (RGB)
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ภาพต้นฉบับ")
            st.image(image, width='stretch')
            
        with col2:
            st.subheader("ผลลัพธ์จาก AI")
            btn_text = f"รันระบบ {ai_category.split(' ')[1]}"
            
            if st.button(btn_text, type="primary"): 
                with st.spinner("AI กำลังวิเคราะห์..."):
                    
                    if "มอนสเตอร์" in ai_category:
                        try:
                            model_mobs = YOLO(r'runs\detect\mob_radar\weights\best.pt')
                            model_animals = YOLO(r'runs\detect\animal_detector_v1-3\weights\best.pt')
                            
                            results_mobs = model_mobs(img_array)
                            results_animals = model_animals(img_array)
                            annotated_img = results_mobs[0].plot()
                            st.image(annotated_img, width='stretch')
                            
                            st.success("🎯 ผลการสแกนเป้าหมาย:")
                            for r in results_mobs[0].boxes:
                                name = model_mobs.names[int(r.cls)]
                                conf = float(r.conf)
                                st.write(f"- 🔴 **มอนสเตอร์:** {name} (มั่นใจ {conf:.2%})")
                            for r in results_animals[0].boxes:
                                name = model_animals.names[int(r.cls)]
                                conf = float(r.conf)
                                st.write(f"- 🟢 **สัตว์ฟาร์ม:** {name} (มั่นใจ {conf:.2%})")
                        except Exception as e:
                            st.error(f"ไม่สามารถรันโมเดล YOLO ได้: {e}")
                            
                    elif "ไบโอม" in ai_category:
                        try:
                            model = tf.keras.models.load_model('minecraft_biome_custom.h5')
                            with open('class_names.pkl', 'rb') as f:
                                import pickle
                                classes = pickle.load(f)
                            
                            img_resized = cv2.resize(img_array, (128, 128))
                            img_input = np.expand_dims(img_resized, axis=0)
                            
                            preds = model.predict(img_input)
                            top_class_id = np.argmax(preds[0])
                            name = classes[top_class_id].replace('_', ' ').title()
                            conf = preds[0][top_class_id]
                            
                            st.image(img_array, width='stretch')
                            st.success(f"🌍 ไบโอมที่ยืนอยู่: **{name}** (ความมั่นใจ {conf:.2%})")
                        except Exception as e:
                             st.error(f"ไม่สามารถรันโมเดล CNN ไบโอม ได้: {e}")
                             
                    elif "บล็อก" in ai_category:
                        try:
                            model = tf.keras.models.load_model("models/minecraft_cnn_best.keras")
                            with open("models/class_names.txt", "r") as f:
                                class_names = f.read().splitlines()
                                
                            img_resized = cv2.resize(img_array, (64, 64))
                            img_input = np.expand_dims(img_resized, axis=0)
                            
                            preds = model.predict(img_input)
                            top_class_id = np.argmax(preds[0])
                            name = class_names[top_class_id]
                            conf = preds[0][top_class_id]
                            
                            st.image(img_array, width='stretch')
                            st.success(f"🧱 บล็อกตรงหน้า: **{name}** (ความมั่นใจ {conf:.2%})")
                        except Exception as e:
                             st.error(f"ไม่สามารถรันโมเดล CNN บล็อก ได้: {e}")
                             
                    elif "ไอเทม" in ai_category:
                        try:
                            img_scaled = cv2.resize(img_array, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                            gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
                            
                            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                            thresh_inv = cv2.bitwise_not(thresh)
                            contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for cnt in contours:
                                x, y, w, h = cv2.boundingRect(cnt)
                                if w > 100 or h > 100:
                                    cv2.drawContours(thresh_inv, [cnt], -1, 0, -1)
                            
                            final_thresh = cv2.bitwise_not(thresh_inv)
                            custom_config = r'--psm 11 -c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :\'"'
                            text = pytesseract.image_to_string(final_thresh, config=custom_config)
                            
                            st.image(final_thresh, caption="ภาพที่ผ่านกระบวนการทำลายเส้นตาราง (Grid Destroyer)", width='stretch')
                            
                            clean_lines = []
                            for line in text.split('\n'):
                                clean_line = "".join([c for c in line if c.isalnum() or c.isspace() or c in ":'-+"]).strip()
                                if len(clean_line) > 2:
                                    clean_lines.append(clean_line)
                            final_text = " ".join(clean_lines)
                            
                            if final_text:
                                st.success("📝 ข้อความที่สกัดได้จากภาพ:")
                                st.code(final_text)
                            else:
                                st.warning("⚠ ไม่พบข้อความ หรือข้อความอ่านยากเกินไป")
                        except Exception as e:
                             st.error(f"OCR Error: {e}")

with tab2:
    st.header("🚀 Live Game Control (Run Scripts Locally)")
    st.write("ส่วนนี้สำหรับผู้ใช้งานที่รันแอปบนเครื่องตัวเอง เพื่อเปิดใช้งานระบบช่วยเหลือเข้าสู่เกมจริง")
    
    python_path = sys.executable 
    
    # 📌 1. สร้าง Session State เพื่อเก็บสถานะของ Process ที่กำลังรันอยู่เบื้องหลัง
    if 'processes' not in st.session_state:
        st.session_state.processes = {
            'radar': None,
            'biome': None,
            'block': None,
            'ocr': None
        }
        
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("⚔️ Radar & Auto-Combat")
        # ใช้ Toggle (สวิตช์) เพื่อสั่งเปิด/ปิด
        radar_on = st.toggle("เปิดใช้งาน Radar (detect_mobs.py)", key="tgl_radar")
        
        # ตรวจสอบการเปิด/ปิด
        if radar_on and st.session_state.processes['radar'] is None:
            st.session_state.processes['radar'] = subprocess.Popen([python_path, "detect_mobs.py"])
        elif not radar_on and st.session_state.processes['radar'] is not None:
            st.session_state.processes['radar'].terminate() # สั่งปิดสคริปต์
            st.session_state.processes['radar'] = None
            
        # แสดงสถานะ
        if st.session_state.processes['radar'] is not None:
            st.success("✅ กำลังรัน detect_mobs.py... (ทำงานอยู่เบื้องหลัง)")
            
            
        st.subheader("🌍 Biome Detection")
        biome_on = st.toggle("เปิดใช้งาน Biome Scanner (detect_biome.py)", key="tgl_biome")
        
        if biome_on and st.session_state.processes['biome'] is None:
            st.session_state.processes['biome'] = subprocess.Popen([python_path, "detect_biome.py"])
        elif not biome_on and st.session_state.processes['biome'] is not None:
            st.session_state.processes['biome'].terminate()
            st.session_state.processes['biome'] = None
            
        if st.session_state.processes['biome'] is not None:
            st.success("✅ กำลังรัน detect_biome.py... (ทำงานอยู่เบื้องหลัง)")
            
            
    with col_b:
        st.subheader("🧱 Block Scanner")
        block_on = st.toggle("เปิดใช้งาน Block Scanner (detect_blocks.py)", key="tgl_block")
        
        if block_on and st.session_state.processes['block'] is None:
            st.session_state.processes['block'] = subprocess.Popen([python_path, "detect_blocks.py"])
        elif not block_on and st.session_state.processes['block'] is not None:
            st.session_state.processes['block'].terminate()
            st.session_state.processes['block'] = None
            
        if st.session_state.processes['block'] is not None:
            st.success("✅ กำลังรัน detect_blocks.py... (ทำงานอยู่เบื้องหลัง)")
            
            
        st.subheader("📖 Smart OCR (Inventory)")
        ocr_on = st.toggle("เปิดใช้งาน OCR System (detect_items.py)", key="tgl_ocr")
        
        if ocr_on and st.session_state.processes['ocr'] is None:
            st.session_state.processes['ocr'] = subprocess.Popen([python_path, "detect_items.py"])
        elif not ocr_on and st.session_state.processes['ocr'] is not None:
            st.session_state.processes['ocr'].terminate()
            st.session_state.processes['ocr'] = None
            
        if st.session_state.processes['ocr'] is not None:
            st.success("✅ กำลังรัน detect_items.py... (ทำงานอยู่เบื้องหลัง)")

    st.markdown("---")
    st.warning("⚠️ **หมายเหตุ:** หากต้องการเลิกใช้งาน ควรมากดปิดสวิตช์ให้ครบทุกอันก่อนปิดหน้าต่างเว็บ เพื่อป้องกันไม่ให้สคริปต์ค้างอยู่ในระบบคอมพิวเตอร์ของคุณ")