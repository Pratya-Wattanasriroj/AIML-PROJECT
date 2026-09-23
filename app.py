import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import pickle

# --- ตั้งค่า API ฐานข้อมูลโภชนาการ (USDA FoodData Central) ---
USDA_API_KEY = "hZPwbL7V0i1G6oDk4BN3x6xqXDUaqVNIrBqRpgx8"

def get_nutrition(food_name):
    # 1. ค้นหาอาหารจากชื่อ
    search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search?api_key={USDA_API_KEY}&query={food_name}&pageSize=1"
    
    try:
        response = requests.get(search_url, timeout=5)
        data = response.json()
        
        # ถ้าระบบค้นหาเจอเมนูอาหาร
        if 'foods' in data and len(data['foods']) > 0:
            food_info = data['foods'][0]
            nutrients = food_info.get('foodNutrients', [])
            
            # ดึงค่าสารอาหาร (USDA ใช้รหัส ID สำหรับสารอาหารแต่ละตัว)
            # 1008 = Energy (kcal), 1003 = Protein, 1005 = Carbohydrate, 1004 = Total lipid (fat)
            nutrition_data = {
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "fat": 0
            }
            
            for nutrient in nutrients:
                if nutrient['nutrientNumber'] == '1008': # แคลอรี
                    nutrition_data["calories"] = round(nutrient.get('value', 0))
                elif nutrient['nutrientNumber'] == '1003': # โปรตีน
                    nutrition_data["protein"] = round(nutrient.get('value', 0), 1)
                elif nutrient['nutrientNumber'] == '1005': # คาร์บ
                    nutrition_data["carbs"] = round(nutrient.get('value', 0), 1)
                elif nutrient['nutrientNumber'] == '1004': # ไขมัน
                    nutrition_data["fat"] = round(nutrient.get('value', 0), 1)
                    
            return nutrition_data
            
    except Exception as e:
        print(f"API Error: {e}")
        pass
    
    return None

@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model('food_custom_cnn.h5')
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    return model, class_names

st.set_page_config(page_title="AI Food & Calorie Estimator", page_icon="🍲", layout="centered")
st.title("🍲 AI ตรวจจับเมนูและคำนวณแคลอรี (Custom CNN)")
st.write("อัปโหลดภาพอาหารเพื่อให้โครงข่ายประสาทเทียมวิเคราะห์ประเภทและคำนวณสารอาหารหลัก")

try:
    model, class_names = load_trained_model()
except Exception:
    st.error("ไม่พบไฟล์โมเดล กรุณารัน train_model.py ก่อนเริ่มใช้งานหน้าเว็บ")
    st.stop()

uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="ภาพที่ส่งเข้าตรวจ", use_container_width=True)

    with st.spinner("โครงข่ายประสาทเทียมกำลังประมวลผล..."):
        # ปรับขนาดภาพให้เท่ากับอินพุตของ Custom CNN (128x128)
        img_resized = image.resize((128, 128))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = predictions[0]
        confidence = float(np.max(score)) * 100
        predicted_class = class_names[np.argmax(score)]

        st.divider()

        if confidence < 50.0:
            st.warning(f"AI ยังไม่มั่นใจเพียงพอ (ค่าความเชื่อมั่น {confidence:.2f}%) แนะนำให้ใช้ภาพมุมตรงที่มีแสงชัดเจน")
        else:
            display_name = predicted_class.replace('_', ' ').title()
            st.success(f"**เมนูที่ทำนาย:** {display_name} (Confidence: {confidence:.2f}%)")
            
            with st.spinner("กำลังเชื่อมต่อฐานข้อมูลโภชนาการของรัฐบาลสหรัฐฯ (USDA)..."):
                nutrition = get_nutrition(display_name)

                if nutrition:
                    st.subheader("📊 ข้อมูลโภชนาการ (ต่อ 100 กรัม)")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🔥 พลังงาน", f"{nutrition['calories']} kcal")
                    c2.metric("🥩 โปรตีน", f"{nutrition['protein']} g")
                    c3.metric("🍚 คาร์โบไฮเดรต", f"{nutrition['carbs']} g")
                    c4.metric("🥓 ไขมัน", f"{nutrition['fat']} g")
                else:
                    st.info(f"จำแนกสำเร็จว่าเป็น {display_name} แต่ไม่พบข้อมูลสารอาหารใน API โภชนาการ")