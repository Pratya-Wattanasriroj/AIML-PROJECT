import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, 
                                     Rescaling, BatchNormalization, GlobalAveragePooling2D,
                                     RandomFlip, RandomRotation, RandomZoom)
import pickle

IMG_SIZE = (128, 128)

# 1. โหลดข้อมูลตามปกติ
train_dataset = image_dataset_from_directory('dataset_custom', validation_split=0.2, subset="training", seed=123, image_size=IMG_SIZE, batch_size=32)
validation_dataset = image_dataset_from_directory('dataset_custom', validation_split=0.2, subset="validation", seed=123, image_size=IMG_SIZE, batch_size=32)

class_names = train_dataset.class_names
print(f"ไบโอมที่ AI ต้องเรียน: {class_names}")

with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

# 2. สร้างสถาปัตยกรรมที่แกะรอยมาจาก Pre-trained Model
model = Sequential([
    # --- ชั้นที่ 1: Data Augmentation (พลิกแพลงโจทย์) ---
    RandomFlip("horizontal", input_shape=(128, 128, 3)),
    
    # RandomRotation(0.1),
    # RandomZoom(0.1),
    
    Rescaling(1./255),

    # --- ชั้นที่ 2: สกัดจุดเด่น Block 1 ---
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(), # คุมสติ
    MaxPooling2D(2, 2),

    # --- ชั้นที่ 3: สกัดจุดเด่น Block 2 ---
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # --- ชั้นที่ 4: สกัดจุดเด่น Block 3 (มองลึกขึ้น) ---
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(), #Scaling data 
    MaxPooling2D(2, 2), # Avarage data แต่ละแผ่นแทนที่จะหาทั้งหมด

    # --- ชั้นที่ 5: Global Average Pooling ---
    GlobalAveragePooling2D(), 
    
    # --- ชั้นที่ 6: สมองส่วนตัดสินใจ ---
    Dense(128, activation='relu'),
    Dropout(0.5), # ป้องกันการท่องจำ
    Dense(len(class_names), activation='softmax')
])

model.summary()

# 3. เริ่มฝึกสอน
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("เริ่มฝึก AI ทายไบโอม (Custom CNN แบบอัปเกรด)...")
# เทรนจาก 0 ต้องใช้เวลาเรียนนานกว่า Pre-trained นิดนึงครับ ให้เวลาสัก 20 รอบ
model.fit(train_dataset, validation_data=validation_dataset, epochs=20)

model.save('minecraft_biome_custom.h5')
print("เสร็จสิ้น! บันทึกสมองเรียบร้อยในชื่อ minecraft_biome_custom.h5")


