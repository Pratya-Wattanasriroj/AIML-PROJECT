import os
import tensorflow as tf
from tensorflow.keras import layers, models

def main():
    dataset_dir = "dataset_real_game"
    img_height, img_width = 64, 64
    batch_size = 16

    # 1. โหลดข้อมูลและแบ่ง Train/Val (80/20) อัตโนมัติจากโฟลเดอร์
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"🗂️ พบทั้งหมด {num_classes} คลาส: {class_names}")

    # ปรับแต่งประสิทธิภาพการโหลดข้อมูลเข้า RAM
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. สร้างโครงสร้าง Custom CNN
    # เพิ่มความหลากหลายให้ภาพจำลอง (แก้ Overfitting และมุมมองที่ต่างกัน)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = models.Sequential([
        layers.InputLayer(input_shape=(img_height, img_width, 3)),
        data_augmentation,
        layers.Rescaling(1./255), # แปลงค่าสีจาก 0-255 ให้อยู่ในช่วง 0-1
        
        # ดึงจุดเด่นของภาพ (ลวดลายและสี)
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # ยุบข้อมูลและทำนายผล
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # ปิดการทำงานเซลล์สมอง 50% ชั่วคราวเพื่อบังคับให้ไม่จำข้อสอบ
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.summary()

    # 3. ตั้งค่า Callback เพื่อเซฟเฉพาะโมเดลรอบที่ฉลาดที่สุด
    os.makedirs("models", exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/minecraft_cnn_best.keras",
        save_best_only=True,
        monitor="val_accuracy"
    )
    
    # ถ้าความแม่นยำไม่เพิ่มขึ้น 10 รอบติด ให้หยุดเทรนทันที
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True
    )

    # 4. เริ่มเทรน 50 รอบ
    print("\nเริ่มต้นเทรน Custom CNN...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[checkpoint, early_stop]
    )

    # บันทึกรายชื่อคลาสเก็บไว้ให้สคริปต์สแกนดึงไปใช้พูด
    with open("models/class_names.txt", "w") as f:
        f.write("\n".join(class_names))
        
    print("\n🎉 เทรนเสร็จสมบูรณ์! โมเดลถูกบันทึกไว้ที่ 'models/minecraft_cnn_best.keras'")

if __name__ == '__main__':
    main()