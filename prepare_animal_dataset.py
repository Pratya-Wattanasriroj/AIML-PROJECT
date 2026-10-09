import os
import shutil
import pandas as pd

def main():
    # ตรวจสอบชื่อไฟล์ให้ตรงกับของคุณ
    img_source_dir = 'animalimages'      # โฟลเดอร์รูปดิบของคุณ
    boxes_csv = 'animalboxes.csv'        # ไฟล์ CSV กล่อง (หรือ boxes.csv)
    frames_csv = 'animalframes.csv'      # ไฟล์ CSV เฟรม (หรือ frames.csv)
    
    yolo_dir = 'animal_yolo_dataset'
    img_train_dir = os.path.join(yolo_dir, 'images/train')
    lbl_train_dir = os.path.join(yolo_dir, 'labels/train')

    print("⏳ กำลังสร้างโฟลเดอร์ YOLO...")
    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)

    # 1. โหลดข้อมูลจาก CSV
    print("⏳ กำลังโหลดและคัดกรองข้อมูลจาก CSV...")
    boxes_df = pd.read_csv(boxes_csv)
    frames_df = pd.read_csv(frames_csv)

    # 2. นำข้อมูลมาเชื่อมกันด้วยชื่อเฟรม
    df = pd.merge(boxes_df, frames_df, on='frame', how='inner')

    # 3. คัดกรองเฉพาะ 4 สัตว์เป้าหมาย
    target_mobs = ['chicken', 'cow', 'pig', 'sheep']
    df_filtered = df[df['mob'].isin(target_mobs)].copy()

    # 4. เปลี่ยนรหัสคลาสให้ตรงกับ YAML (0=Chicken, 1=Cow, 2=Pig, 3=Sheep)
    class_mapping = {'chicken': 0, 'cow': 1, 'pig': 2, 'sheep': 3}
    df_filtered['yolo_class'] = df_filtered['mob'].map(class_mapping)

    # 5. สร้างไฟล์ .txt และก๊อปปี้รูปภาพ
    grouped = df_filtered.groupby('frame')
    print(f"🎯 เจอภาพสัตว์เป้าหมายทั้งหมด {len(grouped)} รูป กำลังจัดเตรียมไฟล์...")
    
    count = 0
    for frame_name, group in grouped:
        img_filename = f"{frame_name}.jpg"
        src_img_path = os.path.join(img_source_dir, img_filename)
        
        # เช็คว่ามีรูปในโฟลเดอร์จริงๆ
        if not os.path.exists(src_img_path):
            continue
            
        # 5.1 ก๊อปปี้รูปลงโฟลเดอร์ images/train
        dst_img_path = os.path.join(img_train_dir, img_filename)
        shutil.copy2(src_img_path, dst_img_path)
        
        # 5.2 สร้างไฟล์ .txt ลงโฟลเดอร์ labels/train
        txt_filename = f"{frame_name}.txt"
        txt_path = os.path.join(lbl_train_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for _, row in group.iterrows():
                # ฟอร์แมต YOLO: class_id cx cy w h
                line = f"{row['yolo_class']} {row['cx']:.6f} {row['cy']:.6f} {row['w']:.6f} {row['h']:.6f}\n"
                f.write(line)
        count += 1
        
    print(f"\n✅ เสร็จสมบูรณ์! จัดเตรียมไปทั้งหมด {count} รูป")
    print("✨ ตอนนี้โฟลเดอร์ 'animal_yolo_dataset' พร้อมใช้งานแล้ว!")

if __name__ == '__main__':
    main()