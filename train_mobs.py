from ultralytics import YOLO

def main():
    # 1. โหลดโมเดล YOLOv8 รุ่นเล็กสุดและเร็วสุด (Nano)
    model = YOLO('yolov8n.pt') 
    
    print(" เริ่มเทรน AI เรดาร์จับมอนสเตอร์...")
    
    # 2. เริ่มสอน AI โดยชี้ไปที่ไฟล์ data.yaml ที่อยู่ระดับเดียวกัน
    results = model.train(
        data='data.yaml',  # แก้ให้ชี้ไปที่ไฟล์ data.yaml ตรงๆ 
        epochs=30,         # จำนวนรอบที่เทรน (เริ่มต้นที่ 30 รอบ)
        imgsz=640,         # ขนาดภาพมาตรฐาน YOLO
        batch=16,          # ประมวลผลทีละ 16 รูป
        name='mob_radar'   # ชื่อโฟลเดอร์เซฟผลลัพธ์
    )
    
    print("เทรนเสร็จสิ้น! ไฟล์สมองของคุณ (best.pt) อยู่ในโฟลเดอร์ runs/detect/mob_radar/weights/")

if __name__ == '__main__':
    main()
    