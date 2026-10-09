from ultralytics import YOLO

def main():
    print("เริ่มต้นเทรน AI ตรวจจับสัตว์ในฟาร์ม (YOLOv8 Object Detection)...")
    
    # โหลดโมเดลสำหรับ "ตรวจจับกล่อง" (สังเกตว่าไม่มี -cls ต่อท้าย)
    model = YOLO('yolov8n.pt') 
    
    # เริ่มเทรนโดยดึงข้อมูลจากไฟล์ animal_data.yaml ที่คุณมีอยู่แล้ว
    results = model.train(
        data='animal_data.yaml',  
        epochs=50,                # 50 รอบกำลังดีสำหรับ 4 คลาส
        imgsz=640,                # ใช้ 640 ซึ่งเป็นขนาดมาตรฐานที่ YOLO ทำงานได้ดีที่สุดในการหากล่อง
        batch=16,           
        name='animal_detector_v1' # เซฟชื่อโฟลเดอร์โมเดล
    )
    
    print("\nเทรนสัตว์เสร็จสมบูรณ์!")
    print("ไฟล์โมเดลจะถูกบันทึกไว้ที่: runs/detect/animal_detector_v1/weights/best.pt")

if __name__ == '__main__':
    main()