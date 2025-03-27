from ultralytics import YOLO
def train_and_evaluate():
    #加载预训练模型
    model = YOLO("yolov8s.pt")
    #训练模型
    results = model.train(data="shouxiedataset\data.yaml", epochs=300, imgsz=416, batch=32, device=0, workers=0,amp = True)
if __name__ == '__main__':
    train_and_evaluate()