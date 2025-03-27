# -*- coding: utf-8 -*-
from tkinter import *   # Tk, Label, Button, Entry, Canvas, filedialog
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk
import sys
import io
from ultralytics import YOLO
import cv2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
def clear():
    print("正在清空...")
    canvas.delete('all')
    global N
    N = np.zeros((288, 288), dtype=np.uint8)

def recognite():
    print("正在识别...")
    global N
    im = Image.fromarray(N, mode='L').convert('L')
    # 反转图像颜色
    im = ImageOps.invert(im)
    # 保存画布内容为图片
    im.save('canvas_image.png')

    # 
    im_new = im.resize((28, 28))
    im_new_np = np.array(im_new).astype(np.float32) / 255.0
    im_new_np = (im_new_np * 255).astype(np.uint8)
    im_new_np = cv2.cvtColor(im_new_np, cv2.COLOR_GRAY2BGR)

    global model
    results = model([im_new_np])

    # 显示预测结果
    predictions = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            cls = int(box.cls[0])
            conf = box.conf[0]
            print(f'class: {cls}, confidence: {conf}, box: {r}')
            predictions.append(f'预测为：{cls}')

    # 将预测结果显示在标签上
    var.set('\n'.join(predictions))

def paint(event):
    x, y = event.x, event.y
    
    # 设置线条宽度
    line_width = 19  # 你可以根据需要调整这个值

    # 绘制当前点
    canvas.create_oval(x - line_width // 2, y - line_width // 2, x + line_width // 2, y + line_width // 2, fill='white', outline='white')

    # 更新 N 数组中的像素值
    for i in range(x - line_width // 2, x + line_width // 2 + 1):
        for j in range(y - line_width // 2, y + line_width // 2 + 1):
            if 0 <= i < 288 and 0 <= j < 288:
                N[j, i] = 255

def select_file():
    global file_path
    file_path = filedialog.askopenfilename(
        title="选择文件",
        filetypes=(("视频文件", "*.mp4 *.avi"), ("图片文件", "*.jpg *.jpeg *.png"))
    )
    if file_path:
        print(f"选择的文件路径: {file_path}")

def select_model_file():
    global model_path
    model_path = filedialog.askopenfilename(
        title="选择文件",
        filetypes=(("模型文件", "*.pt"),)
    )
    global model
    if model_path:
        print(f"选择的文件路径: {model_path}")
        model = YOLO(model_path)

def image_detect():
    print("正在图片检测...")
    if not file_path:
        print("请先选择一个图片文件。")
        return
    
    image = cv2.imread(file_path)
    if image is None:
        print("Error: Could not open image.")
        return
    
    # 调整图像大小为 416x416
    image_resized = cv2.resize(image, (416, 416))

    global model
    results = model(image_resized)

     # 显示预测结果
    predictions = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            cls = int(box.cls[0])
            conf = box.conf[0]
            print(f'class: {cls}, confidence: {conf}, box: {r}')
            predictions.append(f'预测为：{cls}')
    
    # 将预测结果显示在标签上
    var.set('\n'.join(predictions))

    # 绘制预测结果图
    image_with_boxes = image.copy()
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            cls = int(box.cls[0])
            conf = box.conf[0]
            cv2.rectangle(image_with_boxes, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, f'{cls}', (r[0], r[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 显示处理后的图像
    image_with_boxes = cv2.resize(image_with_boxes, (416, 416))
    cv2.imshow('Image Detection', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def video_detect():
    print("正在视频检测...")
    if not file_path:
        print("请先选择一个视频文件。")
        return
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0])
                conf = box.conf[0]
                cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                cv2.putText(frame, f'{cls}', (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Video Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def camera_detect():
    print("正在实时摄像头检测...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0])
                conf = box.conf[0]
                cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                cv2.putText(frame, f'{cls} {conf:.2f}', (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Camera Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    N = np.zeros((288, 288), dtype=np.uint8)
    root = Tk()
    root.geometry('800x600+500+200')
    root.title('手写数字识别系统')
    lbl_title = Label(root, text='手写数字识别系统V2.0')
    lbl_title.grid(row=0, column=0, columnspan=3)
    var = StringVar()
    txt_result = Entry(root, textvariable=var, bg='green')
    txt_result.grid(row=1, column=0, columnspan=3)
    btn_clear = Button(root, text='清空', command=clear)
    btn_clear.grid(row=2, column=0)
    btn_recognite = Button(root, text='识别', command=recognite)
    btn_recognite.grid(row=2, column=2)
    btn_select_file = Button(root, text='选择视频或图片文件', command=select_file)
    btn_select_file.grid(row=2, column=1)
    btn_select_model_file = Button(root, text='选择模型文件', command=select_model_file)
    btn_select_model_file.grid(row=2, column=4)
    btn_video_detect = Button(root, text='视频检测', command=video_detect)
    btn_video_detect.grid(row=4, column=0)
    btn_image_detect = Button(root, text='图片检测', command=image_detect)
    btn_image_detect.grid(row=4, column=1)
    btn_camera_detect = Button(root, text='实时摄像头检测', command=camera_detect)
    btn_camera_detect.grid(row=4, column=2)
    canvas = Canvas(root, width=288, height=288, bg='black')
    canvas.bind('<B1-Motion>', paint)
    canvas.grid(row=3, column=0, columnspan=3)

    root.mainloop()