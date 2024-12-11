import os
import argparse
import json
import requests
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import sys
import random
from yolov5 import YOLOv5

# 파서 설정
parser = argparse.ArgumentParser(description='Download images and annotations for TACO dataset')
parser.add_argument('--dataset_path', required=False, default='./data/annotations.json', help='Path to annotations')
parser.add_argument('--output_path', required=False, default='./dataset/yolo', help='Path to save YOLO formatted dataset')
parser.add_argument('--sample_size', type=int, default=1, help='Number of random images to process')  # 랜덤 이미지 수 설정
parser.add_argument('--resume', action='store_true', help='Resume from last successful download')  # 이어서 다운로드 옵션
args = parser.parse_args()

dataset_dir = os.path.dirname(args.dataset_path)
output_dir = args.output_path
sample_size = args.sample_size  # 랜덤 이미지 수
resume = args.resume  # 이어서 다운로드 여부

# YOLO 모델 로딩
model = YOLOv5("yolov5s.pt")  # yolov5s 모델을 사용하는 경우. 모델 파일 경로는 적절하게 변경해야 함.
model.names = ['Plastic', 'Metal', 'Paper', 'Glass', 'Cardboard', 'Clothing', 'Electronics', 'Batteries', 'Tires', 'beverage can', 'Furniture', 'Toys', 'Books', 'Appliances', 'Paint', 'Tools', 'Tanks', 'Tubs', 'Trash Bags', 'Plastic Bottles', 'Glass Bottles', 'Soda Cans', 'Food Wrappers', 'Plastic Lids', 'Paper Bags', 'Plastic Cups', 'Plastic Wrap', 'Egg Cartons', 'Aluminum Foil', 'Food Packaging', 'Plastic Containers', 'Plastic Utensils', 'Plastic Straws', 'Plastic Trays', 'Glass Jars', 'Plastic Jugs', 'Food Cans', 'Plastic Cutlery', 'Plastic Plates', 'Plastic Bags', 'Yogurt Containers', 'Plastic Tubs', 'Take-out Containers', 'Paper Plates', 'Toothpaste Tubes', 'Plastic Packaging', 'Styrofoam', 'Cigarette Butts', 'Cigarette Pack', 'Cardboard Boxes', 'Toothbrush', 'Plastic Clamshells', 'Ketchup Bottles', 'Wine Bottles', 'Cosmetic Bottles', 'Beer Bottles', 'Milk Cartons', 'Pill Bottles', 'Tetra Paks', 'Glass Jars', 'Laminated Packaging', 'Empty Bottles', 'Plastic Food Packaging', 'Plastic Bags', 'Plastic Lids', 'Cigarette Pack', 'Takeout Cups', 'Pizza Boxes', 'Pet Bottles', 'Plastic Film', 'Polystyrene Packaging', 'Cups', 'Soda Bottles', 'Coffee Cups']

# Create output directories
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

# 이미지 다운로드 완료 목록
downloaded_images_file = os.path.join(output_dir, 'downloaded_images.txt')

# 이전에 다운로드한 이미지 목록을 로드 (있다면)
if resume and os.path.exists(downloaded_images_file):
    with open(downloaded_images_file, 'r') as f:
        downloaded_images = set(f.read().splitlines())
else:
    downloaded_images = set()

# Load annotations
with open(args.dataset_path, 'r') as f:
    annotations = json.loads(f.read())

    nr_images = len(annotations['images'])

    # 랜덤하게 선택한 이미지의 인덱스
    random_indices = random.sample(range(nr_images), min(sample_size, nr_images))  # 최대 sample_size 개수만큼 랜덤으로 선택

    for i in random_indices:
        image = annotations['images'][i]

        file_name = image['file_name']
        if file_name in downloaded_images:
            print(f"Skipping {file_name}, already downloaded.")
            continue

        url_original = image['flickr_url']
        url_resized = image['flickr_640_url']

        file_path = os.path.join(output_dir, 'images', file_name)

        # Create subdir if necessary
        subdir = os.path.dirname(file_path)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        if not os.path.isfile(file_path):
            # Load and Save Image
            response = requests.get(url_original)
            img = Image.open(BytesIO(response.content))

            # Save image
            if img._getexif():
                img.save(file_path, exif=img.info["exif"])
            else:
                img.save(file_path)

        # Save YOLO format annotations
        annotations_file = os.path.join(output_dir, 'labels', file_name.replace('.jpg', '.txt'))  # Assuming images are .jpg

        # Ensure the directory for labels exists
        label_dir = os.path.dirname(annotations_file)
        os.makedirs(label_dir, exist_ok=True)  # Create the label directory if it doesn't exist

        # Get the annotations for this image
        image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image['id']]

        with open(annotations_file, 'w') as label_file:
            for ann in image_annotations:
                # YOLO format: class_id x_center y_center width height
                bbox = ann['bbox']  # [x, y, width, height]
                x_center = (bbox[0] + bbox[2] / 2) / image['width']
                y_center = (bbox[1] + bbox[3] / 2) / image['height']
                width = bbox[2] / image['width']
                height = bbox[3] / image['height']

                # Get class_id
                class_id = ann['category_id']  # Assuming category_id directly maps to YOLO class id
                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        # 이미지 파일을 OpenCV로 로드하여 화면에 표시
        img_cv = cv2.imread(file_path)
        
        # YOLOv5 모델로 객체 인식
        results = model.predict(img_cv)  # 모델 예측

        # 결과에서 감지된 객체를 표시
        for det in results.xyxy[0]:  # xyxy 좌표 결과
            x1, y1, x2, y2, conf, cls = map(int, det[:6])  # 좌표와 클래스 추출
            class_name = model.names[cls] if cls < len(model.names) else f"class{cls}"  # 이름 확인
            label = f"{class_name} {conf:.2f}"  # 클래스 이름과 확률
            
            # 경계 상자와 라벨 그리기
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 경계 상자
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 이미지 축소
        img_resized = cv2.resize(img_cv, (0, 0), fx=0.25, fy=0.25)

        # 결과 이미지 표시
        cv2.imshow('Image', img_resized)  # 축소된 이미지 출력
        cv2.waitKey(0)  # 키가 눌리면 종료
        cv2.destroyAllWindows()

        # 기록된 이미지를 다운로드 완료 목록에 추가
        with open(downloaded_images_file, 'a') as f:
            f.write(file_name + '\n')

        # 진행 상태 표시
        bar_size = 30
        progress = int(bar_size * (i + 1) / nr_images)
        sys.stdout.write(f"[{'#' * progress}{' ' * (bar_size - progress)}] {i + 1}/{nr_images}\r")
        sys.stdout.flush()

