import os
import shutil
import random

# 원본 이미지와 라벨 경로
image_dir = './dataset/images'  # 원본 이미지 디렉토리 (배치 폴더 포함)
label_dir = './dataset/labels'  # 원본 라벨 디렉토리 (배치 폴더 포함)

# 나눌 경로
train_image_dir = './dataset/images/train'  # 훈련용 이미지 경로
val_image_dir = './dataset/images/val'      # 검증용 이미지 경로
train_label_dir = './dataset/labels/train'  # 훈련용 라벨 경로
val_label_dir = './dataset/labels/val'      # 검증용 라벨 경로

# 폴더 생성
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 배치 폴더에서 이미지 파일 리스트 가져오기
image_batches = os.listdir(image_dir)

# 모든 배치 폴더에서 이미지와 라벨 파일 이동
for batch in image_batches:
    batch_image_dir = os.path.join(image_dir, batch)
    batch_label_dir = os.path.join(label_dir, batch)

    # 배치가 디렉토리인 경우
    if os.path.isdir(batch_image_dir):
        images = os.listdir(batch_image_dir)  # 이미지 파일 리스트 가져오기
        random.shuffle(images)  # 이미지 랜덤 셔플

        # train: 80%, val: 20%로 분리
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # 훈련용 이미지와 라벨 이동
        for img_file in train_images:
            img_path = os.path.join(batch_image_dir, img_file)
            label_path = os.path.join(batch_label_dir, img_file.replace('.jpg', '.txt'))  # 라벨 파일은 .txt 확장자

            # 파일이 이미 존재하면 건너뛰기
            if not os.path.exists(os.path.join(train_image_dir, img_file)):
                shutil.copy(img_path, train_image_dir)  # 훈련용 이미지 복사
            if not os.path.exists(os.path.join(train_label_dir, img_file.replace('.jpg', '.txt'))):
                shutil.copy(label_path, train_label_dir)  # 훈련용 라벨 복사

        # 검증용 이미지와 라벨 이동
        for img_file in val_images:
            img_path = os.path.join(batch_image_dir, img_file)
            label_path = os.path.join(batch_label_dir, img_file.replace('.jpg', '.txt'))  # 라벨 파일은 .txt 확장자

            # 파일이 이미 존재하면 건너뛰기
            if not os.path.exists(os.path.join(val_image_dir, img_file)):
                shutil.copy(img_path, val_image_dir)  # 검증용 이미지 복사
            if not os.path.exists(os.path.join(val_label_dir, img_file.replace('.jpg', '.txt'))):
                shutil.copy(label_path, val_label_dir)  # 검증용 라벨 복사

print("데이터셋 분할 및 이동 완료")
