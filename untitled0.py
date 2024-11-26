import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 데이터 경로 설정
base_dir = "C:\\Users\\cksrm\\Deeplunning\\cow_datasets\\"  # 데이터셋 폴더 경로 (수정 필요)

# 이미지 데이터 증강 및 로드
train_datagen = ImageDataGenerator(
    rescale=1.0/255,         # 픽셀 값을 [0,1]로 정규화
    rotation_range=20,       # 이미지 회전
    width_shift_range=0.2,   # 가로 이동
    height_shift_range=0.2,  # 세로 이동
    shear_range=0.2,         # 이미지 왜곡
    zoom_range=0.2,          # 확대/축소
    horizontal_flip=True,    # 좌우 반전
    validation_split=0.2     # 데이터의 20%를 검증 데이터로 사용
)

# 학습 데이터셋 로드
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),  # 이미지 크기
    batch_size=32,
    class_mode='categorical',  # 다중 클래스 분류
    subset='training'
)

# 검증 데이터셋 로드
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # 클래스 개수에 맞게 출력
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# 모델 저장
model.save("cow_marbling_model.h5")
print("Model saved as cow_marbling_model.h5")
