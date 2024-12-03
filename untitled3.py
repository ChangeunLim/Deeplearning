import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 모델 로드
model = load_model("cow_marbling_model.h5")

# 카메라 캡처 설정
camera = cv2.VideoCapture(0)  # 기본 카메라 (ID가 0) 사용

if not camera.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("카메라가 열렸습니다. 'q'를 눌러 종료하세요.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break
    
    # 프레임 표시
    cv2.imshow("Camera Feed", frame)
    
    # 'c'를 누르면 이미지 캡처 및 모델 예측
    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        # 이미지 전처리 (128x128로 리사이즈 및 모델 입력 형식에 맞추기)
        resized_frame = cv2.resize(frame, (128, 128))
        img_array = np.expand_dims(resized_frame / 255.0, axis=0)
        
        # 모델 예측
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        print(f"Predicted class: {class_idx}")
        print(f"Predicted probabilities: {predictions}")
        
        # 등급 출력
        if class_idx == 2:
            print('3등급')
        elif class_idx == 0:
            print('1등급')
        else:
            print('1++등급')

    # 'q'를 누르면 종료
    if key & 0xFF == ord('q'):
        break

# 자원 해제
camera.release()
cv2.destroyAllWindows()
