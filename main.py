import os
import cv2
import json
from pathlib import Path

# 세그멘테이션 기준으로 이미지 잘라내기
def cropBySeg(image, point, padd=5):
    ylist, xlist = [], []  # y좌표 리스트, x좌표 리스트
    for x, y in point:
        ylist.append(int(y))
        xlist.append(int(x))

    # 잘라낼 이미지 좌표리스트, 처음과 마지막 포인트
    crop_y, crop_x = [min(ylist), max(ylist)], [min(xlist), max(xlist)]

    # 잘라낼 이미지의 가로 세로 길이
    w, h = crop_x[1] - crop_x[0], crop_y[1] - crop_y[0]

    # 이미지 잘라내기
    cropped_image = image[crop_y[0] - padd:crop_y[1] + padd, crop_x[0] - padd:crop_x[1] + padd]
    return cropped_image, w, h

# 이미지를 세그멘트대로 잘라낸 다음, 세로 이미지로 변경
def preprocessing(imagepath, imagename, jsonpath, jsonname, saved_dir, pad=10):
    # json 파일 읽어들이기
    with open(f"{jsonpath}{jsonname}.json", 'rt') as f:
        json_data = json.load(f)
    
    # 세그멘테이션 정보
    points = json_data['label_info']['shapes'][0]['points']
    
    # 이미지 읽기
    image = cv2.imread(f'{imagepath}{imagename}.jpg', cv2.IMREAD_COLOR)
    
    # 이미지 잘라내기
    cropped_image, w, h = cropBySeg(image, points, padd=pad)

    if w > h:  # 가로 이미지일 경우
        # 90도 회전시키기
        cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
    
    # 바탕화면에 저장할 경로
    desktop_path = str(Path.home() / "C:\\Users\\cksrm\\deeplunning\\cow_11_new\\")  # 바탕화면 경로
    save_path = os.path.join(desktop_path, f"{imagename}_crop.jpg")  # 파일 저장 경로

    # 잘라낸 (그리고 회전시킨) 이미지 저장
    cv2.imwrite(save_path, cropped_image)
    # 잘라낸 이미지 보여주기 + 없애도 됨
    
    return 0

# 경로 설정
impath = "C:\\Users\\cksrm\\deeplunning\\cow_image_11\\"  # * 소고기 이미지 파일 경로
jpath = "C:\\Users\\cksrm\\deeplunning\\cow_laber_11\\"  # * 소고기 이미지 좌표 경로

# 여러 파일 처리
for i in range(1, 80000):  # * 1부터 N까지의 이미지 번호
    file_number = f"{i:06d}"  # 파일 번호를 6자리 형식으로 만듦
    imname = f"QC_cow_segmentation_1++_{file_number}"  # * 이미지 이름 확인하고 실행
    jname = f"QC_cow_segmentation_1++_{file_number}"  # * 이미지 좌표 이름 확인하고 실행
    
    try:
        # 이미지와 JSON 파일을 처리
        preprocessing(impath, imname, jpath, jname, saved_dir="", pad=10)
    except Exception as e:
        print(f"Error processing file {file_number}: {e}")
