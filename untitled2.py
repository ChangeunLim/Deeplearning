from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("cow_marbling_model.h5")

img_path = "C:\\Users\\cksrm\\Deeplunning\\cow_datasets\\cow_3\\QC_cow_segmentation_3_000831_crop.jpg"  # 테스트할 이미지 경로
img = image.load_img(img_path, target_size=(128, 128))
img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])
print(f"Predicted class: {class_idx}")