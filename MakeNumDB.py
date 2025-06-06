import cv2
import os

# 紙幣画像フォルダ
dataset_path = "./dataset"
template_output_path = "./templates"

os.makedirs(template_output_path, exist_ok=True)

for value in [1000, 5000, 10000]:
    img_path = f"{dataset_path}/{value}yen/{value}_1.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # OCR対象のROI指定
    roi = img[80:130, 50:200]  # 位置を微調整
    cv2.imwrite(f"{template_output_path}/{value}.jpg", roi)

print("✅ テンプレート画像を作成しました！")
