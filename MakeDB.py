import cv2
import os

#データセットフォルダ
dataset_path = "./dataset"
output_path = "./processed_dataset"

# 出力フォルダを作成
os.makedirs(output_path, exist_ok=True)

# 各紙幣のフォルダを処理
for category in ["1000yen", "5000yen", "10000yen"]:
    input_folder = os.path.join(dataset_path, category)
    output_folder = os.path.join(output_path, category)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".jpg"):
            img_path = os.path.join(input_folder, file)
            img = cv2.imread(img_path)

            # 📌 画像の前処理
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレースケール化
            img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)  # ノイズ除去
            img_edges = cv2.Canny(img_blur, 50, 150)  # エッジ強調

            # 保存
            output_file = os.path.join(output_folder, file)
            cv2.imwrite(output_file, img_edges)

print("✅ 画像前処理完了！")