import cv2
import os
import pickle

# ORB特徴量抽出器
orb = cv2.ORB_create(nfeatures=1000)

# データセットフォルダ
dataset_path = "./processed_dataset"
features_db = {}

# 各紙幣画像の特徴量を抽出
for category in ["1000yen", "5000yen", "10000yen"]:
    folder_path = os.path.join(dataset_path, category)
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 特徴点と記述子を取得
            kp, des = orb.detectAndCompute(img, None)

            # 保存（特徴点の情報を辞書に追加）
            if des is not None:
                if category not in features_db:
                    features_db[category] = []
                features_db[category].append(des)

# 特徴データを保存（pickle形式）
with open("banknote_features.pkl", "wb") as f:
    pickle.dump(features_db, f)

print("✅ 特徴データベースを作成しました！")
