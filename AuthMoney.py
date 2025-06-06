import cv2
import pytesseract
import numpy as np
import os

#OCRの設定
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows用
tess_config = "--psm 6"

#紙幣データベース
dataset_path = "./dataset"
banknote_images = {}
orb = cv2.ORB_create(nfeatures=500)

for value in [1000, 5000, 10000]:
    folder_path = os.path.join(dataset_path, f"{value}yen")
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg")]

    if image_files:
        img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        banknote_images[value] = (kp, des, img)

#カメラ起動
cap = cv2.VideoCapture(0)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #ORB特徴点検出
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    best_match_value = None
    best_match_count = 0
    best_good_matches = []

    for value, (kp, des, banknote_img) in banknote_images.items():
        if des is None or des_frame is None:
            continue

        matches = bf.knnMatch(des_frame, des, k=2)

        #Loweの比率テスト
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # RANSACを適用
        if len(good_matches) > best_match_count:
            src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            if len(src_pts) >= 4:  # RANSAC最低点
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    best_match_count = len(good_matches)
                    best_match_value = value
                    best_good_matches = good_matches

    #OCRで金額を認識
    detected_value_ocr = None
    roi = gray[50:200, 50:400]  # 画面の上部をOCR対象に
    text = pytesseract.image_to_string(roi, config=tess_config)

    for yen in [1000, 5000, 10000]:
        if str(yen) in text:
            detected_value_ocr = yen
            break

    # 結果を統合
    final_value = best_match_value if best_match_count > 10 else detected_value_ocr

    if final_value:
        cv2.putText(frame, f"Value: {final_value} Yen", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AMforCM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
