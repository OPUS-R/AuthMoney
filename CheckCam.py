import cv2

cap = cv2.VideoCapture(0)  # カメラのインデックス（0 = 内蔵カメラ）

if not cap.isOpened():
    print("❌ カメラが開けませんでした！")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ フレームを取得できませんでした！")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
