import cv2
import mediapipe as mp
import numpy as np
import socket
import json
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UE_IP, UE_PORT = "127.0.0.1", 6000

# Добавим таймаут для сокета
sock.settimeout(1.0)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: не удалось открыть вебкамеру.")
    exit(1)

limb_connections = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27), (27, 29),
    (24, 26), (26, 28), (28, 30)
]
display_points = [11, 13, 15, 12, 14, 16, 23, 24, 25, 26, 27, 28, 29, 30]

print("Starting pose detection...")
print(f"UDP target: {UE_IP}:{UE_PORT}")

frame_count = 0
last_sent_time = 0

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    frame_count += 1
    current_time = time.time()

    # Отправляем данные даже если поза не обнаружена (пустые данные)
    data_packet = {
        "timestamp": current_time,
        "frame_count": frame_count,
        "pose_detected": False,
        "angles": {
            "left_arm": 1.0,
            "right_arm": 2.0,
            "left_leg": 3.0,
            "right_leg": 4.0,
        },
        "joints": {}
    }

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Создаем словарь с координатами всех точек
        joints_data = {}
        for i, landmark in enumerate(lm):
            joints_data[str(i)] = {
                "x": float(landmark.x * w),
                "y": float(landmark.y * h),
                "z": float(landmark.z),
                "visibility": float(landmark.visibility)
            }

        # Рассчитываем углы только если нужные точки видны
        try:
            # Проверяем видимость точек перед расчетом углов
            if (lm[11].visibility > 0.5 and lm[13].visibility > 0.5 and lm[15].visibility > 0.5):
                L_arm = calculate_angle((lm[11].x, lm[11].y), (lm[13].x, lm[13].y), (lm[15].x, lm[15].y))
            else:
                L_arm = 0.0
            
            if (lm[12].visibility > 0.5 and lm[14].visibility > 0.5 and lm[16].visibility > 0.5):
                R_arm = calculate_angle((lm[12].x, lm[12].y), (lm[14].x, lm[14].y), (lm[16].x, lm[16].y))
            else:
                R_arm = 0.0
                
            if (lm[23].visibility > 0.5 and lm[25].visibility > 0.5 and lm[27].visibility > 0.5):
                L_leg = calculate_angle((lm[23].x, lm[23].y), (lm[25].x, lm[25].y), (lm[27].x, lm[27].y))
            else:
                L_leg = 0.0
                
            if (lm[24].visibility > 0.5 and lm[26].visibility > 0.5 and lm[28].visibility > 0.5):
                R_leg = calculate_angle((lm[24].x, lm[24].y), (lm[26].x, lm[26].y), (lm[28].x, lm[28].y))
            else:
                R_leg = 0.0

            data_packet.update({
                "pose_detected": True,
                "angles": {
                    "left_arm": float(L_arm),
                    "right_arm": float(R_arm),
                    "left_leg": float(L_leg),
                    "right_leg": float(R_leg),
                },
                "joints": joints_data
            })

        except Exception as e:
            print(f"Angle calculation error: {e}")

        # Визуализация
        overlay = frame.copy()
        pts = [(int(landmark.x * w), int(landmark.y * h)) for landmark in lm]

        for a, b in limb_connections:
            if a < len(pts) and b < len(pts):
                cv2.line(overlay, pts[a], pts[b], (0, 255, 200), 5, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        for idx in display_points:
            if idx < len(pts):
                cv2.circle(frame, pts[idx], 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pts[idx], 6, (255, 80, 120), -1, cv2.LINE_AA)

        # Отображаем углы
        cv2.putText(frame, f"L-arm {int(data_packet['angles']['left_arm'])}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 100, 150), 2)
        cv2.putText(frame, f"R-arm {int(data_packet['angles']['right_arm'])}", (30, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 150, 255), 2)
        cv2.putText(frame, f"L-leg {int(data_packet['angles']['left_leg'])}", (30, 110), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 150), 2)
        cv2.putText(frame, f"R-leg {int(data_packet['angles']['right_leg'])}", (30, 140), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 255, 200), 2)

    # Всегда отправляем данные (даже пустые)
    try:
        json_data = json.dumps(data_packet)
        sock.sendto(json_data.encode('utf-8'), (UE_IP, UE_PORT))
        
        # Логируем каждую 10-ю отправку чтобы не засорять консоль
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: Sent data to UE (pose detected: {data_packet['pose_detected']})")
            
    except Exception as e:
        print(f"[UDP ERROR] {e}")

    cv2.imshow("Pose Limbs Only", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
print("Program terminated")