import cv2
import mediapipe as mp
import numpy as np
import socket
import json
import time
import gc  # для очистки памяти вручную

# === Mediapipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === UDP Socket для Unreal ===
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UE_IP, UE_PORT = "127.0.0.1", 5000

# === Функция вычисления угла между 3 точками ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# === Преобразование координат Mediapipe → Unreal (в см, с центром в тазу) ===
def mediapipe_to_unreal_converter(lm):
    pelvis = np.array([
        (lm[23].x + lm[24].x) / 2,
        (lm[23].y + lm[24].y) / 2,
        (lm[23].z + lm[24].z) / 2
    ])
    scale = 100  # перевод нормализованных координат (0–1) в сантиметры

    def conv(p):
        x = (p.x - pelvis[0]) * scale
        y = -(p.y - pelvis[1]) * scale  # в Unreal Y вверх
        z = -(p.z - pelvis[2]) * scale  # глубина камеры — ось Z в Unreal
        return {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}
    return conv

# === Дефолтный пакет (если человек не найден) ===
def default_packet():
    return {
        "pose_detected": False,
        "angles": {
            "left_arm": 0.0,
            "right_arm": 0.0,
            "left_leg": 0.0,
            "right_leg": 0.0
        },
        "limb_points": {
            "left_arm": [{"x":0,"y":0,"z":0}]*3,
            "right_arm": [{"x":0,"y":0,"z":0}]*3,
            "left_leg": [{"x":0,"y":0,"z":0}]*3,
            "right_leg": [{"x":0,"y":0,"z":0}]*3
        }
    }

# === Захват с камеры ===
cap = cv2.VideoCapture(0)
print("Starting pose detection — sending Unreal-ready JSON data...")

frame_count = 0
start_time = time.time()

try:
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        data = default_packet()

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            conv = mediapipe_to_unreal_converter(lm)

            try:
                # === Расчет углов ===
                L_arm = calculate_angle((lm[11].x, lm[11].y), (lm[13].x, lm[13].y), (lm[15].x, lm[15].y))
                R_arm = calculate_angle((lm[12].x, lm[12].y), (lm[14].x, lm[14].y), (lm[16].x, lm[16].y))
                L_leg = calculate_angle((lm[23].x, lm[23].y), (lm[25].x, lm[25].y), (lm[27].x, lm[27].y))
                R_leg = calculate_angle((lm[24].x, lm[24].y), (lm[26].x, lm[26].y), (lm[28].x, lm[28].y))

                # === Формируем JSON ===
                data = {
                    "pose_detected": True,
                    "angles": {
                        "left_arm": round(L_arm, 2),
                        "right_arm": round(R_arm, 2),
                        "left_leg": round(L_leg, 2),
                        "right_leg": round(R_leg, 2)
                    },
                    "limb_points": {
                        "left_arm": [conv(lm[11]), conv(lm[13]), conv(lm[15])],
                        "right_arm": [conv(lm[12]), conv(lm[14]), conv(lm[16])],
                        "left_leg": [conv(lm[23]), conv(lm[25]), conv(lm[27])],
                        "right_leg": [conv(lm[24]), conv(lm[26]), conv(lm[28])]
                    }
                }

                # === Визуализация ===
                overlay = frame.copy()
                pts = [(int(l.x * w), int(l.y * h)) for l in lm]
                for a, b in [(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(24,26),(26,28)]:
                    cv2.line(overlay, pts[a], pts[b], (0,255,200), 3, cv2.LINE_AA)
                for i in [11,13,15,12,14,16,23,25,27,24,26,28]:
                    cv2.circle(overlay, pts[i], 5, (255,100,150), -1)
                frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

            except Exception as e:
                print(f"Angle calc error: {e}")

        # === Отправка JSON ===
        try:
            sock.sendto(json.dumps(data).encode('utf-8'), (UE_IP, UE_PORT))
        except Exception as e:
            print(f"UDP send error: {e}")

        # === FPS и очистка ===
        frame_count += 1
        if frame_count % 60 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"[INFO] FPS: {fps:.2f}")
            gc.collect()  # очистка памяти

        # === Отображаем окно ===
        cv2.imshow("Pose Detection — Unreal-ready JSON", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Лёгкое ограничение FPS
        time.sleep(0.02)

finally:
    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    pose.close()
    gc.collect()
    print("✅ Pose detection stopped, resources released.")

