#!/usr/bin/env python3
"""使用訓練好的模型，透過 webcam 即時偵測 bottle / candy，並傳送結果到 ESP32。"""

import platform
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import serial
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "best.pt"

CONF = 0.7

# ===== ESP32 Serial 設定 =====
# Mac 常見格式：
# /dev/cu.usbserial-xxxx
# /dev/cu.usbmodemxxxx
#
# 可以先用這個指令查：
# ls /dev/cu.*
SERIAL_PORT = "/dev/cu.usbserial-0001"
BAUD_RATE = 115200

# 每隔幾秒最多傳一次，避免 ESP32 被資料塞爆
SEND_INTERVAL = 0.5


def open_video_source(source) -> cv2.VideoCapture:
    """
    開啟攝影機來源。
    - 整數：本機攝影機 index（0=內建，1 常為 iPhone 連續互通相機等）
    - 字串：網路串流 URL（iPhone IP 攝影 App 的 http://... ）
    """
    if isinstance(source, int):
        if platform.system() == "Darwin":
            cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(source)

    if cap.isOpened():
        for _ in range(8):
            cap.read()
            time.sleep(0.05)

    return cap


def find_webcam(preferred_index: Optional[int] = None) -> Optional[cv2.VideoCapture]:
    indices = [preferred_index] if preferred_index is not None else list(range(3))
    for cam_id in indices:
        if cam_id is None:
            continue
        cap = open_video_source(cam_id)
        ok, frame = cap.read()

        if ok and frame is not None and frame.size > 0:
            print(f"已連接攝影機 index={cam_id}")
            return cap

        cap.release()

    return None


def parse_args(argv):
    """解析：python3 detect_webcam.py [模型路徑] [--camera N | --url URL]"""
    model_path = None
    camera_index = None
    stream_url = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--camera" and i + 1 < len(argv):
            camera_index = int(argv[i + 1])
            i += 2
        elif arg == "--url" and i + 1 < len(argv):
            stream_url = argv[i + 1]
            i += 2
        elif not arg.startswith("--") and model_path is None:
            model_path = Path(arg)
            i += 1
        else:
            i += 1
    return model_path, camera_index, stream_url


def open_serial() -> Optional[serial.Serial]:
    """開啟 ESP32 Serial。若失敗則回傳 None，讓程式仍可只做 webcam 偵測。"""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # ESP32 開啟 Serial 後通常會重啟，等它穩定
        print(f"已連接 ESP32：{SERIAL_PORT}，baud={BAUD_RATE}")
        return ser

    except serial.SerialException as e:
        print(f"警告：無法連接 ESP32 Serial：{SERIAL_PORT}")
        print(f"原因：{e}")
        print("程式會繼續執行 webcam 偵測，但不會傳送資料到 ESP32。")
        return None


def get_best_detection(results, model):
    """
    從 YOLO 結果中取出信心值最高的物件。
    回傳格式：(label, confidence)
    若沒有偵測到任何物件，回傳 ("none", 0.0)
    """
    best_label = "none"
    best_conf = 0.0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf > best_conf:
                best_label = label
                best_conf = conf

    return best_label, best_conf


def main() -> int:
    model_arg, camera_index, stream_url = parse_args(sys.argv)
    model_path = model_arg if model_arg is not None else DEFAULT_MODEL

    if not model_path.exists():
        print(f"找不到模型：{model_path}")
        print("請先執行：python3 train.py")
        return 1

    print(f"載入模型：{model_path}")
    model = YOLO(str(model_path))

    print(f"可辨識類別：{model.names}")
    print(f"信心門檻：{CONF}")
    print("即時辨識已啟動。按 q 離開。")

    if stream_url:
        print(f"使用網路串流：{stream_url}")
        cap = open_video_source(stream_url)
        if not cap.isOpened():
            cap.release()
            cap = None
    else:
        cap = find_webcam(camera_index)

    if cap is None:
        print("\n錯誤：無法從 webcam 讀取畫面。")
        print("請確認：")
        print("  1. 系統設定 → 隱私權與安全性 → 相機 → 允許 Terminal / Cursor / Python")
        print("  2. 沒有其他 App 佔用攝影機")
        print("  3. 可嘗試關閉 FaceTime、Zoom 等後再執行")
        print("  4. 使用 iPhone：見腳本說明，或 --camera 1 / --url http://...")
        return 1

    ser = open_serial()

    last_message = ""
    last_send_time = 0.0

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print("讀取畫面失敗，重試中…")
                time.sleep(0.1)
                continue

            results = model.predict(
                frame,
                conf=CONF,
                verbose=False,
                stream=False,
            )

            label, confidence = get_best_detection(results, model)

            # 傳給 ESP32 的格式：label,confidence
            # 例如：candy,0.86
            # 或：none,0.00
            message = f"{label},{confidence:.2f}\n"

            now = time.time()

            if ser is not None:
                # 只有訊息改變，或超過固定間隔，才傳送
                if message != last_message or now - last_send_time >= SEND_INTERVAL:
                    ser.write(message.encode("utf-8"))
                    print("Send to ESP32:", message.strip())

                    last_message = message
                    last_send_time = now

            annotated = results[0].plot()

            # 在畫面左上角也顯示目前要送給 ESP32 的結果
            cv2.putText(
                annotated,
                f"Send: {label} {confidence:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("YOLO Webcam - bottle / candy (按 q 離開)", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()

        if ser is not None:
            ser.close()

        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())