"""
layout_detector.py
──────────────────
Определяет где вебка и где игровая область на экране стримера.

Алгоритм:
  1. Вытащить кадр из середины VOD
  2. Найти лицо через MediaPipe Face Detection
  3. «Примагнитить» найденную область к ближайшему углу
     (вебки почти всегда стоят в углу)
  4. Игровая зона = всё остальное
  5. Показать превью и спросить подтверждение
  6. Сохранить в layout_config.json
"""

import os
import sys
import json
import subprocess
import numpy as np

# ── опциональные импорты ────────────────────────────────────────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    HAS_MP = True
except ImportError:
    HAS_MP = False


# ════════════════════════════════════════════════════════════════
#  ИЗВЛЕЧЕНИЕ КАДРА
# ════════════════════════════════════════════════════════════════

def extract_frame_ffmpeg(video_file: str, timestamp: int) -> np.ndarray | None:
    """
    Вытаскивает один кадр через ffmpeg (не требует OpenCV).
    Возвращает numpy array (H, W, 3) или None.
    """
    tmp = "_preview_frame.png"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(timestamp),
        "-i", video_file,
        "-frames:v", "1",
        tmp,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not os.path.exists(tmp):
        return None

    if HAS_CV2:
        frame = cv2.imread(tmp)
        os.remove(tmp)
        return frame
    # без OpenCV — вернём путь к файлу
    return tmp


# ════════════════════════════════════════════════════════════════
#  ДЕТЕКЦИЯ ЛИЦА
# ════════════════════════════════════════════════════════════════

def detect_face_mediapipe(frame: np.ndarray) -> dict | None:
    """
    Ищет лицо через MediaPipe.
    Возвращает словарь {x, y, w, h} в пикселях или None.
    """
    if not HAS_MP or not HAS_CV2:
        return None

    mp_face = mp.solutions.face_detection
    h, w = frame.shape[:2]

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as det:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = det.process(rgb)

    if not res.detections:
        return None

    # Берём детекцию с максимальным score
    best = max(res.detections, key=lambda d: d.score[0])
    bb = best.location_data.relative_bounding_box

    # Добавляем отступ вокруг лица (вебка всегда больше лица)
    PAD = 0.8
    rx = max(0.0, bb.xmin - bb.width  * PAD)
    ry = max(0.0, bb.ymin - bb.height * PAD)
    rw = min(1.0 - rx, bb.width  * (1 + 2 * PAD))
    rh = min(1.0 - ry, bb.height * (1 + 2 * PAD))

    return {
        "x": int(rx * w),
        "y": int(ry * h),
        "w": int(rw * w),
        "h": int(rh * h),
        "frame_w": w,
        "frame_h": h,
    }


def detect_face_opencv(frame: np.ndarray) -> dict | None:
    """Запасной вариант — каскады Хаара (встроены в OpenCV)."""
    if not HAS_CV2:
        return None

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    h_frame, w_frame = frame.shape[:2]
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Расширяем под полноразмерную вебку
    PAD = 0.8
    nx = max(0, int(x - w * PAD))
    ny = max(0, int(y - h * PAD))
    nw = min(w_frame - nx, int(w * (1 + 2 * PAD)))
    nh = min(h_frame - ny, int(h * (1 + 2 * PAD)))

    return {"x": nx, "y": ny, "w": nw, "h": nh, "frame_w": w_frame, "frame_h": h_frame}


# ════════════════════════════════════════════════════════════════
#  ПРИВЯЗКА К УГЛУ
# ════════════════════════════════════════════════════════════════

def snap_to_corner(cam: dict) -> dict:
    """
    Привязывает область вебки к ближайшему углу кадра.
    Так границы получаются чёткими и клип выглядит аккуратно.
    """
    fw, fh = cam["frame_w"], cam["frame_h"]
    cx = cam["x"] + cam["w"] / 2
    cy = cam["y"] + cam["h"] / 2

    snapped = dict(cam)
    snapped["x"] = 0          if cx < fw / 2 else fw - cam["w"]
    snapped["y"] = 0          if cy < fh / 2 else fh - cam["h"]
    return snapped


# ════════════════════════════════════════════════════════════════
#  ВЫЧИСЛЕНИЕ ИГРОВОЙ ЗОНЫ
# ════════════════════════════════════════════════════════════════

def calc_game_zone(cam: dict) -> dict:
    """
    Возвращает игровую зону — часть кадра без вебки.
    Стратегия: убираем полосу, в которой находится вебка.
    """
    fw, fh = cam["frame_w"], cam["frame_h"]

    # Вебка в верхней половине → игра снизу
    if cam["y"] < fh / 2:
        return {"x": 0, "y": cam["h"], "w": fw, "h": fh - cam["h"], "frame_w": fw, "frame_h": fh}
    # Вебка в нижней половине → игра сверху
    else:
        return {"x": 0, "y": 0, "w": fw, "h": fh - cam["h"], "frame_w": fw, "frame_h": fh}


# ════════════════════════════════════════════════════════════════
#  ПРЕВЬЮ
# ════════════════════════════════════════════════════════════════

def save_preview(frame: np.ndarray, cam: dict, game: dict, path: str = "layout_preview.jpg"):
    """Рисует найденные зоны на кадре и сохраняет как превью."""
    if not HAS_CV2:
        return

    preview = frame.copy()
    # Игровая зона — синий
    cv2.rectangle(preview,
                  (game["x"], game["y"]),
                  (game["x"] + game["w"], game["y"] + game["h"]),
                  (255, 100, 0), 4)
    cv2.putText(preview, "GAME", (game["x"] + 20, game["y"] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0), 4)

    # Вебка — зелёный
    cv2.rectangle(preview,
                  (cam["x"], cam["y"]),
                  (cam["x"] + cam["w"], cam["y"] + cam["h"]),
                  (0, 220, 80), 4)
    cv2.putText(preview, "CAM", (cam["x"] + 20, cam["y"] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 220, 80), 4)

    cv2.imwrite(path, preview)
    print(f"  📸 Превью сохранено: {path}")


# ════════════════════════════════════════════════════════════════
#  РУЧНАЯ НАСТРОЙКА (запасной вариант)
# ════════════════════════════════════════════════════════════════

def manual_setup(frame_w: int, frame_h: int) -> dict:
    """
    Интерактивная настройка: пользователь вводит координаты руками.
    Вызывается если автодетекция не нашла лицо.
    """
    print("\n  Автоопределение не сработало. Введи координаты вебки вручную.")
    print(f"  Размер кадра: {frame_w} × {frame_h} пикселей\n")
    print("  Типичные значения для угловой вебки 300×300:")
    print("    Нижний правый угол: x=980, y=780, w=300, h=300")
    print("    Нижний левый угол:  x=0,   y=780, w=300, h=300\n")

    def ask_int(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return int(val) if val.isdigit() else default

    x = ask_int("x (левый край вебки)", frame_w - 320)
    y = ask_int("y (верхний край вебки)", frame_h - 320)
    w = ask_int("w (ширина вебки)", 300)
    h = ask_int("h (высота вебки)", 300)

    return {"x": x, "y": y, "w": w, "h": h, "frame_w": frame_w, "frame_h": frame_h}


# ════════════════════════════════════════════════════════════════
#  ГЛАВНАЯ ФУНКЦИЯ
# ════════════════════════════════════════════════════════════════

def detect_or_load_layout(video_file: str, config_file: str, vod_duration: int) -> dict:
    """
    Основная точка входа.
    - Если config_file существует → загружает и возвращает
    - Иначе → детектирует, показывает превью, спрашивает подтверждение, сохраняет
    """

    # ── уже настроено? ──────────────────────────────────────────
    if os.path.exists(config_file):
        with open(config_file) as f:
            layout = json.load(f)
        print(f"  ✅ Layout загружен из {config_file}")
        print(f"     Вебка:  x={layout['webcam']['x']} y={layout['webcam']['y']} "
              f"w={layout['webcam']['w']} h={layout['webcam']['h']}")
        print(f"     Игра:   x={layout['game']['x']}   y={layout['game']['y']}   "
              f"w={layout['game']['w']} h={layout['game']['h']}")
        return layout

    print("\n🎥 Первый запуск — определяем расположение вебки...")

    if not HAS_CV2:
        print("  ⚠️  OpenCV не установлен → пропускаем детекцию, используем blur-режим")
        return {"mode": "blur"}

    # ── вытаскиваем кадр из середины VOD ───────────────────────
    mid = vod_duration // 2
    frame = extract_frame_ffmpeg(video_file, mid)
    if frame is None or isinstance(frame, str):
        print("  ⚠️  Не удалось извлечь кадр → режим blur")
        return {"mode": "blur"}

    h_frame, w_frame = frame.shape[:2]

    # ── детектируем лицо ────────────────────────────────────────
    cam = None
    if HAS_MP:
        cam = detect_face_mediapipe(frame)
        if cam:
            print("  ✅ Лицо найдено (MediaPipe)")
    if cam is None:
        cam = detect_face_opencv(frame)
        if cam:
            print("  ✅ Лицо найдено (OpenCV каскады)")

    if cam is None:
        print("  ❓ Лицо не найдено автоматически")
        cam = manual_setup(w_frame, h_frame)

    # ── привязываем к углу ──────────────────────────────────────
    cam = snap_to_corner(cam)

    # ── считаем игровую зону ────────────────────────────────────
    game = calc_game_zone(cam)

    # ── превью ──────────────────────────────────────────────────
    save_preview(frame, cam, game)

    print(f"\n  Найдено:")
    print(f"    Вебка: x={cam['x']} y={cam['y']} w={cam['w']} h={cam['h']}")
    print(f"    Игра:  x={game['x']} y={game['y']} w={game['w']} h={game['h']}")
    print(f"\n  Открой layout_preview.jpg и проверь разметку.")

    confirm = input("  Всё верно? (y / n / skip для режима blur): ").strip().lower()

    if confirm == "skip":
        return {"mode": "blur"}

    if confirm != "y":
        print("  Введи координаты вручную:")
        cam  = manual_setup(w_frame, h_frame)
        game = calc_game_zone(cam)

    # ── сохраняем ───────────────────────────────────────────────
    layout = {
        "mode":   "smart",
        "webcam": {"x": cam["x"],  "y": cam["y"],  "w": cam["w"],  "h": cam["h"]},
        "game":   {"x": game["x"], "y": game["y"], "w": game["w"], "h": game["h"]},
        "frame":  {"w": w_frame,   "h": h_frame},
    }
    with open(config_file, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"\n  💾 Layout сохранён в {config_file}")
    print("  (Чтобы переопределить — удали файл и запусти снова)\n")

    return layout
