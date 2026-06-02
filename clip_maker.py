"""
clip_maker.py
─────────────
Нарезает клипы и конвертирует в вертикальный формат 1080×1920.

Два режима:
  smart  — игра сверху + вебка снизу (знает координаты обеих зон)
  blur   — фон из размытого оригинала (запасной вариант)
"""

import os
import subprocess
from datetime import timedelta


TARGET_W = 1080
TARGET_H = 1920


# ════════════════════════════════════════════════════════════════
#  РЕЖИМ: SMART (вебка + игра по отдельности)
# ════════════════════════════════════════════════════════════════

def _smart_vf(layout: dict, webcam_share: float = 0.33) -> str:
    """
    Строит ffmpeg -vf строку для смарт-монтажа:
      ┌───────────────┐
      │               │ ← игра (67% высоты)
      │    GAMEPLAY   │
      │               │
      ├───────────────┤
      │   WEBCAM  👤  │ ← вебка (33% высоты)
      └───────────────┘
    """
    cam   = layout["webcam"]
    game  = layout["game"]

    cam_h  = int(TARGET_H * webcam_share)           # 634 px
    game_h = TARGET_H - cam_h                        # 1286 px

    # crop игровой зоны → масштабируем до 1080×game_h
    game_filter = (
        f"[0:v]crop={game['w']}:{game['h']}:{game['x']}:{game['y']},"
        f"scale={TARGET_W}:{game_h}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_W}:{game_h}:(ow-iw)/2:(oh-ih)/2:black[game_out]"
    )

    # crop вебки → масштабируем до 1080×cam_h
    cam_filter = (
        f"[0:v]crop={cam['w']}:{cam['h']}:{cam['x']}:{cam['y']},"
        f"scale={TARGET_W}:{cam_h}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_W}:{cam_h}:(ow-iw)/2:(oh-ih)/2:black[cam_out]"
    )

    # Стек: игра сверху, вебка снизу
    stack = "[game_out][cam_out]vstack[final]"

    return f"{game_filter};{cam_filter};{stack}", "final"


# ════════════════════════════════════════════════════════════════
#  РЕЖИМ: BLUR (размытый фон)
# ════════════════════════════════════════════════════════════════

def _blur_vf() -> tuple[str, str]:
    """
    Стандартный вертикальный формат с размытым фоном.
    Работает без знания layout.
    """
    vf = (
        "[0:v]split=2[fg][bg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "boxblur=luma_radius=35:luma_power=3:"
        "chroma_radius=35:chroma_power=3[blurred];"
        "[fg]scale=1080:-2[scaled];"
        "[blurred][scaled]overlay=(W-w)/2:(H-h)/2[final]"
    )
    return vf, "final"


# ════════════════════════════════════════════════════════════════
#  НАРЕЗКА ОДНОГО КЛИПА
# ════════════════════════════════════════════════════════════════

def make_clip(
    input_file: str,
    start_time:  int,
    duration:    int,
    clip_index:  int,
    score:       float,
    layout:      dict,
    output_dir:  str,
    webcam_share: float = 0.33,
) -> str:
    """
    Вырезает фрагмент и конвертирует в вертикальный формат.
    Возвращает путь к готовому файлу или '' при ошибке.
    """
    os.makedirs(output_dir, exist_ok=True)

    time_label = str(timedelta(seconds=start_time)).replace(":", "-")
    output_file = os.path.join(output_dir, f"clip_{clip_index:02d}_{time_label}.mp4")

    mode = layout.get("mode", "blur")

    if mode == "smart":
        vf_chain, out_label = _smart_vf(layout, webcam_share)
    else:
        vf_chain, out_label = _blur_vf()

    vf_full = f"{vf_chain};[{out_label}]copy[out]" if mode == "smart" else vf_chain
    # Упрощаем — используем map через -vf напрямую
    vf_for_ffmpeg = vf_chain.replace(f"[{out_label}]", "").rstrip(";") \
        if mode == "blur" else vf_chain

    # Для smart режима нужно вывести финальный лейбл
    vf_arg = vf_chain if mode == "blur" else f"{vf_chain}"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_time),
        "-i", input_file,
        "-t", str(duration),
        "-filter_complex", vf_arg,
        "-map", f"[final]" if mode == "smart" else "",
        "-map", "0:a?",
        "-c:v", "libx264", "-crf", "22", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_file,
    ]

    # Убираем пустой элемент если нет map
    cmd = [c for c in cmd if c != ""]

    # Для blur map не нужен отдельно (фильтр отдаёт напрямую)
    if mode == "blur":
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start_time),
            "-i", input_file,
            "-t", str(duration),
            "-vf", vf_arg,
            "-c:v", "libx264", "-crf", "22", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_file,
        ]

    ts = str(timedelta(seconds=start_time))
    mode_icon = "🎯" if mode == "smart" else "🌫️"
    print(f"  {mode_icon} Клип {clip_index:02d}: {ts} | score={score:.3f} | {mode}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Попытка 2: fallback на blur при ошибке smart
        if mode == "smart":
            print(f"     ⚠️  Smart-режим дал ошибку, переключаюсь на blur...")
            return make_clip(
                input_file, start_time, duration, clip_index, score,
                {"mode": "blur"}, output_dir, webcam_share
            )
        print(f"     ❌ ffmpeg error:\n{result.stderr[-400:]}")
        return ""

    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / 1024 / 1024
        print(f"     ✅ {os.path.basename(output_file)} ({size_mb:.1f} МБ)")
        return output_file

    return ""
