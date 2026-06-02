#!/usr/bin/env python3
"""
main.py — точка входа TwitchClipper

Использование:
  python main.py                            # использует config.py
  python main.py --vod URL                  # конкретный VOD
  python main.py --vod URL --clips 7        # 7 клипов
  python main.py --reset-layout             # сбросить настройку вебки
"""

import os
import re
import sys
import json
import argparse
import subprocess
import requests
from datetime import timedelta

import config
from chat_analyzer   import download_chat, find_best_moments
from layout_detector import detect_or_load_layout
from clip_maker      import make_clip


# ════════════════════════════════════════════════════════════════
#  УТИЛИТЫ
# ════════════════════════════════════════════════════════════════

def parse_twitch_duration(s: str) -> int:
    m = re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s or "")
    h, mi, sec = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + sec


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ ffmpeg не найден. Установи: https://ffmpeg.org/download.html")
        sys.exit(1)


def check_yt_dlp():
    try:
        import yt_dlp
    except ImportError:
        print("❌ yt-dlp не установлен: pip install yt-dlp")
        sys.exit(1)


# ════════════════════════════════════════════════════════════════
#  TWITCH API
# ════════════════════════════════════════════════════════════════

def get_vod_info(vod_url: str) -> tuple[str, dict]:
    video_id = vod_url.rstrip("/").split("/")[-1]
    headers  = {
        "Client-ID":     config.TWITCH_CLIENT_ID,
        "Authorization": f"Bearer {config.TWITCH_ACCESS_TOKEN}",
    }
    r = requests.get(
        f"https://api.twitch.tv/helix/videos?id={video_id}",
        headers=headers, timeout=10,
    )
    if r.status_code == 401:
        print("❌ 401 Unauthorized — проверь TWITCH_ACCESS_TOKEN в config.py")
        sys.exit(1)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        print("❌ VOD не найден. Проверь ссылку.")
        sys.exit(1)
    return video_id, data[0]


# ════════════════════════════════════════════════════════════════
#  СКАЧИВАНИЕ VOD
# ════════════════════════════════════════════════════════════════

def download_vod(vod_url: str, video_id: str, quality: int) -> str:
    import yt_dlp

    output_file = f"{video_id}.mp4"
    if os.path.exists(output_file):
        size = os.path.getsize(output_file) / 1024 / 1024
        print(f"  ♻️  VOD уже скачан ({size:.0f} МБ), пропускаем")
        return output_file

    print(f"  ⬇️  Скачиваем VOD ({quality}p)... это займёт время")
    ydl_opts = {
        "format":    f"best[height<={quality}]",
        "outtmpl":   output_file,
        "quiet":     False,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([vod_url])
    return output_file


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TwitchClipper → TikTok")
    parser.add_argument("--vod",          help="Twitch VOD URL (переопределяет config.py)")
    parser.add_argument("--clips",  type=int, help="Количество клипов")
    parser.add_argument("--reset-layout", action="store_true",
                        help="Удалить сохранённый layout и переопределить")
    args = parser.parse_args()

    # ── шапка ───────────────────────────────────────────────────
    print("╔══════════════════════════════════════════╗")
    print("║   🎮 TwitchClipper → TikTok  v2.0        ║")
    print("╚══════════════════════════════════════════╝\n")

    # ── проверяем зависимости ───────────────────────────────────
    check_ffmpeg()
    check_yt_dlp()

    # ── настройки ───────────────────────────────────────────────
    vod_url    = args.vod   or config.VOD_URL
    top_clips  = args.clips or config.TOP_CLIPS

    if "XXXXXXXXXX" in vod_url or "СЮДА" in config.TWITCH_CLIENT_ID:
        print("⚠️  Заполни config.py: VOD_URL, TWITCH_CLIENT_ID, TWITCH_ACCESS_TOKEN")
        sys.exit(1)

    if args.reset_layout and os.path.exists(config.LAYOUT_CONFIG_FILE):
        os.remove(config.LAYOUT_CONFIG_FILE)
        print("🗑  layout_config.json удалён, будет переопределён\n")

    # ── инфо о VOD ──────────────────────────────────────────────
    print("🔍 Получаем информацию о VOD...")
    video_id, vod_info = get_vod_info(vod_url)
    title    = vod_info.get("title", "—")
    duration = parse_twitch_duration(vod_info.get("duration", "0s"))
    print(f"  📺 {title}")
    print(f"  ⏱  {str(timedelta(seconds=duration))}")
    print(f"  🆔 {video_id}\n")

    # ── анализ чата ─────────────────────────────────────────────
    print("📊 Анализируем чат...")
    chat_buckets, sub_events = download_chat(vod_url, duration)

    # ── лучшие моменты ──────────────────────────────────────────
    print("\n🎯 Ищем лучшие моменты...")
    moments = find_best_moments(
        chat_buckets, sub_events, duration,
        top_n=top_clips,
        clip_duration=config.CLIP_DURATION,
        min_gap=config.MIN_GAP,
    )
    if not moments:
        print("😔 Моменты не найдены")
        sys.exit(0)

    for i, (t, sc) in enumerate(moments, 1):
        print(f"  #{i:02d}  {str(timedelta(seconds=t)):>8}  score={sc:.3f}")

    # ── скачиваем VOD ───────────────────────────────────────────
    print(f"\n⬇️  Скачиваем VOD...")
    vod_file = download_vod(vod_url, video_id, config.VOD_QUALITY)

    # ── определяем layout (вебка + игра) ────────────────────────
    print("\n🖼  Определяем расположение вебки и игровой зоны...")
    layout = detect_or_load_layout(vod_file, config.LAYOUT_CONFIG_FILE, duration)

    # ── нарезаем клипы ──────────────────────────────────────────
    print(f"\n✂️  Нарезаем {len(moments)} клипов → {config.OUTPUT_DIR}/")
    clips = []
    for i, (start, score) in enumerate(moments, 1):
        path = make_clip(
            input_file   = vod_file,
            start_time   = start,
            duration     = config.CLIP_DURATION,
            clip_index   = i,
            score        = score,
            layout       = layout,
            output_dir   = config.OUTPUT_DIR,
            webcam_share = config.WEBCAM_SHARE,
        )
        if path:
            clips.append(path)

    # ── итог ────────────────────────────────────────────────────
    print(f"\n{'═'*48}")
    print(f"✅ Готово! {len(clips)} / {len(moments)} клипов:")
    for c in clips:
        mb = os.path.getsize(c) / 1024 / 1024
        print(f"   📱 {c}  ({mb:.1f} МБ)")

    log_path = os.path.join(config.OUTPUT_DIR, "session_log.json")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "vod_url":  vod_url,
            "video_id": video_id,
            "title":    title,
            "duration": duration,
            "layout":   layout,
            "moments":  [{"start": s, "score": round(sc, 4)} for s, sc in moments],
            "clips":    clips,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📋 Лог: {log_path}")
    print(f"{'═'*48}\n")


if __name__ == "__main__":
    main()
