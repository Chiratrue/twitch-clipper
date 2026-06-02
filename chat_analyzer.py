"""
chat_analyzer.py
────────────────
Скачивает чат VOD и находит лучшие моменты по активности.
"""

import sys
import numpy as np
from collections import defaultdict
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ЧАТА
# ════════════════════════════════════════════════════════════════

# Типы событий Twitch и их «вес» в скоре
EVENT_WEIGHTS = {
    "subscription":    3,
    "resubscription":  2,
    "subgift":         3,
    "submysterygift":  5,   # большой подарок сабов
    "cheer":           3,   # биты
    "raid":            8,   # рейд — очень заметное событие
}


def download_chat(vod_url: str, vod_duration: int) -> tuple[dict, list]:
    """
    Возвращает:
      chat_buckets  — {время: кол-во сообщений}, шаг 5 сек
      sub_events    — [(время, вес)] события (сабы, биты, рейды)
    """
    try:
        from chat_downloader import ChatDownloader
    except ImportError:
        print("  ⚠️  chat-downloader не установлен → pip install chat-downloader")
        return defaultdict(int), []

    downloader = ChatDownloader()
    try:
        chat = downloader.get_chat(vod_url, max_attempts=3)
    except Exception as e:
        print(f"  ⚠️  Не удалось загрузить чат: {e}")
        return defaultdict(int), []

    chat_buckets = defaultdict(int)
    sub_events   = []
    total        = 0

    iterator = tqdm(chat, desc="  📨 Чат", unit=" msgs", smoothing=0.1) \
               if HAS_TQDM else chat

    for msg in iterator:
        t = msg.get("time_in_seconds", 0)
        if not (0 <= t <= vod_duration):
            continue

        bucket = int(t // 5) * 5
        chat_buckets[bucket] += 1
        total += 1

        msg_type = msg.get("message_type", "")
        weight = EVENT_WEIGHTS.get(msg_type, 0)
        if weight:
            sub_events.append((float(t), weight))

    print(f"  ✅ {total} сообщений, {len(sub_events)} событий")
    return chat_buckets, sub_events


# ════════════════════════════════════════════════════════════════
#  ПОСТРОЕНИЕ СИГНАЛОВ
# ════════════════════════════════════════════════════════════════

def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def build_chat_signal(buckets: dict, max_t: int) -> np.ndarray:
    sig = np.zeros(max_t + 1)
    for t, v in buckets.items():
        if 0 <= int(t) <= max_t:
            sig[int(t)] += v
    sig = uniform_filter1d(sig, size=30)   # сглаживаем по 30 сек
    return _norm(sig)


def build_event_signal(events: list, max_t: int) -> np.ndarray:
    sig = np.zeros(max_t + 1)
    for t, w in events:
        t = int(t)
        if 0 <= t <= max_t:
            # Событие «светится» 30 секунд
            end = min(max_t + 1, t + 30)
            sig[t:end] += w
    return _norm(sig)


# ════════════════════════════════════════════════════════════════
#  ПОИСК ЛУЧШИХ МОМЕНТОВ
# ════════════════════════════════════════════════════════════════

def find_best_moments(
    chat_buckets:  dict,
    sub_events:    list,
    vod_duration:  int,
    top_n:         int,
    clip_duration: int,
    min_gap:       int,
) -> list[tuple[int, float]]:
    """
    Возвращает список (start_time_sec, score) для top_n лучших моментов.
    """
    max_t = vod_duration

    chat_sig  = build_chat_signal(chat_buckets, max_t)
    event_sig = build_event_signal(sub_events,  max_t)

    # Итоговый скор: чат 60%, события 40%
    score = 0.60 * chat_sig + 0.40 * event_sig
    score = uniform_filter1d(score, size=20)

    # Отступ от краёв (пропускаем первые/последние 5 минут)
    margin = 300
    score[:margin]                     = 0
    score[max(0, max_t - margin):]     = 0

    # Порог: выше 60-го перцентиля ненулевых значений
    nonzero = score[score > 0]
    threshold = np.percentile(nonzero, 60) if len(nonzero) > 10 else 0

    min_distance = max(min_gap, clip_duration + 5)

    peaks, _ = find_peaks(score, distance=min_distance, height=threshold)

    if len(peaks) == 0:
        print("  ⚠️  Пики не найдены, берём равномерно по времени")
        step  = max_t // (top_n + 1)
        peaks = np.array([step * i for i in range(1, top_n + 1)])

    # Топ N по скору
    top_idx   = np.argsort(score[peaks])[::-1][:top_n]
    top_peaks = sorted(peaks[top_idx])   # хронологический порядок

    moments = []
    for p in top_peaks:
        start = max(margin, int(p) - clip_duration // 2)
        start = min(start, max_t - clip_duration)
        moments.append((start, float(score[p])))

    return moments
