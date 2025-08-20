# nearest_airport.py — ближайший крупный/международный аэропорт с локальным Parquet-кэшем
# Формат ответа:
#   {"airport": str|null, "error": str|null}    

import os, math, csv, json, threading, time
from io import StringIO
from typing import Optional, Dict, Any, List, Tuple

import requests
import pyarrow as pa
import pyarrow.parquet as pq

# --- Константы (настройки) ---
USER_AGENT = "nearest-airport-cli/1.0 (+contact@example.com)"  # User-Agent для запросов, чтобы сайт не блокировал
OURAIRPORTS_PRIMARY = "https://ourairports.com/data/airports.csv"
OURAIRPORTS_FALLBACK = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv"

PARQUET_FILE = "airports_cache.parquet"   # локальный кэш в файле
CACHE_TTL_SECONDS = 7 * 24 * 3600         # кэш живёт 7 дней

MAX_AIRPORT_DISTANCE_KM = 300.0           # максимальное расстояние поиска аэропорта
AIRPORT_TYPES = {"large_airport", "medium_airport"}  # какие типы аэропортов нам нужны
REQUIRE_SCHEDULED_SERVICE = True          # брать только те аэропорты, где есть регулярные рейсы

# --- Функция для расчёта расстояния между координатами (по формуле гаверсинуса) ---
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float, phi1: float, cos_phi1: float) -> float:
    R = 6371.0088
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + cos_phi1*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# --- Кэш в памяти (чтобы не перечитывать файл при каждом запросе) ---
_AIRPORTS_CACHE: List[Dict[str, Any]] = []
_AIRPORTS_LOCK = threading.Lock()

# --- Загрузка текста по URL (CSV) ---
def _download_text(url: str, timeout: int = 20) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        return r.text
    except requests.RequestException:
        return None

# --- Проверка, что parquet-файл ещё свежий ---
def _parquet_is_fresh(path: str) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path) < CACHE_TTL_SECONDS)

# --- Запись списка аэропортов в Parquet ---
def _write_parquet(airports: List[Dict[str, Any]]):
    if not airports:
        return  # не сохраняем пустой список
    table = pa.Table.from_arrays(
        [
            pa.array([a["name"] for a in airports], type=pa.string()),
            pa.array([a["lat"]  for a in airports], type=pa.float64()),
            pa.array([a["lon"]  for a in airports], type=pa.float64()),
        ],
        names=["name","lat","lon"]
    )
    pq.write_table(table, PARQUET_FILE)

# --- Чтение аэропортов из Parquet ---
def _read_parquet() -> List[Dict[str, Any]]:
    table = pq.read_table(PARQUET_FILE, columns=["name","lat","lon"])
    cols = table.to_pydict()
    return [{"name": n, "lat": float(la), "lon": float(lo)}
            for n, la, lo in zip(cols["name"], cols["lat"], cols["lon"])]

# --- Парсинг CSV-файла (берём только нужные аэропорты) ---
def _parse_airports_csv(text: str) -> List[Dict[str, Any]]:
    airports: List[Dict[str, Any]] = []
    reader = csv.DictReader(StringIO(text))
    get = lambda row, key: (row.get(key) or "").strip()

    for row in reader:
        tp = get(row, "type")
        if tp not in AIRPORT_TYPES:   # только большие/средние аэропорты
            continue
        if REQUIRE_SCHEDULED_SERVICE:
            if (row.get("scheduled_service") or "").lower().strip() != "yes":
                continue

        name = get(row, "name")
        lat_s, lon_s = row.get("latitude_deg"), row.get("longitude_deg")
        if not name or not lat_s or not lon_s:
            continue
        try:
            lat, lon = float(lat_s), float(lon_s)
        except ValueError:
            continue

        airports.append({"name": name, "lat": lat, "lon": lon})

    return airports

# --- Загрузка и кэширование списка аэропортов ---
def load_airports_once(force_refresh: bool = False, debug: bool = False) -> List[Dict[str, Any]]:
    global _AIRPORTS_CACHE
    with _AIRPORTS_LOCK:
        if _AIRPORTS_CACHE and not force_refresh:
            return _AIRPORTS_CACHE

        # 1) Если свежий parquet есть — читаем его
        if not force_refresh and _parquet_is_fresh(PARQUET_FILE):
            try:
                _AIRPORTS_CACHE = _read_parquet()
                if debug:
                    print(f"[debug] loaded {len(_AIRPORTS_CACHE)} airports from parquet")
                return _AIRPORTS_CACHE
            except Exception:
                pass  # если parquet повреждён, то перекачаем CSV

        # 2) Скачиваем CSV
        txt = _download_text(OURAIRPORTS_PRIMARY) or _download_text(OURAIRPORTS_FALLBACK)
        airports = _parse_airports_csv(txt) if txt else []

        # 3) Сохраняем результат в parquet
        if airports:
            try:
                _write_parquet(airports)
                if debug:
                    print(f"[debug] wrote parquet with {len(airports)} airports")
            except Exception:
                if debug:
                    print("[debug] failed to write parquet (skipping)")

        # 4) Если сеть недоступна, пробуем использовать старый parquet
        if not airports and os.path.exists(PARQUET_FILE):
            try:
                airports = _read_parquet()
                if debug:
                    print(f"[debug] fallback to existing parquet ({len(airports)} airports)")
            except Exception:
                airports = []

        _AIRPORTS_CACHE = airports
        if debug:
            print(f"[debug] airports loaded: count={len(_AIRPORTS_CACHE)}")
        return _AIRPORTS_CACHE

# --- Поиск ближайшего аэропорта ---
def nearest_airport(lat: float, lon: float, airports: List[Dict[str, Any]], debug: bool = False) -> Tuple[Optional[str], Optional[str]]:
    if not airports:
        return None, None

    phi1 = math.radians(lat)
    cos_phi1 = math.cos(phi1)

    best_name = None
    best_dist = float("inf")

    for a in airports:
        d = haversine_km(lat, lon, a["lat"], a["lon"], phi1, cos_phi1)
        if d < best_dist:
            best_dist = d
            best_name = a["name"]

    if debug and best_name is not None:
        print(f"[debug] nearest: {best_name} ({best_dist:.3f} km)")

    if best_dist > MAX_AIRPORT_DISTANCE_KM:
        return None, None
    return best_name, None

# --- Публичная функция (то, что можно вызывать снаружи) ---
def get_nearest_airport(lat: float, lon: float, *, refresh_airports: bool = False, debug: bool = False) -> Dict[str, Any]:
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"airport": None, "error": "Invalid coordinates"}

    airports = load_airports_once(force_refresh=refresh_airports, debug=debug)
    airport, err = nearest_airport(lat, lon, airports, debug=debug)
    return {"airport": airport if airport else None, "error": err if err else None}

# --- CLI: запуск из терминала ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nearest major/international airport (Parquet cache)")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--radius-km", type=float, default=None)
    parser.add_argument("--refresh-airports", action="store_true", help="Force refresh (ignore local parquet)")
    args = parser.parse_args()

    if args.radius_km is not None:
        MAX_AIRPORT_DISTANCE_KM = float(args.radius_km)

    result = get_nearest_airport(args.lat, args.lon, refresh_airports=args.refresh_airports, debug=args.debug)
    print(json.dumps(result, ensure_ascii=False))
