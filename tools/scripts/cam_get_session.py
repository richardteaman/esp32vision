import os, time
import requests
from datetime import datetime

BASE = "http://192.168.1.4"   # <-- IP из Serial
ROOT_NAME = "captured"          # папка рядом со скриптом

session_name = input("Имя сессии (например wood_day): ").strip() or "session"
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, ROOT_NAME)
OUT = os.path.join(ROOT, f"{ts}_{session_name}")
os.makedirs(OUT, exist_ok=True)

N = int(input("Сколько кадров? (например 300): ") or "300")
DELAY = float(input("Задержка сек? (например 0.3): ") or "0.3")

for i in range(N):
    r = requests.get(f"{BASE}/capture", timeout=5, headers={"Cache-Control":"no-cache"})
    r.raise_for_status()

    fn = os.path.join(OUT, f"img_{i:05d}.jpg")
    with open(fn, "wb") as f:
        f.write(r.content)

    if (i + 1) % 25 == 0:
        print(f"{i+1}/{N}")
    time.sleep(DELAY)

print("done:", OUT)
