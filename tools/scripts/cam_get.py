import os, time
import requests

BASE = "http://192.168.1.123"   # <-- IP из Serial
OUT  = "coins_raw"
N    = 500                      # сколько кадров
DELAY = 0.30                    # сек между кадрами

os.makedirs(OUT, exist_ok=True)

for i in range(N):
    r = requests.get(f"{BASE}/capture", timeout=5, headers={"Cache-Control":"no-cache"})
    r.raise_for_status()
    fn = os.path.join(OUT, f"img_{i:05d}.jpg")
    with open(fn, "wb") as f:
        f.write(r.content)
    print("saved", fn)
    time.sleep(DELAY)
