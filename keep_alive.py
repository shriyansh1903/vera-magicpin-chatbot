import time
import os
import threading
import urllib.request
import logging

logger = logging.getLogger("keep_alive")
logger.setLevel(logging.INFO)

PING_URL = os.getenv("PING_URL", "http://127.0.0.1:8000/v1/healthz")
INTERVAL_SECONDS = int(os.getenv("PING_INTERVAL", "300"))  # Ping every 5 minutes

def keep_alive_loop(target_url: str = None, interval: int = None):
    url = target_url or PING_URL
    sleep_time = interval or INTERVAL_SECONDS
    logger.info(f"Starting keep-alive loop for URL: {url} (Interval: {sleep_time}s)")
    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VeraKeepAlive/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.getcode()
                logger.info(f"Keep-alive ping successful: HTTP {status}")
        except Exception as e:
            logger.warning(f"Keep-alive ping error: {e}")
        time.sleep(sleep_time)

def start_keep_alive_thread(target_url: str = None, interval: int = None):
    t = threading.Thread(target=keep_alive_loop, args=(target_url, interval), daemon=True)
    t.start()
    return t

if __name__ == "__main__":
    keep_alive_loop()
