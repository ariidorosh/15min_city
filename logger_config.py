# logger_config.py
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_FILE_TIMESTAMPED = os.path.join(LOG_DIR, f"15min_city_{_ts}.log")
LOG_FILE_LATEST = os.path.join(LOG_DIR, "15min_city_latest.log")

logger = logging.getLogger("15min_city")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh_ts = RotatingFileHandler(LOG_FILE_TIMESTAMPED, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh_latest = RotatingFileHandler(LOG_FILE_LATEST, maxBytes=10_000_000, backupCount=2, encoding="utf-8")

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh_ts.setFormatter(fmt)
    fh_latest.setFormatter(fmt)

    logger.addHandler(fh_ts)
    logger.addHandler(fh_latest)

# приглушує лишні логери від бібліотек
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("overpy").setLevel(logging.WARNING)
logging.getLogger("osmnx").setLevel(logging.INFO)
