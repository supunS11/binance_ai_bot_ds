try:
    import requests
except ImportError:
    requests = None

import config
from logger import log_error, log_warning


def _enabled():
    return (
        config.TELEGRAM_ENABLED
        and bool(config.TELEGRAM_BOT_TOKEN)
        and bool(config.TELEGRAM_CHAT_ID)
    )


def send_telegram_message(message):
    if not _enabled():
        return False

    if requests is None:
        log_warning("Telegram unavailable | requests package missing")
        return False

    try:
        url = (
            f"https://api.telegram.org/bot"
            f"{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        )
        response = requests.post(
            url,
            json={
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": message[:4096],
                "disable_web_page_preview": True,
            },
            timeout=config.TELEGRAM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True

    except Exception as e:
        log_error(f"Telegram send error: {type(e).__name__}")
        return False
