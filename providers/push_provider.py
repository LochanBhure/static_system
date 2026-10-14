"""ntfy.sh push notifications with an optional Pushover fallback."""

import json
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import config


def _message(alert):
    title = alert.get("title") or alert.get("alert_type", "CCTV alert")
    message = alert.get("message") or alert.get("reason", "")
    return str(title), str(message)


def _send_ntfy(alert, context):
    if not config.NTFY_TOPIC:
        raise ValueError("NTFY_TOPIC must be set to enable ntfy notifications.")
    title, message = _message(alert)
    body = {
        "title": title,
        "message": message,
        "priority": 5 if context.get("level") == "HIGH" else 4,
    }
    request = Request(
        f"https://ntfy.sh/{quote(config.NTFY_TOPIC, safe='')}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=config.OLLAMA_TIMEOUT) as response:
        if response.status < 200 or response.status >= 300:
            raise URLError(f"ntfy returned HTTP {response.status}.")


def _send_pushover(alert, context):
    if not config.PUSHOVER_TOKEN or not config.PUSHOVER_USER_KEY:
        raise ValueError(
            "PUSHOVER_TOKEN and PUSHOVER_USER_KEY are required for Pushover."
        )
    title, message = _message(alert)
    payload = urlencode({
        "token": config.PUSHOVER_TOKEN,
        "user": config.PUSHOVER_USER_KEY,
        "title": title,
        "message": message,
        "priority": 1 if context.get("level") == "HIGH" else 0,
    }).encode("utf-8")
    request = Request(
        "https://api.pushover.net/1/messages.json",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=config.OLLAMA_TIMEOUT) as response:
        if response.status < 200 or response.status >= 300:
            raise URLError(f"Pushover returned HTTP {response.status}.")


def send_push_alert(alert, context):
    """Dispatch a push alert using the configured provider and fallback."""
    provider = config.PUSH_PROVIDER.lower()
    if provider == "pushover":
        _send_pushover(alert, context)
        return
    if provider != "ntfy":
        raise ValueError(f"Unsupported PUSH_PROVIDER: {config.PUSH_PROVIDER!r}")
    if not config.NTFY_TOPIC:
        if config.PUSHOVER_TOKEN and config.PUSHOVER_USER_KEY:
            _send_pushover(alert, context)
            return
        raise ValueError(
            "NTFY_TOPIC is required for ntfy; configure Pushover credentials "
            "to use the fallback."
        )

    try:
        _send_ntfy(alert, context)
    except (URLError, TimeoutError) as ntfy_error:
        if not (config.PUSHOVER_TOKEN and config.PUSHOVER_USER_KEY):
            raise
        try:
            _send_pushover(alert, context)
        except (URLError, TimeoutError) as pushover_error:
            raise RuntimeError(
                f"ntfy failed ({ntfy_error}); Pushover fallback failed "
                f"({pushover_error})."
            ) from pushover_error
