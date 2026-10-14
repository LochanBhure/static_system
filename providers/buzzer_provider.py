"""Optional Raspberry Pi GPIO or local sound-file alert output."""

from pathlib import Path
from importlib import import_module
import time

import config


def send_buzzer_alert(_alert, _context):
    """Activate the selected local alarm; mode='none' intentionally does nothing."""
    del _alert, _context
    mode = config.BUZZER_MODE.lower()
    if mode == "none":
        return
    if mode == "gpio":
        try:
            GPIO = import_module("RPi.GPIO")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "BUZZER_MODE='gpio' requires the Raspberry Pi RPi.GPIO package."
            ) from error

        pin = int(config.BUZZER_GPIO_PIN)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        try:
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(float(config.BUZZER_DURATION_SECONDS))
        finally:
            GPIO.output(pin, GPIO.LOW)
            GPIO.cleanup(pin)
        return
    if mode == "sound":
        if not config.SOUND_FILE_PATH:
            raise ValueError(
                "SOUND_FILE_PATH must be set when BUZZER_MODE='sound'."
            )
        sound_path = Path(config.SOUND_FILE_PATH)
        if not sound_path.is_file():
            raise FileNotFoundError(f"Alert sound file not found: {sound_path}")
        try:
            playsound = import_module("playsound").playsound
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "BUZZER_MODE='sound' requires the optional playsound package."
            ) from error
        playsound(str(sound_path))
        return
    raise ValueError(f"Unsupported BUZZER_MODE: {config.BUZZER_MODE!r}")
