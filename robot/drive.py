"""Serial bridge to the Arduino differential-drive base."""

from __future__ import annotations

import time
from dataclasses import dataclass

import serial
from serial.tools import list_ports


STOP_US = 1500
# Modest cruise speed for first tests (us offset from stop)
DEFAULT_SPEED_US = 120


@dataclass
class DriveConfig:
    port: str | None = None
    baud: int = 115200
    timeout_s: float = 1.0


def list_serial_ports() -> list[tuple[str, str]]:
    return [(p.device, p.description or "") for p in list_ports.comports()]


def guess_arduino_port() -> str | None:
    ports = list_ports.comports()
    preferred = ("arduino", "usbmodem", "usbserial", "wchusb", "ch340", "cp210")
    for p in ports:
        blob = f"{p.device} {p.description} {p.manufacturer or ''}".lower()
        if any(k in blob for k in preferred):
            # Skip Bluetooth / debug noise
            if "bluetooth" in blob or "debug-console" in blob:
                continue
            return p.device
    return None


class DriveBase:
    def __init__(self, config: DriveConfig | None = None) -> None:
        self.config = config or DriveConfig()
        self._ser: serial.Serial | None = None

    @property
    def connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def connect(self, port: str | None = None) -> str:
        if self.connected:
            return self._ser.port  # type: ignore[union-attr]

        port = port or self.config.port or guess_arduino_port()
        if not port:
            available = ", ".join(f"{d} ({desc})" for d, desc in list_serial_ports()) or "none"
            raise RuntimeError(f"No Arduino port found. Available: {available}")

        self._ser = serial.Serial(port, self.config.baud, timeout=self.config.timeout_s)
        time.sleep(2.0)  # Arduino reset after USB open
        self._ser.reset_input_buffer()
        self.stop()
        return port

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            try:
                self.stop()
            except Exception:
                pass
            self._ser.close()
        self._ser = None

    def _send(self, line: str) -> str:
        if not self.connected:
            raise RuntimeError("Drive base not connected")
        assert self._ser is not None
        payload = (line.strip() + "\n").encode("ascii")
        self._ser.write(payload)
        self._ser.flush()
        reply = self._ser.readline().decode("ascii", errors="replace").strip()
        return reply

    def ping(self) -> str:
        return self._send("PING")

    def stop(self) -> str:
        return self._send("STOP")

    def drive_us(self, left_us: int, right_us: int) -> str:
        return self._send(f"DRIVE,{int(left_us)},{int(right_us)}")

    def forward(self, speed_us: int = DEFAULT_SPEED_US) -> str:
        return self.drive_us(STOP_US + speed_us, STOP_US + speed_us)

    def backward(self, speed_us: int = DEFAULT_SPEED_US) -> str:
        return self.drive_us(STOP_US - speed_us, STOP_US - speed_us)

    def turn_left(self, speed_us: int = DEFAULT_SPEED_US) -> str:
        return self.drive_us(STOP_US - speed_us, STOP_US + speed_us)

    def turn_right(self, speed_us: int = DEFAULT_SPEED_US) -> str:
        return self.drive_us(STOP_US + speed_us, STOP_US - speed_us)
