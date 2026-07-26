"""Differential-drive base: velocity commands in, servo pulse widths out.

Two things matter here beyond the serial plumbing:

* **Velocity, not pulse widths.** The skill loop thinks in m/s and rad/s. The
  conversion to per-wheel microseconds lives in one place, right here.
* **The watchdog is not optional.** Every velocity command carries a duration, and if
  the next one does not arrive in time the wheels stop on their own. One stalled
  Python thread without it drives the robot into a wall in front of the judges. The
  firmware has an independent timeout for the case where this process dies outright.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass

import serial
from serial.tools import list_ports

from robot.config import (
    DEFAULT_SPEED_US,
    DRIVE_BAUD,
    DRIVE_PORT,
    FIRMWARE_WATCHDOG_MS,
    FULL_SCALE_US,
    MAX_ANGULAR,
    MAX_LINEAR,
    MAX_US,
    MIN_US,
    STOP_US,
    WATCHDOG_GRACE_MS,
    WHEEL_BASE_M,
)

WATCHDOG_TICK_S = 0.02


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class DriveConfig:
    port: str | None = DRIVE_PORT
    baud: int = DRIVE_BAUD
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


def velocity_to_us(linear: float, angular: float) -> tuple[int, int]:
    """(linear m/s, angular rad/s) -> (left_us, right_us).

    Positive angular is counter-clockwise, so the left wheel is the one that slows
    down. Both wheels are scaled by the same factor, which means a command asking for
    more than the base can do arrives slower but still pointing the right way.
    """
    linear = _clamp(linear, -MAX_LINEAR, MAX_LINEAR)
    angular = _clamp(angular, -MAX_ANGULAR, MAX_ANGULAR)

    left = linear - angular * WHEEL_BASE_M / 2
    right = linear + angular * WHEEL_BASE_M / 2
    full_scale = MAX_LINEAR + MAX_ANGULAR * WHEEL_BASE_M / 2

    biggest = max(abs(left), abs(right))
    if biggest > full_scale:
        left *= full_scale / biggest
        right *= full_scale / biggest

    return (
        round(STOP_US + FULL_SCALE_US * left / full_scale),
        round(STOP_US + FULL_SCALE_US * right / full_scale),
    )


class DriveController:
    """Shared velocity/watchdog behaviour. Subclasses only have to move wheels."""

    def __init__(self) -> None:
        self._deadline = 0.0
        self._cmd = (0.0, 0.0)
        self._last_us = (STOP_US, STOP_US)
        self._io_lock = threading.RLock()
        self._shutdown = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    # -- subclass hooks ---------------------------------------------------
    @property
    def connected(self) -> bool:
        raise NotImplementedError

    def _write_us(self, left_us: int, right_us: int) -> str:
        raise NotImplementedError

    def _tick(self, linear: float, angular: float, dt: float) -> None:
        """Called every watchdog tick while a command is live. Only the simulator
        needs it; on hardware the wheels are already turning."""

    # -- public API -------------------------------------------------------
    def drive_us(self, left_us: int, right_us: int, ms: int | None = None) -> str:
        left_us = int(_clamp(left_us, MIN_US, MAX_US))
        right_us = int(_clamp(right_us, MIN_US, MAX_US))
        with self._io_lock:
            self._last_us = (left_us, right_us)
            self._cmd = (0.0, 0.0)
            self._deadline = (
                0.0 if ms is None else time.monotonic() + (ms + WATCHDOG_GRACE_MS) / 1000.0
            )
            return self._write_us(left_us, right_us)

    def drive_velocity(self, linear: float = 0.0, angular: float = 0.0, ms: int = 300) -> str:
        left_us, right_us = velocity_to_us(linear, angular)
        with self._io_lock:
            reply = self.drive_us(left_us, right_us, ms=ms)
            self._cmd = (linear, angular)
            return reply

    def stop(self) -> str:
        with self._io_lock:
            self._cmd = (0.0, 0.0)
            self._deadline = 0.0
            self._last_us = (STOP_US, STOP_US)
            return self._write_us(STOP_US, STOP_US)

    def forward(self, speed_us: int = DEFAULT_SPEED_US) -> str:
        return self.drive_us(STOP_US + speed_us, STOP_US + speed_us)

    def backward(self, speed_us: int = DEFAULT_SPEED_US) -> str:
        return self.drive_us(STOP_US - speed_us, STOP_US - speed_us)

    def turn_left(self, speed_us: int = DEFAULT_SPEED_US) -> str:
        return self.drive_us(STOP_US - speed_us, STOP_US + speed_us)

    def turn_right(self, speed_us: int = DEFAULT_SPEED_US) -> str:
        return self.drive_us(STOP_US + speed_us, STOP_US - speed_us)

    def close(self) -> None:
        self._shutdown.set()

    # -- watchdog ---------------------------------------------------------
    def _watch(self) -> None:
        while not self._shutdown.wait(WATCHDOG_TICK_S):
            deadline = self._deadline
            if deadline and time.monotonic() > deadline:
                try:
                    self.stop()
                except Exception:  # a dead serial link must not kill the watchdog
                    self._deadline = 0.0
                continue
            linear, angular = self._cmd
            if deadline and (linear or angular):
                self._tick(linear, angular, WATCHDOG_TICK_S)


class DriveBase(DriveController):
    """Serial bridge to the Arduino drivebase sketch."""

    def __init__(self, config: DriveConfig | None = None) -> None:
        self.config = config or DriveConfig()
        self._ser: serial.Serial | None = None
        super().__init__()

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
        time.sleep(2.0)  # Arduino resets when the USB port is opened
        self._ser.reset_input_buffer()
        self._send(f"WD,{FIRMWARE_WATCHDOG_MS}")
        self.stop()
        return port

    def close(self) -> None:
        super().close()
        if self._ser and self._ser.is_open:
            with contextlib.suppress(Exception):
                self._write_us(STOP_US, STOP_US)
            self._ser.close()
        self._ser = None

    def _send(self, line: str) -> str:
        if not self.connected:
            raise RuntimeError("Drive base not connected")
        assert self._ser is not None
        with self._io_lock:
            self._ser.write((line.strip() + "\n").encode("ascii"))
            self._ser.flush()
            # The firmware emits unsolicited WARN lines when its own failsafe fires,
            # so skip ahead to the acknowledgement rather than mistaking one for the
            # other and desyncing every reply after it.
            for _ in range(4):
                reply = self._ser.readline().decode("ascii", errors="replace").strip()
                if not reply or reply.startswith(("OK", "ERR")):
                    return reply
            return reply

    def _write_us(self, left_us: int, right_us: int) -> str:
        if not self.connected:
            # Stops are issued on shutdown and by the watchdog, when the link may
            # already be gone. Never raise on the way down.
            if (left_us, right_us) == (STOP_US, STOP_US):
                return ""
            raise RuntimeError("Drive base not connected")
        return self._send(f"DRIVE,{left_us},{right_us}")

    def ping(self) -> str:
        return self._send("PING")


class MockDrive(DriveController):
    """Drives the simulator instead of wheels, so everything above works unplugged."""

    @property
    def connected(self) -> bool:
        return True

    def connect(self, port: str | None = None) -> str:
        return "mock"

    def ping(self) -> str:
        return "OK PONG (mock)"

    def _write_us(self, left_us: int, right_us: int) -> str:
        return f"OK DRIVE {left_us} {right_us} (mock)"

    def _tick(self, linear: float, angular: float, dt: float) -> None:
        from robot.world import WORLD

        WORLD.drive(linear, angular, dt)


def make_drive(mock: bool, config: DriveConfig | None = None) -> DriveController:
    return MockDrive() if mock else DriveBase(config)
