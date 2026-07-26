"""Perception. Two backends behind one interface.

    color -- HSV blob matching. No network, works against the simulator, and doubles
             as the real-hardware fallback when the VLM is slow or the Wi-Fi is bad.
    vlm   -- the demo path. One tight prompt, JSON out, provider-agnostic.

Both return normalized bboxes [x0, y0, x1, y1] in 0..1, origin top-left.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

from robot.config import (
    HOME_TAG_LABEL,
    VISION_BACKEND,
    VLM_MAX_EDGE,
    VLM_MODEL,
    VLM_PROVIDER,
    VLM_TIMEOUT_S,
)

Bbox = tuple[float, float, float, float]


@dataclass
class Detection:
    label: str
    bbox: Bbox
    conf: float

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def as_dict(self) -> dict:
        return {"label": self.label, "bbox": list(self.bbox), "conf": self.conf}


# --------------------------------------------------------------------------
# color backend
# --------------------------------------------------------------------------
# (hue_lo, hue_hi, sat_min, val_min). Retune on the real objects under the real demo
# lighting -- there is a calibration helper in scripts/calibrate.py.
# hue_lo > hue_hi means the range wraps past 180, which is the only way to catch red.
HSV_PROFILES: dict[str, tuple[int, int, int, int]] = {
    "strawberries": (170, 10, 110, 80),
    "banana": (22, 34, 110, 110),
    "granola bar": (10, 20, 80, 60),
    "water bottle": (85, 100, 60, 100),
    HOME_TAG_LABEL: (140, 165, 100, 90),
}
MIN_BLOB_AREA_PX = 250


def _largest_blob(frame: np.ndarray, profile) -> Detection | None:
    h_lo, h_hi, s_min, v_min = profile
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if h_lo > h_hi:  # wraps past 180: red
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (h_lo, s_min, v_min), (179, 255, 255)),
            cv2.inRange(hsv, (0, s_min, v_min), (h_hi, 255, 255)),
        )
    else:
        mask = cv2.inRange(hsv, (h_lo, s_min, v_min), (h_hi, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_BLOB_AREA_PX:
        return None
    x, y, w, h = cv2.boundingRect(biggest)
    fh, fw = frame.shape[:2]
    return Detection("", (x / fw, y / fh, (x + w) / fw, (y + h) / fh), 0.8)


def _match_profile(query: str) -> tuple | None:
    """Tolerate what a customer actually types: 'Strawberries', 'some strawberries'."""
    q = query.strip().lower()
    if q in HSV_PROFILES:
        return HSV_PROFILES[q]
    for label, profile in HSV_PROFILES.items():
        if label in q or q in label:
            return profile
    return None


class ColorVision:
    name = "color"

    async def scene(self, frame: np.ndarray) -> list[Detection]:
        out = []
        for label, profile in HSV_PROFILES.items():
            det = _largest_blob(frame, profile)
            if det:
                out.append(Detection(label, det.bbox, det.conf))
        return out

    async def locate(self, frame: np.ndarray, query: str) -> Detection | None:
        profile = _match_profile(query)
        if profile is None:
            # unknown label: fall back to whatever is biggest and not the tag
            candidates = [d for d in await self.scene(frame) if d.label != HOME_TAG_LABEL]
            best = max(candidates, key=lambda d: d.height, default=None)
            return Detection(query, best.bbox, best.conf) if best else None
        det = _largest_blob(frame, profile)
        return Detection(query, det.bbox, det.conf) if det else None

    async def aclose(self) -> None:
        pass


# --------------------------------------------------------------------------
# vlm backend
# --------------------------------------------------------------------------
_SCENE_PROMPT = (
    "List every distinct physical object a small robot could pick up in this image. "
    'Reply with JSON only: {"objects": [{"label": "...", "bbox": [x0,y0,x1,y1]}]} '
    "where bbox is normalized 0-1, origin top-left. Use short everyday labels."
)

_LOCATE_PROMPT = (
    "Find the {query} in this image. Reply with JSON only: "
    '{{"found": true|false, "bbox": [x0,y0,x1,y1]}} '
    "where bbox is normalized 0-1, origin top-left. "
    "If it is not clearly visible, return found=false."
)


def _shrink(frame: np.ndarray) -> bytes:
    h, w = frame.shape[:2]
    scale = min(1.0, VLM_MAX_EDGE / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])[1].tobytes()


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in model reply: {text[:200]}")
    return json.loads(match.group(0))


def _normalize(bbox) -> Bbox:
    vals = [float(v) for v in bbox]
    # Some models emit 0-1000 instead of 0-1. Rescale rather than fail.
    if max(vals) > 1.5:
        vals = [v / 1000.0 for v in vals]
    x0, y0, x1, y1 = vals
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


class VlmVision:
    """Talks to Gemini, OpenAI or Anthropic over plain REST -- no SDKs to version-pin."""

    name = "vlm"

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=VLM_TIMEOUT_S)

    async def _ask(self, jpeg: bytes, prompt: str) -> dict:
        b64 = base64.b64encode(jpeg).decode()
        if VLM_PROVIDER == "gemini":
            model = VLM_MODEL or "gemini-2.0-flash"
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent"
            )
            r = await self._http.post(
                url,
                headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0,
                        "responseMimeType": "application/json",
                    },
                },
            )
            r.raise_for_status()
            return _parse_json(r.json()["candidates"][0]["content"]["parts"][0]["text"])

        if VLM_PROVIDER == "openai":
            model = VLM_MODEL or "gpt-4o-mini"
            r = await self._http.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={
                    "model": model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                            ],
                        }
                    ],
                },
            )
            r.raise_for_status()
            return _parse_json(r.json()["choices"][0]["message"]["content"])

        if VLM_PROVIDER == "anthropic":
            model = VLM_MODEL or "claude-sonnet-4-5"
            r = await self._http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": model,
                    "max_tokens": 512,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                },
            )
            r.raise_for_status()
            return _parse_json(r.json()["content"][0]["text"])

        raise ValueError(f"unknown VLM provider {VLM_PROVIDER!r}")

    async def scene(self, frame: np.ndarray) -> list[Detection]:
        data = await self._ask(_shrink(frame), _SCENE_PROMPT)
        return [
            Detection(str(o["label"]).lower(), _normalize(o["bbox"]), 0.9)
            for o in data.get("objects", [])
            if o.get("bbox")
        ]

    async def locate(self, frame: np.ndarray, query: str) -> Detection | None:
        data = await self._ask(_shrink(frame), _LOCATE_PROMPT.format(query=query))
        if not data.get("found") or not data.get("bbox"):
            return None
        return Detection(query, _normalize(data["bbox"]), 0.9)

    async def aclose(self) -> None:
        await self._http.aclose()


def make_vision(backend: str = VISION_BACKEND):
    return VlmVision() if backend == "vlm" else ColorVision()
