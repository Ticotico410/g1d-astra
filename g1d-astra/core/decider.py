"""Image-grounded GPT-6 Astra skill selection used by G1-D demos."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy.typing as npt
from PIL import Image


@dataclass
class Decision:
    skill: str
    confidence: float | None = None
    note: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _responses_url() -> str:
    base = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
            or "https://api.openai.com/v1").strip().rstrip("/")
    return f"{base}/responses"


def _png_b64(rgb: npt.NDArray, size: int = 512) -> str:
    image = Image.fromarray(rgb)
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@dataclass
class Astra:
    """Send current RGB observations and allowed skills to the Responses API."""

    model: str = "gpt-6-astra"
    effort: str = "high"
    calls: int = 0
    seconds: float = 0.0
    history: list[str] = field(default_factory=list)

    def choose(self, state: dict[str, Any], skills: dict[str, str], instructions: str,
               images: dict[str, npt.NDArray]) -> Decision:
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "skill": {"type": "string", "enum": list(skills)},
                "note": {"type": "string"},
            },
            "required": ["skill", "note"],
        }
        prompt = (
            f"{instructions}\n\nSkills:\n"
            + "\n".join(f"- {key}: {value}" for key, value in skills.items())
            + f"\n\nState:\n{json.dumps(state, indent=1, ensure_ascii=False)}\n"
            + "Recent decisions:\n" + "\n".join(self.history[-6:])
            + "\n\nPick one skill. In note, briefly describe the observation and reason."
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for name, rgb in images.items():
            content.extend([
                {"type": "input_text", "text": f"camera '{name}':"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{_png_b64(rgb)}"},
            ])
        body = {
            "model": self.model,
            "reasoning": {"effort": self.effort},
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "decision",
                                  "schema": schema, "strict": True}},
        }
        api_key = (os.environ.get("OPENAI_API_KEY")
                   or os.environ.get("UNITREE_OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing or empty")
        url = _responses_url()
        started = time.time()
        response = httpx.post(url, json=body, timeout=600,
                              headers={"Authorization": f"Bearer {api_key}"})
        if response.status_code == 401:
            raise RuntimeError(f"401 Unauthorized from {url}: {response.text[:300]}")
        response.raise_for_status()
        data = response.json()
        self.calls += 1
        self.seconds += time.time() - started
        output = "".join(
            part.get("text", "")
            for item in data.get("output", []) if item.get("type") == "message"
            for part in item.get("content", [])
        )
        parsed = json.loads(output)
        self.history.append(f"{parsed['skill']}: {parsed['note']}")
        return Decision(parsed["skill"], note=parsed["note"], raw=parsed)
