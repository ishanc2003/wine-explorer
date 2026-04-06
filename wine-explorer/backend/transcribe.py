import tempfile
import os
from faster_whisper import WhisperModel

_model: WhisperModel | None = None

def load_model() -> None:
    global _model
    print("Loading Whisper model (base)...")
    _model = WhisperModel("base", device="cuda", compute_type="float16")
    print("Whisper model loaded.")

def transcribe(audio_bytes: bytes) -> str:
    if _model is None:
        raise RuntimeError("Whisper model not loaded. Call load_model() first.")

    suffix = ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, _ = _model.transcribe(tmp_path, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)

    return text
