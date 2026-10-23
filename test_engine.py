import os, importlib
import numpy as np


def test_dispatch_parakeet(monkeypatch):
    monkeypatch.setenv("ASR_ENGINE", "parakeet"); monkeypatch.setenv("ASR_NO_AUTOSTART", "1")
    import app
    importlib.reload(app)
    called = {}
    monkeypatch.setattr(app, "transcribe_parakeet", lambda a, p: called.setdefault("x", "texte"))
    assert app.ENGINE == "parakeet" and app.MODEL_NAME.startswith("nemo-parakeet")
    assert app.transcribe_blocking(np.zeros(16000, dtype=np.float32), "ctx") == "texte"


def test_defaut_whisper(monkeypatch):
    monkeypatch.delenv("ASR_ENGINE", raising=False); monkeypatch.setenv("ASR_NO_AUTOSTART", "1")
    import app
    importlib.reload(app)
    assert app.ENGINE == "whisper" and app.MODEL_NAME == "medium"
