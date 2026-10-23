"""Serveur de transcription MedTranscript (CPU) — faster-whisper + file d'attente.

POST /v1/chunk   (multipart) : session, seq, wav (16 kHz mono 16 bits), prompt (texte précédent, facultatif)
                  → { text, queue_ms, compute_ms, audio_s, seq }
GET  /health     → état, modèle, file
Aucun audio n'est écrit sur disque ni conservé.
"""
import io, os, wave
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from queue_engine import TranscriptionQueue

ENGINE = os.environ.get("ASR_ENGINE", "whisper")                # "whisper" (faster-whisper) ou "parakeet" (onnx-asr)
MODEL_NAME = os.environ.get("ASR_MODEL", "medium" if ENGINE == "whisper" else "nemo-parakeet-tdt-0.6b-v3")
COMPUTE = os.environ.get("ASR_COMPUTE", "int8")
CPU_THREADS = int(os.environ.get("ASR_CPU_THREADS", "4"))     # threads par transcription
WORKERS = int(os.environ.get("ASR_WORKERS", "1"))             # transcriptions simultanées
TOKEN = os.environ.get("ASR_TOKEN", "")
BEAM = int(os.environ.get("ASR_BEAM", "1"))
VOCAB = os.environ.get("ASR_VOCAB", "prothèse totale de hanche, prothèse totale de genou, ligament croisé antérieur, "
                       "gonarthrose, coxarthrose, arthroplastie, ostéotomie, ménisque, cotyle, fémoral")
MAX_AUDIO_S = 40

_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE,
                              cpu_threads=CPU_THREADS, num_workers=WORKERS)
    return _model


_parakeet = None


def get_parakeet():
    """Parakeet TDT 0.6B v3 via onnx-asr (ONNX Runtime CPU). Pas de remplissage à 30 s :
    le temps de calcul est proportionnel à la durée du bloc. Le prompt n'est pas utilisé."""
    global _parakeet
    if _parakeet is None:
        import onnx_asr
        import onnxruntime as rt
        so = rt.SessionOptions()
        so.intra_op_num_threads = CPU_THREADS
        _parakeet = onnx_asr.load_model(MODEL_NAME, quantization="int8" if COMPUTE == "int8" else None,
                                        sess_options=so)
    return _parakeet


def transcribe_parakeet(audio, prompt):
    return (get_parakeet().recognize(audio, sample_rate=16000) or "").strip()


def transcribe_blocking(audio, prompt):
    if ENGINE == "parakeet":
        return transcribe_parakeet(audio, prompt)
    seg, _ = get_model().transcribe(
        audio, language="fr", beam_size=BEAM, vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=((prompt or "")[-400:] + " " + VOCAB).strip() if prompt else VOCAB,
    )
    return " ".join(s.text.strip() for s in seg).strip()


def read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("WAV 16 kHz mono 16 bits attendu")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def create_app(fn=None):
    queue = TranscriptionQueue(fn or transcribe_blocking, workers=WORKERS)
    app = FastAPI(title="MedTranscript ASR")

    @app.on_event("startup")
    async def _start():
        if fn is None:
            import asyncio
            await asyncio.to_thread(get_parakeet if ENGINE == "parakeet" else get_model)          # charge le modèle avant d'accepter du trafic
        await queue.start()

    def check(auth):
        if TOKEN and auth != f"Bearer {TOKEN}":
            raise HTTPException(401, "non autorisé")

    @app.get("/health")
    async def health():
        s = queue.stats
        return {"ok": True, "engine": ENGINE, "model": MODEL_NAME, "compute": COMPUTE, "workers": WORKERS,
                "cpu_threads": CPU_THREADS, "waiting": queue.waiting, **s,
                "speed_x_realtime": round(s["audio_s_total"] / (s["compute_ms_total"] / 1000), 2) if s["compute_ms_total"] else None}

    @app.post("/v1/chunk")
    async def chunk(session: str = Form(...), seq: int = Form(...), prompt: str = Form(""),
                    wav: UploadFile = File(...), authorization: str = Header("")):
        check(authorization)
        try:
            audio = read_wav(await wav.read())
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e))
        if len(audio) / 16000 > MAX_AUDIO_S:
            raise HTTPException(400, f"bloc trop long (> {MAX_AUDIO_S} s)")
        try:
            return await queue.submit(session, seq, audio, prompt)
        except RuntimeError as e:
            raise HTTPException(503, str(e))

    return app


app = create_app() if os.environ.get("ASR_NO_AUTOSTART") != "1" else None
