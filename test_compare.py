"""Test de bout en bout du script de comparaison contre de faux serveurs (aucun appel réel)."""
import json, threading, time, wave
import numpy as np, uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
import compare_pipelines as C

calls = []


def fake_servers():
    app = FastAPI()

    @app.post("/api/auth")
    async def auth(r: Request): return {"token": "tok"}

    @app.get("/api/admin/user-profile")
    async def prof(email: str): return {"profile": {"specialty": "epaule", "doctor_name": "Dr Test", "specialty_title": "Chirurgie de l'Épaule", "societies": ""}}

    @app.post("/api/analyze")
    async def analyze(r: Request):
        b = await r.json(); calls.append(("analyze", b["stage"], b["final"]))
        return {"analysis": {"cote": "droit", "type_consultation": "suivi"}}

    @app.post("/api/transcribe")
    async def transcribe(r: Request):
        b = await r.json(); calls.append(("transcribe", b["partie"], b.get("bench_model"), b.get("as_user_email")))
        if b["partie"] == "suite": assert b["observation_redigee"].startswith("OBS")
        return {"observation": ("OBS " if b["partie"] == "observation" else "SUITE ") + b["transcription"][:20], "skill": "S", "consult_type": "suivi", "model": "m"}

    @app.post("/api/admin/judge-cr")
    async def judge(r: Request): return {"judge": {"score_global": 7, "fidelite_clinique": 8, "completude": 6, "erreurs_notables": ["x"]}}

    @app.post("/gpu/transcribe")
    async def gpu(file: UploadFile = File(...), specialty: str = Form("")): return {"text": f"gpu {specialty}"}

    @app.post("/v1/chunk")
    async def chunk(wav: UploadFile = File(...), session: str = Form(...), seq: int = Form(...), prompt: str = Form("")): return {"text": f"cpu {seq}"}
    return app


def test_bout_en_bout(tmp_path, monkeypatch):
    srv = uvicorn.Server(uvicorn.Config(fake_servers(), host="127.0.0.1", port=8771, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start(); time.sleep(1.2)
    root = tmp_path / "Epaule"
    for n, name in ((1, "1 capsulite"), (2, "2 coiffe"), (60, "60 hors plage"), (3, "autre")):
        d = root / name; d.mkdir(parents=True)
        w = wave.open(str(d / "x.wav"), "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.zeros(16000 * 30, dtype=np.int16).tobytes()); w.close()
        (d / "ref.txt").write_text("OBS référence", encoding="utf-8")
    monkeypatch.setenv("MT_EMAIL", "a"); monkeypatch.setenv("MT_PASSWORD", "b")
    out = tmp_path / "res"
    rc = C.main([str(root), "--only", "1-2", "--app-url", "http://127.0.0.1:8771", "--gpu-url", "http://localhost:8771/gpu",
                 "--vps-url", "http://127.0.0.1:8771", "--judge", "--out", str(out)])
    assert rc == 0
    run = next(out.iterdir())
    assert (run / "rapport.html").exists() and (run / "resume.csv").exists()
    assert len(list(run.glob("0*_*/metrics.json"))) == 2          # dossiers 1 et 2 seulement
    m = json.loads(next(run.glob("01_*/metrics.json")).read_text(encoding="utf-8"))
    assert m["A"]["judge"]["score_global"] == 7 and m["blocks"] == 4   # 30 s en blocs de 8 s
    assert ("transcribe", "observation", "sonnet", "drmaximelaporte@runbox.com") in calls
    assert ("transcribe", "suite", None, "drmaximelaporte@runbox.com") in calls
    assert ("analyze", 1, False) in calls and ("analyze", 3, True) in calls


def test_selection():
    assert C.parse_only("1-3,7") == {1, 2, 3, 7}
