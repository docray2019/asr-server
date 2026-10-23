"""Comparaison de deux pipelines de transcription sur des consultations de test.

  A : MedTranscript tel qu'en production — Whisper large-v3 sur le serveur GPU (/transcribe)
  B : serveur CPU (ce dépôt) — Parakeet (ou Whisper) via /v1/chunk
Même découpage « en direct » pour les deux (blocs de 8 s, recouvrement de 1 s), puis la MÊME chaîne
MedTranscript : analyse Haiku (/api/analyze, étapes à 25 s, 4 min, 7 min + finale) → observation Sonnet 5.5
(/api/transcribe partie=observation) → courrier + résumé patient Haiku 5.5 (partie=suite), avec le profil
d'un utilisateur donné (--as-user). Chaque observation est comparée à l'observation certifiée du dossier.

Dossier racine : sous-dossiers dont le nom commence par un nombre (1 à 55), chacun avec un .wav
(16 kHz mono 16 bits) et un .txt (observation certifiée).

Identification : MT_TOKEN (jeton de session admin) OU MT_EMAIL + MT_PASSWORD (compte admin sans 2FA).
Exemple (prétest sur les dossiers 1 à 3) :
  python compare_pipelines.py "C:\\...\\Epaule_MaxLaporte" --only 1-3 --vps-url http://127.0.0.1:8000 --vps-token essai --judge
"""
import argparse, concurrent.futures as cf, csv, difflib, html, json, os, re, sys, time
from pathlib import Path
import httpx
from bench import blocks, wav_bytes
from merge import merge_overlap

STAGE_MARKS = [(1, 25), (2, 240), (3, 420)]      # étapes d'analyse de l'application (secondes de dictée)


# ── Sélection des dossiers ─────────────────────────────────────────────────────
def parse_only(spec):
    if not spec:
        return None
    keep = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        keep.update(range(int(a), int(b or a) + 1))
    return keep


def find_cases(root, only=None):
    cases = []
    for d in sorted(Path(root).iterdir()):
        m = re.match(r"^(\d+)", d.name)
        if not d.is_dir() or not m or not (1 <= int(m.group(1)) <= 55):
            continue
        n = int(m.group(1))
        if only and n not in only:
            continue
        wavs, txts = sorted(d.glob("*.wav")), sorted(d.glob("*.txt"))
        cases.append({"n": n, "dir": d, "wav": wavs[0] if wavs else None, "txt": txts[0] if txts else None,
                      "warn": ("plusieurs .wav" if len(wavs) > 1 else "") + (" plusieurs .txt" if len(txts) > 1 else "")})
    return sorted(cases, key=lambda c: c["n"])


# ── HTTP avec reprises ─────────────────────────────────────────────────────────
def call(client, method, url, retries=3, **kw):
    last = None
    for i in range(retries):
        try:
            r = client.request(method, url, **kw)
            if r.status_code in (502, 503, 504) and i < retries - 1:
                time.sleep(2 * (i + 1)); continue
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            last = e
            if i < retries - 1:
                time.sleep(2 * (i + 1))
    detail = ""
    if isinstance(last, httpx.HTTPStatusError):
        detail = " — " + last.response.text[:300]
    raise RuntimeError(f"{method} {url} : {last}{detail}")


# ── Transcription en blocs de 8 s / recouvrement 1 s ───────────────────────────
def transcribe(cuts, send, label=""):
    """send(pcm_wav_bytes, seq, prompt) -> texte. Renvoie (parts, durées_par_bloc, wall_s)."""
    parts, lat, ctx = [], [], ""
    t0 = time.perf_counter()
    for seq, (pcm, _nominal) in enumerate(cuts):
        t = time.perf_counter()
        text = send(wav_bytes(pcm), seq, ctx)
        lat.append(time.perf_counter() - t)
        text = merge_overlap(" ".join(parts), text) if seq else text
        parts.append(text)
        ctx = (ctx + " " + text).strip()[-400:]
        if label and ((seq + 1) % 5 == 0 or seq + 1 == len(cuts)):
            print(f"     {label} : bloc {seq + 1}/{len(cuts)}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    return parts, lat, time.perf_counter() - t0


def text_upto(parts, cuts, seconds):
    acc, out = 0.0, []
    for p, (_pcm, nominal) in zip(parts, cuts):
        acc += nominal
        out.append(p)
        if acc >= seconds:
            break
    return " ".join(x for x in out if x).strip()


# ── Chaîne MedTranscript (analyse → observation → suite) ───────────────────────
def run_chain(client, app, headers, profile, as_user, parts, cuts, duration_s):
    transcript = " ".join(x for x in parts if x).strip()
    analysis, log = {}, []
    t0 = time.perf_counter()
    stages = [(s, sec, False) for s, sec in STAGE_MARKS if duration_s >= sec] + [(3, None, True)]
    for stage, sec, final in stages:
        txt = transcript if final else text_upto(parts, cuts, sec)
        if not txt.strip():
            continue
        r = call(client, "POST", f"{app}/api/analyze", headers=headers, timeout=120,
                 json={"transcript": txt, "stage": stage, "previousAnalysis": analysis, "final": final})
        for k, v in (r.get("analysis") or {}).items():
            if v is not None:
                analysis[k] = v
        log.append({"stage": stage, "final": final})
    t_analyse = time.perf_counter() - t0

    base = {"transcription": transcript, "profile": profile, "analysis": analysis, "histoirePrealable": "",
            "duration_s": duration_s, "as_user_email": as_user}
    t1 = time.perf_counter()
    obs = call(client, "POST", f"{app}/api/transcribe", headers=headers, timeout=300,
               json={**base, "partie": "observation", "bench_model": "sonnet"})
    t_obs = time.perf_counter() - t1
    t2 = time.perf_counter()
    suite = call(client, "POST", f"{app}/api/transcribe", headers=headers, timeout=300,
                 json={**base, "partie": "suite", "observation_redigee": obs["observation"]})
    t_suite = time.perf_counter() - t2
    return {"transcript": transcript, "analysis": analysis, "observation": obs["observation"],
            "suite": suite["observation"], "skill": obs.get("skill"), "consult_type": obs.get("consult_type"),
            "model_obs": obs.get("model"), "model_suite": suite.get("model"),
            "t_analyse": t_analyse, "t_obs": t_obs, "t_suite": t_suite}


def sim(a, b):
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3) if a and b else None


# ── Un dossier ─────────────────────────────────────────────────────────────────
def read_text_any(path):
    """Lit un .txt en UTF-8 (avec ou sans BOM), sinon en Windows-1252 (Bloc-notes / Word)."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def cr_text(res, k):
    """Texte soumis au juge : l'observation, à défaut le courrier généré."""
    r = res.get(k) or {}
    return (r.get("observation") or "").strip() or (r.get("suite") or "").strip()


def judge_case(res, ref, cfg, headers):
    for k in ("A", "B"):
        if k in res and cr_text(res, k) and not res[k].get("judge"):
            print(f"     juge {cfg.judge_model} : {k}…", flush=True)
            with httpx.Client() as c:
                j = call(c, "POST", f"{cfg.app_url}/api/admin/judge-cr", headers=headers, timeout=300,
                         json={"transcript": res[k]["transcript"], "reference": ref, "candidate": cr_text(res, k), "judge_model": cfg.judge_model})
            res[k]["judge"] = j.get("judge")


def duel_case(res, ref, cfg, headers):
    """Opus compare A et B entre eux et avec la référence, en aveugle (ordre X/Y alterné selon le n° du cas)."""
    if res.get("duel") or not all(cr_text(res, k) for k in ("A", "B")):
        return
    first, second = ("A", "B") if res["n"] % 2 else ("B", "A")
    print(f"     duel {cfg.judge_model} : A contre B (aveugle)…", flush=True)
    with httpx.Client() as c:
        j = call(c, "POST", f"{cfg.app_url}/api/admin/judge-cr", headers=headers, timeout=300, json={
            "mode": "pair", "reference": ref, "judge_model": cfg.judge_model,
            "transcript_x": res[first]["transcript"], "candidate_x": cr_text(res, first),
            "transcript_y": res[second]["transcript"], "candidate_y": cr_text(res, second)})
    d = j.get("judge") or {}
    if d.get("parse_error"):
        res["duel"] = d
        return
    m = {"X": first, "Y": second}
    best = d.get("meilleur")
    out = {"gagnant": m.get(best, "egal"), "ordre": f"X={first}, Y={second}",
           "origine_ecarts": d.get("origine_ecarts"), "verdict": d.get("verdict")}
    for key in ("score", "fidelite", "completude", "erreurs"):
        for xy, k in m.items():
            out[f"{key}_{k}"] = d.get(f"{key}_{xy}")
    res["duel"] = out


def run_case(case, cfg, headers, profile, out_dir):
    ref = read_text_any(case["txt"]).strip()
    cuts = blocks(str(case["wav"]), 8, 8, 1.0)
    duration_s = round(sum(n for _p, n in cuts), 1)
    res = {"n": case["n"], "name": case["dir"].name, "duration_s": duration_s, "blocks": len(cuts), "reference": ref}
    specialty = profile.get("specialty") or "epaule"

    def send_a(wavb, seq, ctx):
        with httpx.Client(timeout=httpx.Timeout(120, connect=10)) as c:
            return call(c, "POST", f"{cfg.gpu_url}/transcribe", headers=headers,
                        files={"file": ("chunk.wav", wavb, "audio/wav")}, data={"specialty": specialty}).get("text", "").strip()

    def send_b(wavb, seq, ctx):
        with httpx.Client(timeout=httpx.Timeout(600, connect=10)) as c:
            return call(c, "POST", f"{cfg.vps_url}/v1/chunk", headers=({"Authorization": f"Bearer {cfg.vps_token}"} if cfg.vps_token else {}),
                        files={"wav": ("b.wav", wavb, "audio/wav")},
                        data={"session": f"case{case['n']}", "seq": seq, "prompt": ctx}).get("text", "").strip()

    pipes = {}
    if not cfg.skip_a: pipes["A"] = send_a
    if not cfg.skip_b: pipes["B"] = send_b
    with cf.ThreadPoolExecutor(2) as ex:
        futs = {k: ex.submit(transcribe, cuts, fn, k) for k, fn in pipes.items()}
        asr = {k: f.result() for k, f in futs.items()}
    for k, (parts, lat, wall) in asr.items():
        res[k] = {"asr_wall_s": round(wall, 1), "asr_x_realtime": round(duration_s / wall, 2) if wall else None,
                  "chunk_latency_mean_s": round(sum(lat) / len(lat), 2), "chunk_latency_max_s": round(max(lat), 2)}
    with cf.ThreadPoolExecutor(2) as ex:
        futs = {k: ex.submit(run_chain, httpx.Client(), cfg.app_url, headers, profile, cfg.as_user, asr[k][0], cuts, duration_s)
                for k in asr}
        for k, f in futs.items():
            res[k].update(f.result())

    if cfg.judge:
        if cfg.solo_judge: judge_case(res, ref, cfg, headers)
        duel_case(res, ref, cfg, headers)
    if "A" in res and "B" in res:
        res["sim_transcript_AB"] = sim(res["A"]["transcript"], res["B"]["transcript"])
        res["sim_obs_AB"] = sim(res["A"]["observation"], res["B"]["observation"])
    for k in asr:
        res[k]["sim_obs_ref"] = sim(res[k]["observation"], ref)

    d = out_dir / f"{case['n']:02d}_{re.sub(r'[^\w-]+', '_', case['dir'].name)[:40]}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "reference.txt").write_text(ref, encoding="utf-8")
    for k in asr:
        for part in ("transcript", "observation", "suite"):
            (d / f"{k}_{part}.txt").write_text(res[k][part], encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# ── Rapport ────────────────────────────────────────────────────────────────────
def pre(t):
    return f"<pre>{html.escape(t or '')}</pre>"


def write_report(results, out_dir):
    cols = ["n", "name", "duration_s", "A_x_realtime", "B_x_realtime", "sim_transcript_AB", "sim_obs_AB",
            "A_sim_obs_ref", "B_sim_obs_ref", "A_judge", "B_judge", "duel_gagnant", "duel_score_A", "duel_score_B", "duel_origine", "A_skill", "B_skill"]
    with open(out_dir / "resume.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";"); w.writerow(cols)
        for r in results:
            g = lambda k, f_: (r.get(k) or {}).get(f_)
            w.writerow([r["n"], r["name"], r["duration_s"], g("A", "asr_x_realtime"), g("B", "asr_x_realtime"),
                        r.get("sim_transcript_AB"), r.get("sim_obs_AB"), g("A", "sim_obs_ref"), g("B", "sim_obs_ref"),
                        (g("A", "judge") or {}).get("score_global"), (g("B", "judge") or {}).get("score_global"),
                        (r.get("duel") or {}).get("gagnant"), (r.get("duel") or {}).get("score_A"), (r.get("duel") or {}).get("score_B"),
                        (r.get("duel") or {}).get("origine_ecarts"), g("A", "skill"), g("B", "skill")])
    parts = ["<!doctype html><html lang=fr><head><meta charset=utf-8><title>Comparaison A/B</title><style>body{font-family:system-ui;margin:20px}"
             "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;vertical-align:top}"
             "pre{white-space:pre-wrap;font-size:12px;margin:0}.g{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}"
             "details{margin:6px 0}</style></head><body><h1>A (GPU Whisper large-v3) vs B (CPU)</h1>"]
    w = [(r.get("duel") or {}).get("gagnant") for r in results]
    parts.append(f"<p><b>Duels Opus :</b> A gagne {w.count('A')} · B gagne {w.count('B')} · égalité {w.count('egal')} sur {sum(1 for x in w if x)} cas jugés.</p>")
    for r in results:
        parts.append(f"<h2>{html.escape(r['name'])} — {r['duration_s']} s</h2><table><tr><th></th><th>A</th><th>B</th></tr>")
        for lab, f_ in [("Transcription (× temps réel)", "asr_x_realtime"), ("Latence moyenne / bloc (s)", "chunk_latency_mean_s"),
                        ("Skill", "skill"), ("Type", "consult_type"), ("Similarité obs. / référence", "sim_obs_ref")]:
            parts.append(f"<tr><td>{lab}</td><td>{(r.get('A') or {}).get(f_)}</td><td>{(r.get('B') or {}).get(f_)}</td></tr>")
        for k in ("A", "B"):
            j = (r.get(k) or {}).get("judge")
        parts.append("<tr><td>Juge Opus (global / fidélité / complétude)</td>" + "".join(
            f"<td>{html.escape(json.dumps({x: (r.get(k) or {}).get('judge', {}).get(x) for x in ('score_global','fidelite_clinique','completude')}) if (r.get(k) or {}).get('judge') else '—')}</td>"
            for k in ("A", "B")) + "</tr></table>")
        d = r.get("duel")
        if d and not d.get("parse_error"):
            parts.append(f"<p><b>Duel Opus (aveugle) : gagnant = {html.escape(str(d.get('gagnant')))}</b> — score A {d.get('score_A')} / B {d.get('score_B')}"
                         f" · fidélité A {d.get('fidelite_A')} / B {d.get('fidelite_B')} · complétude A {d.get('completude_A')} / B {d.get('completude_B')}"
                         f" · origine des écarts : {html.escape(str(d.get('origine_ecarts')))}<br>{html.escape(str(d.get('verdict') or ''))}<br>"
                         f"<i>Erreurs A :</i> {html.escape(' ; '.join(map(str, d.get('erreurs_A') or [])) or '—')}<br>"
                         f"<i>Erreurs B :</i> {html.escape(' ; '.join(map(str, d.get('erreurs_B') or [])) or '—')}</p>")
        parts.append("<details open><summary>Observations : référence | A | B</summary><div class=g>"
                     + pre(r["reference"]) + pre((r.get("A") or {}).get("observation")) + pre((r.get("B") or {}).get("observation")) + "</div></details>")
        parts.append("<details><summary>Courrier + résumé : A | B</summary><div class=g><div></div>"
                     + pre((r.get("A") or {}).get("suite")) + pre((r.get("B") or {}).get("suite")) + "</div></details>")
        parts.append("<details><summary>Transcriptions : A | B</summary><div class=g><div></div>"
                     + pre((r.get("A") or {}).get("transcript")) + pre((r.get("B") or {}).get("transcript")) + "</div></details>")
        for k in ("A", "B"):
            j = (r.get(k) or {}).get("judge")
            if j and j.get("erreurs_notables"):
                parts.append(f"<p><b>{k} — erreurs notables :</b> {html.escape(' ; '.join(map(str, j['erreurs_notables'])))}</p>")
    (out_dir / "rapport.html").write_text("".join(parts), encoding="utf-8")


# ── Principal ──────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--only", help="ex. 1-3 ou 1,4,7-9")
    ap.add_argument("--app-url", default=os.environ.get("MT_URL", "https://medtranscript-next.vercel.app"))
    ap.add_argument("--gpu-url", default=os.environ.get("GPU_URL", "https://whisper.medtranscript.fr"))
    ap.add_argument("--cpu-url", "--vps-url", dest="vps_url", default="http://127.0.0.1:8000", help="serveur CPU (B), via le tunnel SSH")
    ap.add_argument("--cpu-token", "--vps-token", dest="vps_token", default=os.environ.get("ASR_TOKEN", ""), help="jeton du serveur CPU (ASR_TOKEN)")
    ap.add_argument("--as-user", default="drmaximelaporte@runbox.com")
    ap.add_argument("--out", default="resultats"); ap.add_argument("--judge", dest="judge", action="store_true", default=True, help=argparse.SUPPRESS)
    ap.add_argument("--solo-judge", action="store_true", help="en plus du duel, une note Opus séparée pour A et B (plus de tokens)")
    ap.add_argument("--no-judge", dest="judge", action="store_false", help="désactive le juge Opus (activé par défaut)")
    ap.add_argument("--judge-model", default="claude-opus-5-5", help="claude-opus-5-5 (défaut) ou claude-opus-4-8")
    ap.add_argument("--skip-a", action="store_true"); ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--resume", action="store_true", help="ne refait pas les dossiers déjà traités")
    ap.add_argument("--dry-run", action="store_true", help="liste les dossiers et fichiers trouvés, sans appel")
    cfg = ap.parse_args(argv)

    from urllib.parse import urlparse
    gh, ch = urlparse(cfg.gpu_url), urlparse(cfg.vps_url)
    if (gh.hostname, gh.port) == (ch.hostname, ch.port) and not (cfg.skip_a or cfg.skip_b):
        sys.exit(f"Les serveurs GPU (A) et CPU (B) pointent vers la même adresse ({cfg.gpu_url}) : vérifiez --gpu-url et --cpu-url.")
    print(f"A = GPU (production, Whisper large-v3) : {cfg.gpu_url}{'  [IGNORÉ]' if cfg.skip_a else ''}")
    print(f"B = CPU (test, serveur asr-server)     : {cfg.vps_url}{'  [IGNORÉ]' if cfg.skip_b else ''}")
    print(f"Application MedTranscript              : {cfg.app_url}")
    cases = find_cases(cfg.root, parse_only(cfg.only))
    print(f"{len(cases)} dossier(s) :")
    bad = False
    for c in cases:
        miss = [x for x, v in (("wav", c["wav"]), ("txt", c["txt"])) if not v]
        bad |= bool(miss)
        print(f"  {c['n']:>2}  {c['dir'].name}  wav={c['wav'].name if c['wav'] else '—'}  txt={c['txt'].name if c['txt'] else '—'}"
              f"{'  MANQUE ' + ','.join(miss) if miss else ''}{'  ' + c['warn'] if c['warn'] else ''}")
    if cfg.dry_run or not cases:
        return 0
    cases = [c for c in cases if c["wav"] and c["txt"]]

    def reachable(url, path):
        try:
            r = httpx.get(url.rstrip("/") + path, timeout=httpx.Timeout(15, connect=8))
            return True, r
        except httpx.HTTPError as e:
            return False, e
    if not cfg.skip_b:
        ok, r = reachable(cfg.vps_url, "/health")
        if not ok: sys.exit(f"Serveur CPU (B) injoignable à {cfg.vps_url} : tunnel SSH ouvert ? serveur lancé ? ({r})")
        try: h = r.json(); print(f"CPU OK : moteur {h.get('engine')} / modèle {h.get('model')}, {h.get('workers')} worker(s)")
        except ValueError: print(f"CPU répond (HTTP {r.status_code}) mais pas comme attendu sur /health")
    if not cfg.skip_a:
        ok, r = reachable(cfg.gpu_url, "/health")
        if not ok: sys.exit(f"Serveur GPU (A) injoignable à {cfg.gpu_url} : est-il allumé ? ({r})")
        print(f"GPU répond (HTTP {r.status_code})")

    token = os.environ.get("MT_TOKEN")
    if not token:
        with httpx.Client() as c:
            r = c.post(f"{cfg.app_url}/api/auth", json={"email": os.environ.get("MT_EMAIL", ""), "password": os.environ.get("MT_PASSWORD", "")})
        if r.status_code != 200 or not r.json().get("token"):
            sys.exit(f"Connexion impossible ({r.status_code}) : {r.text[:200]}\nDéfinissez MT_TOKEN (jeton de session admin) si le compte a la 2FA.")
        token = r.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client() as c:
        profile = call(c, "GET", f"{cfg.app_url}/api/admin/user-profile", headers=headers, params={"email": cfg.as_user})["profile"]
    print(f"Profil de {cfg.as_user} : {profile}")

    out_dir = Path(cfg.out) / time.strftime("%Y%m%d_%H%M%S") if not cfg.resume else Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for c in cases:
        done = list(out_dir.glob(f"{c['n']:02d}_*/metrics.json")) if cfg.resume else []
        if done:
            r = json.loads(done[0].read_text(encoding="utf-8"))
            r["reference"] = read_text_any(c["txt"]).strip()  # corrige d'anciennes références mal décodées
            if cfg.judge and ((cfg.solo_judge and any(k in r and not r[k].get("judge") for k in ("A", "B"))) or not r.get("duel")):
                try:
                    if cfg.solo_judge: judge_case(r, r["reference"], cfg, headers)
                    duel_case(r, r["reference"], cfg, headers)
                    done[0].write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
                except Exception as e:
                    print(f"     juge impossible : {e}")
            results.append(r); print(f"[{c['n']}] déjà traité"); continue
        print(f"[{c['n']}] {c['dir'].name} …", flush=True)
        try:
            r = run_case(c, cfg, headers, profile, out_dir)
            results.append(r)
            print(f"     A {(r.get('A') or {}).get('asr_x_realtime')}× | B {(r.get('B') or {}).get('asr_x_realtime')}× temps réel ; "
                  f"similarité transcriptions {r.get('sim_transcript_AB')} ; observations/réf A {(r.get('A') or {}).get('sim_obs_ref')} B {(r.get('B') or {}).get('sim_obs_ref')}")
        except Exception as e:  # noqa: BLE001
            print(f"     ÉCHEC : {e}")
    results.sort(key=lambda r: r["n"])
    write_report(results, out_dir)
    print(f"\nRésultats dans {out_dir / 'rapport.html'} (et resume.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
