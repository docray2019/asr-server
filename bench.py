"""Banc d'essai : N dictées simultanées (par défaut 4) découpées en blocs de 30 s
(1er bloc de 8 s, comme l'application), envoyées au serveur comme le ferait le navigateur,
au rythme de la parole (le bloc k part quand il a été « dicté »).

  python bench.py audio.wav --url http://IP:8000 --token XXX --sessions 4
Affiche, par bloc : attente en file, calcul, retard total par rapport à la fin du bloc.
Le verdict : le retard reste-t-il stable (le serveur suit) ou grandit-il (il est saturé) ?
"""
import argparse, asyncio, io, time, wave
import httpx
from merge import merge_overlap


def blocks(path, first_s=8, size_s=30, overlap_s=0.0):
    """Découpe en blocs ; chaque bloc (sauf le 1er) commence `overlap_s` plus tôt.
    Renvoie [(pcm, durée_nominale_s)] : la durée nominale sert à cadencer l'envoi."""
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, "WAV 16 kHz mono 16 bits requis"
        pcm = w.readframes(w.getnframes())
    step = 32000
    ov = int(overlap_s * step) // 2 * 2                      # reste aligné sur des échantillons de 16 bits
    out, pos, first = [], 0, True
    while pos < len(pcm):
        n = (first_s if first else size_s) * step
        start = max(0, pos - ov) if not first else 0
        out.append((pcm[start:pos + n], min(n, len(pcm) - pos) / step))
        pos += n; first = False
    return out


def wav_bytes(pcm):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
    return b.getvalue()


async def session(name, cuts, url, token, t_start, results, speed, texts, a_overlap=0.0):
    prompt = ""
    texts[name] = []
    async with httpx.AsyncClient(timeout=600) as c:
        ready_at = 0.0
        for seq, (pcm, nominal_s) in enumerate(cuts):
            ready_at += nominal_s / speed                  # instant où le bloc a fini d'être « dicté »
            wait = t_start + ready_at - time.perf_counter()
            if wait > 0: await asyncio.sleep(wait)
            r = await c.post(f"{url}/v1/chunk", headers={"Authorization": f"Bearer {token}"},
                             data={"session": name, "seq": seq, "prompt": prompt},
                             files={"wav": ("b.wav", wav_bytes(pcm), "audio/wav")})
            r.raise_for_status()
            j = r.json()
            lag = time.perf_counter() - (t_start + ready_at)
            results.append((name, seq, j["audio_s"], j["queue_ms"] / 1000, j["compute_ms"] / 1000, lag))
            print(f"{name} bloc {seq}: audio {j['audio_s']:5.1f}s  file {j['queue_ms']/1000:5.1f}s  calcul {j['compute_ms']/1000:5.1f}s  retard {lag:5.1f}s")
            merged = merge_overlap(" ".join(texts[name]), j["text"]) if seq and a_overlap else j["text"]
            texts[name].append(merged)
            prompt = (prompt + " " + merged).strip()[-400:]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav"); ap.add_argument("--url", required=True); ap.add_argument("--token", default="")
    ap.add_argument("--sessions", type=int, default=4); ap.add_argument("--overlap", type=float, default=0.0, help="recouvrement entre blocs (s), ex. 1"); ap.add_argument("--first", type=int, default=8, help="durée du 1er bloc (s)"); ap.add_argument("--size", type=int, default=30, help="durée des blocs suivants (s) ; 8 pour tout en blocs de 8 s"); ap.add_argument("--out", default="transcription", help="préfixe des fichiers texte écrits (un par session)"); ap.add_argument("--speed", type=float, default=1.0,
                    help="1 = temps réel ; 10 = dictées accélérées (test de saturation rapide)")
    a = ap.parse_args()
    cuts = blocks(a.wav, a.first, a.size, a.overlap)
    print(f"{len(cuts)} blocs par session, {a.sessions} sessions simultanées")
    res, texts, t0 = [], {}, time.perf_counter()
    await asyncio.gather(*[session(f"S{i+1}", cuts, a.url, a.token, t0, res, a.speed, texts, a.overlap) for i in range(a.sessions)])
    for name, parts in texts.items():
        fn = f"{a.out}_{name}.txt"
        with open(fn, "w", encoding="utf-8") as f:
            f.write("\n\n".join(f"[bloc {i}] {t}" for i, t in enumerate(parts)))
        print(f"\nTexte écrit dans {fn}")
    if texts:
        first = next(iter(texts.values()))
        print("\n===== TRANSCRIPTION (session S1) =====")
        for i, t in enumerate(first):
            print(f"[bloc {i}] {t}\n")
    comp = sum(r[4] for r in res); aud = sum(r[2] for r in res)
    print(f"\nVitesse de calcul : {aud/comp:.2f}× la parole par transcription ; retard max {max(r[5] for r in res):.1f}s ; file max {max(r[3] for r in res):.1f}s")
    last = [r[5] for r in res if r[1] >= len(cuts) - 2]
    print("Verdict :", "le serveur SUIT" if max(last) < 60 else "le serveur est SATURÉ (le retard grandit)")

if __name__ == "__main__":
    asyncio.run(main())
