import asyncio, time
import numpy as np
from queue_engine import TranscriptionQueue


def run(coro):
    return asyncio.run(coro)


def test_priorite_premier_bloc():
    order = []

    def fake(audio, prompt):
        time.sleep(0.05); order.append(len(audio)); return "x"

    async def go():
        q = TranscriptionQueue(fake, workers=1)
        await q.start()
        a = np.zeros(1, dtype=np.float32)
        # occupe le worker, puis met en file un bloc 3 (long) puis un bloc 0 (court) arrivé après
        t0 = asyncio.create_task(q.submit("A", 1, np.zeros(10, dtype=np.float32)))
        await asyncio.sleep(0.01)
        t1 = asyncio.create_task(q.submit("B", 2, np.zeros(20, dtype=np.float32)))
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(q.submit("C", 0, np.zeros(30, dtype=np.float32)))
        await asyncio.gather(t0, t1, t2)
        await q.stop()
    run(go())
    assert order == [10, 30, 20]           # le bloc seq=0 (30) double le bloc seq=2 (20)


def test_resultat_et_stats():
    async def go():
        q = TranscriptionQueue(lambda a, p: "bonjour " + p, workers=2)
        await q.start()
        r = await q.submit("S", 0, np.zeros(16000, dtype=np.float32), "ctx")
        await q.stop()
        return r, q.stats
    r, st = run(go())
    assert r["text"] == "bonjour ctx" and r["audio_s"] == 1.0 and st["done"] == 1


def test_erreur_propagee():
    def boom(a, p): raise ValueError("kaput")
    async def go():
        q = TranscriptionQueue(boom, workers=1)
        await q.start()
        try:
            await q.submit("S", 0, np.zeros(10, dtype=np.float32)); return None
        except ValueError as e:
            return str(e)
        finally:
            await q.stop()
    assert run(go()) == "kaput"


def test_file_pleine():
    async def go():
        q = TranscriptionQueue(lambda a, p: "", workers=1, max_queue=0)
        try:
            await q.submit("S", 0, np.zeros(1, dtype=np.float32)); return False
        except RuntimeError:
            return True
    assert run(go())
