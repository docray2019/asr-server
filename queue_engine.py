"""File d'attente de transcription : priorité au premier bloc de chaque dictée,
puis ordre d'arrivée. Indépendant de faster-whisper (le moteur est injecté), donc testable."""
import asyncio, itertools, time


class Job:
    __slots__ = ("session", "seq", "audio", "prompt", "prio", "t_in", "fut", "n")

    def __init__(self, session, seq, audio, prompt, prio, fut, n):
        self.session, self.seq, self.audio, self.prompt = session, seq, audio, prompt
        self.prio, self.fut, self.n, self.t_in = prio, fut, n, time.perf_counter()


class TranscriptionQueue:
    def __init__(self, transcribe_fn, workers=2, max_queue=200):
        """transcribe_fn(audio_f32, prompt) -> str (bloquant, exécuté dans un thread)."""
        self._fn = transcribe_fn
        self._workers = workers
        self._max = max_queue
        self._q = asyncio.PriorityQueue()
        self._count = itertools.count()
        self._tasks = []
        self.stats = {"done": 0, "errors": 0, "compute_ms_total": 0, "audio_s_total": 0.0, "running": 0}

    async def start(self):
        self._tasks = [asyncio.create_task(self._worker()) for _ in range(self._workers)]

    async def stop(self):
        for t in self._tasks:
            t.cancel()

    @property
    def waiting(self):
        return self._q.qsize()

    async def submit(self, session, seq, audio, prompt=""):
        if self._q.qsize() >= self._max:
            raise RuntimeError("file d'attente pleine")
        n = next(self._count)
        prio = 0 if seq == 0 else 1          # le 1er bloc de chaque dictée passe devant
        fut = asyncio.get_running_loop().create_future()
        await self._q.put((prio, n, Job(session, seq, audio, prompt, prio, fut, n)))
        return await fut

    async def _worker(self):
        while True:
            _, _, job = await self._q.get()
            t0 = time.perf_counter()
            queue_ms = int((t0 - job.t_in) * 1000)
            self.stats["running"] += 1
            try:
                text = await asyncio.to_thread(self._fn, job.audio, job.prompt)
                ms = int((time.perf_counter() - t0) * 1000)
                audio_s = len(job.audio) / 16000
                self.stats["done"] += 1
                self.stats["compute_ms_total"] += ms
                self.stats["audio_s_total"] += audio_s
                if not job.fut.done():
                    job.fut.set_result({"text": text, "session": job.session, "seq": job.seq,
                                        "queue_ms": queue_ms, "compute_ms": ms, "audio_s": round(audio_s, 2)})
            except Exception as e:  # noqa: BLE001
                self.stats["errors"] += 1
                if not job.fut.done():
                    job.fut.set_exception(e)
            finally:
                self.stats["running"] -= 1
