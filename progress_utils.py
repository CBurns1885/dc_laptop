# progress_utils.py
import sys, time, datetime as _dt
def ts(): return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
class Timer:
    def __init__(self, name="step"): self.name=name; self.t0=None
    def __enter__(self): self.t0=time.time(); print(f"[{ts()}] ▶ {self.name}..."); sys.stdout.flush(); return self
    def __exit__(self, exc_type, exc, tb):
        dt=time.time()-self.t0; status="OK" if exc is None else f"ERR: {exc}"
        print(f"[{ts()}] ✓ {self.name} done in {dt:.1f}s ({status})"); sys.stdout.flush()
def heartbeat(msg:str): print(f"[{ts()}] … {msg}"); sys.stdout.flush()
