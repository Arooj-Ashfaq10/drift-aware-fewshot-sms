"""Pipeline smoke test with a MOCK model and SYNTHETIC data. Produces no scientific results."""
import os, subprocess, sys, tempfile
import numpy as np

rng = np.random.default_rng(0)
fam = [["free", "prize", "win", "claim", "cash"], ["txt", "reply", "stop", "ringtone", "tone"],
       ["urgent", "call", "account", "bank", "verify"], ["offer", "click", "http", "voucher", "deal"]]
filler = "ok see you later at home today lunch dinner tomorrow thanks sure fine will come now".split()
lines = []
for i in range(750):
    f = fam[i % 4]; lines.append("spam\t" + " ".join(rng.choice(f, 4)) + " " + " ".join(rng.choice(filler, 3)) + f" {i}")
for i in range(4800):
    lines.append("ham\t" + " ".join(rng.choice(filler, 7)) + f" msg{i}")
d = tempfile.mkdtemp(); p = os.path.join(d, "SMSSpamCollection")
open(p, "w", encoding="latin-1").write("\n".join(lines))
for cmd in (
    [sys.executable, "run_reproduction.py", "--backend", "mock", "--uci_path", p, "--out", os.path.join(d, "rep")],
    [sys.executable, "run_drift.py", "--backend", "mock", "--uci_path", p, "--out", os.path.join(d, "res"), "--max_eval", "150"],
):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2500:]); 
    if r.returncode: print(r.stderr[-3000:]); sys.exit(1)
print("SMOKE TEST PASSED; files:", sorted(os.listdir(os.path.join(d, "res", "simulated"))))
