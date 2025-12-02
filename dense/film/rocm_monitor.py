#!/usr/bin/env python3

import threading
import subprocess
import time

def monitor():
    while True:
        out = subprocess.check_output(["rocm-smi", "--showuse"], text=True)
        print("[GPU]", out.strip())
        time.sleep(10)


threading.Thread(target=monitor, daemon=True).start()