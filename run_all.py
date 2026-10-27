# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Validation code for: "Towards a Classification of the Free Group Factors L_K(F_k)
# over R, C and H as Real von Neumann Algebras" (Cooperman & Petrunina V.V., 27 October 2026).

"""Run every validation script for the corrected white paper and print one summary.

Usage:  python run_all.py          (Python 3.9+, NumPy; about 1-2 minutes)
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    ("Section 1  Q-structure and Brauer parity", "validate_section1.py"),
    ("Section 2  Majorana degree and DLA size", "validate_section2.py"),
    ("Section 3  Interpolated free group factors", "validate_section3.py"),
    ("Section 4  Exercise 6 and complexification", "validate_section4.py"),
    ("Section 5  beta-microstate free entropy", "validate_section5.py"),
]

results = []
for title, script in SCRIPTS:
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, capture_output=True, text=True)
    dt = time.time() - t0
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    last = [l for l in proc.stdout.splitlines() if "checks passed" in l]
    results.append((title, proc.returncode == 0, last[-1].split(": ", 1)[-1] if last else "crashed", dt))

print("=" * 72)
print("SUMMARY")
print("=" * 72)
for title, ok, line, dt in results:
    print(f"{'PASS' if ok else 'FAIL'}  {title:<46} {line:<26} {dt:5.1f}s")
sys.exit(0 if all(ok for _, ok, _, _ in results) else 1)
