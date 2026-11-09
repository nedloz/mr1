"""Данные МР 1: регион 30 (TER), 43-й квартал, очистка и подготовка признаков."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(ROOT, "данные", "Доходы домохозяйств_2023.xlsx")
OUT_DIR = os.path.abspath(os.path.join(HERE, ".."))
OUT_CSV = os.path.join(OUT_DIR, "датасет_ter30_per43.csv")
CACHE = os.environ.get("MR1_CACHE", "/tmp/claude-1000/-home-nedloz-dev-analisys/381794d9-0a55-49cf-bf19-d1d1960c6ff6/scratchpad/ter30.csv")

TER, PER = 30, 43
KEEP = ["bud", "rasq", "doxodn", "chlico", "chdet", "chrab", "mest", "kvzv"]


def _region():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    df = pd.read_excel(SRC)
    d = df[df.ter == TER].reset_index(drop=True)
    d.to_csv(CACHE, index=False)
    return d


def load_clean(save=True):
    """Возвращает очищенный датасет: 43-й квартал, без строк chrab > chlico."""
    d = _region()
    q = d[d.per == PER][KEEP].copy()
    q = q[q.chrab <= q.chlico].reset_index(drop=True)
    q["mest01"] = q.mest - 1  # 0 — mest=1, 1 — mest=2 (какой из них село — уточнить)
    q["ln_rasq"] = np.log(q.rasq)
    q["ln_doxodn"] = np.log(q.doxodn)
    if save:
        q.to_csv(OUT_CSV, index=False)
    return q
