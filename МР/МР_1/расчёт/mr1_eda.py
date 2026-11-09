"""Этап 2: первичный анализ — статистики, графики, корреляции, VIF."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import mr1_data as D

TAB = os.path.join(D.OUT_DIR, "таблицы")
FIG = os.path.join(D.OUT_DIR, "рисунки")
Y = "rasq"
XS = ["doxodn", "chlico", "chdet", "chrab", "mest01"]
LABELS = {"rasq": "Расход на потребление на человека, руб.", "doxodn": "Доход на человека, руб.",
          "chlico": "Членов домохозяйства", "chdet": "Детей до 16 лет",
          "chrab": "Работающих", "mest01": "Тип НП (1 = mest 2)"}
plt.rcParams["font.family"] = "DejaVu Sans"


def describe():
    cols = [Y] + XS
    d = D.load_clean(save=False)[cols]
    t = pd.DataFrame({"mean": d.mean(), "median": d.median(), "sd": d.std(),
                      "min": d.min(), "max": d.max(), "skew": d.skew()})
    return t


def corr():
    cols = [Y] + XS
    return D.load_clean(save=False)[cols].corr(method="pearson")


def vif():
    X = sm.add_constant(D.load_clean(save=False)[XS])
    v = {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"}
    return pd.Series(v)


def vif_log():
    d = D.load_clean(save=False)
    X = sm.add_constant(d[["ln_doxodn", "chlico", "chdet", "chrab", "mest01"]])
    v = {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"}
    return pd.Series(v)


def plots():
    d = D.load_clean(save=False)
    os.makedirs(FIG, exist_ok=True)
    cols = [Y] + XS
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for a, c in zip(ax.ravel(), cols):
        a.hist(d[c], bins=30 if c in (Y, "doxodn") else None, color="#4c78a8")
        a.set_title(LABELS[c], fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "01_гистограммы.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(d.ln_rasq, bins=30, color="#4c78a8"); ax[0].set_title("ln(расход на человека)")
    ax[1].hist(d.ln_doxodn, bins=30, color="#4c78a8"); ax[1].set_title("ln(доход на человека)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "02_гистограммы_логарифмов.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for a, c in zip(ax.ravel(), XS):
        a.scatter(d[c], d[Y], s=8, alpha=.5, color="#4c78a8")
        a.set_xlabel(LABELS[c], fontsize=8); a.set_ylabel("rasq", fontsize=8)
    ax.ravel()[-1].scatter(d.ln_doxodn, d.ln_rasq, s=8, alpha=.5, color="#e45756")
    ax.ravel()[-1].set_xlabel("ln(доход)"); ax.ravel()[-1].set_ylabel("ln(расход)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "03_диаграммы_рассеяния.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for a, c in zip(ax.ravel(), [Y, "doxodn", "chlico", "chdet", "chrab"]):
        a.boxplot(d[c], vert=True); a.set_title(LABELS[c], fontsize=9)
    ax.ravel()[-1].boxplot([d[d.mest01 == 0][Y], d[d.mest01 == 1][Y]], tick_labels=["mest=1", "mest=2"])
    ax.ravel()[-1].set_title("rasq по типу НП", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "04_боксплоты.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "05_тепловая_карта.png"), dpi=130); plt.close(fig)


def main():
    os.makedirs(TAB, exist_ok=True)
    describe().round(3).to_csv(os.path.join(TAB, "02_описательная_статистика.csv"))
    corr().round(4).to_csv(os.path.join(TAB, "02_корреляции.csv"))
    pd.DataFrame({"VIF": vif(), "VIF_log": vif_log().rename(index={"ln_doxodn": "doxodn"})}).round(3) \
        .to_csv(os.path.join(TAB, "02_VIF.csv"))
    plots()
    print(describe().round(3).to_string()); print(corr().round(3).to_string())
    print(vif().round(2).to_string()); print(vif_log().round(2).to_string())


if __name__ == "__main__":
    main()
