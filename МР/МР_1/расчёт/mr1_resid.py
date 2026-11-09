"""Этап 5: анализ остатков — QQ, PP, остатки против предсказанных, стандартизированные и
стьюдентизированные остатки, рычаги, расстояние Кука; сравнение «до/после» исключения выбросов."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

import mr1_data as D
import mr1_ols as O

TAB = os.path.join(D.OUT_DIR, "таблицы")
FIG = os.path.join(D.OUT_DIR, "рисунки")


def influence_table(res):
    inf = res.get_influence()
    return pd.DataFrame({"resid": res.resid.values, "fitted": res.fittedvalues.values,
                         "stud_int": inf.resid_studentized_internal,
                         "stud_ext": inf.resid_studentized_external,
                         "h": inf.hat_matrix_diag, "cook": inf.cooks_distance[0]})


def thresholds(res):
    n, k = int(res.nobs), int(res.df_model) + 1
    return {"stud": 2.0, "stud_strict": 3.0, "h": 2 * k / n, "cook": 4 / n}


def counts(res):
    t, th = influence_table(res), thresholds(res)
    return {"|stud_ext|>2": int((t.stud_ext.abs() > th["stud"]).sum()),
            "|stud_ext|>3": int((t.stud_ext.abs() > th["stud_strict"]).sum()),
            f"h>{th['h']:.4f}": int((t.h > th["h"]).sum()),
            f"Cook>{th['cook']:.4f}": int((t.cook > th["cook"]).sum())}


def resid_plots(resid, fitted, h, k, tag, title=""):
    """Универсально: подходит и для OLS, и для регуляризованных моделей (h — рычаги по X)."""
    os.makedirs(FIG, exist_ok=True)
    resid, fitted, h = map(np.asarray, (resid, fitted, h))
    n = len(resid)
    s2 = (resid ** 2).sum() / (n - k)
    stand = resid / np.sqrt(s2)
    s2_i = ((n - k) * s2 - resid ** 2 / (1 - h)) / (n - k - 1)
    stud = resid / np.sqrt(s2_i * (1 - h))
    z = (resid - resid.mean()) / resid.std(ddof=1)
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax = ax.ravel()
    sm.qqplot(z, line="45", ax=ax[0], markersize=3); ax[0].set_title("QQ-график остатков", fontsize=9)
    sm.ProbPlot(z).ppplot(line="45", ax=ax[1], markersize=3); ax[1].set_title("PP-график остатков", fontsize=9)
    ax[2].scatter(fitted, resid, s=8, alpha=.5); ax[2].axhline(0, color="k", lw=.8)
    ax[2].set_title("остатки против предсказанных", fontsize=9)
    idx = np.arange(1, n + 1)
    ax[3].scatter(idx, stand, s=8, alpha=.6)
    for v in (-2, 2): ax[3].axhline(v, color="r", ls="--", lw=.8)
    ax[3].set_title("стандартизированные остатки", fontsize=9)
    ax[4].scatter(idx, stud, s=8, alpha=.6)
    for v in (-3, 3): ax[4].axhline(v, color="r", ls="--", lw=.8)
    for v in (-2, 2): ax[4].axhline(v, color="orange", ls=":", lw=.8)
    ax[4].set_title("стьюдентизированные остатки", fontsize=9)
    ax[5].scatter(h, stud, s=8, alpha=.6)
    ax[5].axvline(2 * k / n, color="r", ls="--", lw=.8)
    ax[5].set_xlabel("рычаг"); ax[5].set_ylabel("стьюдентизир. остаток")
    ax[5].set_title("влияние: рычаг и остаток", fontsize=9)
    if title: fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"07_остатки_{tag}.png"), dpi=130); plt.close(fig)
    return {"shapiro_p": float(st.shapiro(resid).pvalue), "jb_p": float(st.jarque_bera(resid).pvalue),
            "skew": float(st.skew(resid)), "kurt": float(st.kurtosis(resid))}


def before_after(res, d=None, formula=None, limit=3.0):
    """Модель на полных данных и без наблюдений с |stud_ext| > limit."""
    d = O.data() if d is None else d
    f = formula or O.FORMULAS["loglog"]
    t = influence_table(res)
    keep = (t.stud_ext.abs() <= limit).values
    r2 = smf.ols(f, d[keep]).fit(cov_type="HC3")
    r1 = smf.ols(f, d).fit(cov_type="HC3")
    out = pd.DataFrame({"coef_all": r1.params, "p_all": r1.pvalues, "coef_clean": r2.params, "p_clean": r2.pvalues})
    return out, int((~keep).sum()), r1.rsquared, r2.rsquared


def main():
    os.makedirs(TAB, exist_ok=True)
    m = O.fit_models()["loglog"]
    t = influence_table(m)
    t.round(5).to_csv(os.path.join(TAB, "05_влияние_loglog.csv"), index=False)
    norm = resid_plots(m.resid, m.fittedvalues, t.h, int(m.df_model) + 1, "loglog", "OLS loglog")
    ba, n_out, r2a, r2c = before_after(m)
    ba.round(5).to_csv(os.path.join(TAB, "05_до_после_выбросов.csv"))
    pd.set_option("display.width", 200)
    print(counts(m)); print(norm); print(ba.round(4).to_string()); print("excluded", n_out, "R2", round(r2a, 4), round(r2c, 4))


if __name__ == "__main__":
    main()
