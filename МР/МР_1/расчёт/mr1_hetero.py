"""Этап 4: гетероскедастичность — графики, тесты BP/White/GQ, робастные ошибки HC3."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.diagnostic import het_breuschpagan, het_goldfeldquandt, het_white

import mr1_data as D
import mr1_ols as O

TAB = os.path.join(D.OUT_DIR, "таблицы")
FIG = os.path.join(D.OUT_DIR, "рисунки")
ALPHA = 0.05


def tests_table(res):
    """BP, White, Голдфелд–Квандт (упорядочение по первому фактору — доходу, drop=0.2)."""
    y, X = res.model.endog, res.model.exog
    bp = het_breuschpagan(res.resid, X)
    wh = het_white(res.resid, X)
    sort_col = 1  # первый фактор после константы — доход (doxodn / ln_doxodn)
    gq = het_goldfeldquandt(y, X, idx=sort_col, drop=0.2, alternative="two-sided")
    t = pd.DataFrame({"stat": [bp[0], wh[0], gq[0]], "p": [bp[1], wh[1], gq[1]]},
                     index=["Breusch-Pagan", "White", "Goldfeld-Quandt"])
    t["вывод"] = ["гетероскедастичность" if p < ALPHA else "гомоскедастичность" for p in t.p]
    return t


def robust(res):
    return res.get_robustcov_results(cov_type="HC3")


def robust_table(res):
    r = robust(res)
    names = res.model.exog_names
    return pd.DataFrame({"coef": res.params.values, "se_OLS": res.bse.values, "se_HC3": r.bse,
                         "t_OLS": res.tvalues.values, "t_HC3": r.tvalues,
                         "p_OLS": res.pvalues.values, "p_HC3": r.pvalues,
                         "ci_low_HC3": r.conf_int()[:, 0], "ci_high_HC3": r.conf_int()[:, 1]}, index=names)


def plots(res, tag):
    os.makedirs(FIG, exist_ok=True)
    d = O.data()
    names = [n for n in res.model.exog_names if n != "Intercept"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    ax = ax.ravel()
    ax[0].scatter(res.fittedvalues, res.resid, s=8, alpha=.5)
    ax[0].axhline(0, color="k", lw=.8); ax[0].set_title("остатки против предсказанных", fontsize=9)
    for a, n in zip(ax[1:], names):
        a.scatter(d[n], res.resid, s=8, alpha=.5)
        a.axhline(0, color="k", lw=.8); a.set_title(f"остатки против {n}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"06_остатки_{tag}.png"), dpi=130); plt.close(fig)


def main():
    os.makedirs(TAB, exist_ok=True)
    ms = O.fit_models()
    pd.set_option("display.width", 200)
    for k in ("lin", "loglog"):
        t = tests_table(ms[k]); t.round(4).to_csv(os.path.join(TAB, f"04_тесты_гетероскедастичности_{k}.csv"))
        rt = robust_table(ms[k]); rt.round(4).to_csv(os.path.join(TAB, f"04_робастные_{k}.csv"))
        plots(ms[k], k)
        print(k); print(t.round(4).to_string()); print(rt.round(4).to_string())


if __name__ == "__main__":
    main()
