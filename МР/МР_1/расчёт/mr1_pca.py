"""Этап 7: метод главных компонент и регрессия на главных компонентах."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import mr1_data as D

TAB = os.path.join(D.OUT_DIR, "таблицы")
FIG = os.path.join(D.OUT_DIR, "рисунки")
FEATURES = ["ln_doxodn", "chlico", "chdet", "chrab", "mest01"]
THRESH = 0.70


def run():
    d = D.load_clean(save=False)
    X = d[FEATURES].values
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    Z = (X - mu) / sd
    corr = np.corrcoef(X.T)
    w, V = np.linalg.eigh(corr)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    for j in range(V.shape[1]):  # знак: наибольшая по модулю нагрузка положительна
        if V[np.abs(V[:, j]).argmax(), j] < 0:
            V[:, j] *= -1
    scores = Z @ V
    k = int(np.searchsorted(np.cumsum(w) / w.sum(), THRESH) + 1)
    y = d.ln_rasq.values
    res = sm.OLS(y, sm.add_constant(scores[:, :k])).fit()
    gamma = res.params[1:]
    beta_std = V[:, :k] @ gamma           # эффект на стандартизованные X
    beta_orig = beta_std / sd             # эффект в исходных единицах X
    return {"eigvals": w, "V": V, "k": k, "scores": scores, "res": res, "gamma": gamma,
            "beta_std": beta_std, "beta_orig": beta_orig, "sd": sd, "y": y}


def main():
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    r = run()
    w, V, k, res = r["eigvals"], r["V"], r["k"], r["res"]
    names = [f"PC{i + 1}" for i in range(len(w))]
    expl = pd.DataFrame({"eigenvalue": w, "share": w / w.sum(), "cum_share": np.cumsum(w) / w.sum()}, index=names)
    load = pd.DataFrame(V, index=FEATURES, columns=names)
    contrib = (V ** 2) * 100  # вклад переменной в компоненту, %
    contrib = pd.DataFrame(contrib, index=FEATURES, columns=names)
    expl.round(4).to_csv(os.path.join(TAB, "07_pca_доли.csv"))
    load.round(4).to_csv(os.path.join(TAB, "07_pca_нагрузки.csv"))
    contrib.round(2).to_csv(os.path.join(TAB, "07_pca_вклады_%.csv"))
    ci = res.conf_int()
    reg = pd.DataFrame({"coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues,
                        "ci_low": ci[:, 0], "ci_high": ci[:, 1]}, index=["const"] + names[:k])
    reg.round(5).to_csv(os.path.join(TAB, "07_pca_регрессия.csv"))
    back = pd.DataFrame({"beta_std": r["beta_std"], "beta_orig": r["beta_orig"]}, index=FEATURES)
    back.round(5).to_csv(os.path.join(TAB, "07_pca_коэффициенты_в_X.csv"))
    rk = []
    for j in range(1, len(w) + 1):
        m = sm.OLS(r["y"], sm.add_constant(r["scores"][:, :j])).fit()
        rk.append({"k": j, "cum_share_X": np.cumsum(w)[j - 1] / w.sum(), "R2": m.rsquared, "R2adj": m.rsquared_adj,
                   "AIC": m.aic, "sig_all": bool((m.pvalues[1:] < 0.05).all())})
    rk = pd.DataFrame(rk).set_index("k"); rk.round(4).to_csv(os.path.join(TAB, "07_pca_R2_по_числу_ГК.csv"))
    print(rk.round(4).to_string())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, w / w.sum() * 100, color="#4c78a8"); ax.plot(names, np.cumsum(w) / w.sum() * 100, "o-", color="#e45756")
    ax.axhline(70, color="k", ls="--", lw=.8); ax.set_ylabel("% дисперсии"); ax.set_title("Каменистая осыпь")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "08_pca_осыпь.png"), dpi=130); plt.close(fig)
    pd.set_option("display.width", 200)
    print(expl.round(4).to_string()); print(load.round(3).to_string()); print(contrib.round(1).to_string())
    print("k =", k, "R2 =", round(res.rsquared, 4), "R2adj =", round(res.rsquared_adj, 4), "F =", round(res.fvalue, 2), "pF =", res.f_pvalue)
    print(reg.round(4).to_string()); print(back.round(5).to_string())


if __name__ == "__main__":
    main()
