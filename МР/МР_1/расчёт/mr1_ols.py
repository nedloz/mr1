"""Этап 3: OLS и модели с логарифмами, сравнение в исходных единицах."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mr1_data as D

TAB = os.path.join(D.OUT_DIR, "таблицы")

FORMULAS = {
    "lin": "rasq ~ doxodn + chlico + chdet + chrab + mest01",
    "linlog": "rasq ~ ln_doxodn + chlico + chdet + chrab + mest01",
    "semilog": "ln_rasq ~ doxodn + chlico + chdet + chrab + mest01",
    "loglog": "ln_rasq ~ ln_doxodn + chlico + chdet + chrab + mest01",
    "loglog_lnN": "ln_rasq ~ ln_doxodn + ln_chlico + chdet + chrab + mest01",
}
LOGY = {"semilog", "loglog", "loglog_lnN"}


def data():
    d = D.load_clean(save=False)
    d["ln_chlico"] = np.log(d.chlico)
    return d


def fit_models(d=None):
    d = data() if d is None else d
    return {k: smf.ols(f, d).fit() for k, f in FORMULAS.items()}


def metrics(y, yhat, k):
    """Ошибки считаются вручную (правило проекта). k — число параметров с константой."""
    n = len(y)
    e = y - yhat
    ess = (e ** 2).sum()
    tss = ((y - y.mean()) ** 2).sum()
    return {
        "R2_orig": 1 - ess / tss,
        "R2adj_orig": 1 - (ess / (n - k)) / (tss / (n - 1)),
        "RMSE": np.sqrt(ess / n),
        "MAE": np.abs(e).mean(),
        "MAPE_%": 100 * np.abs(e / y).mean(),
    }


def compare(d=None, models=None):
    d = data() if d is None else d
    models = fit_models(d) if models is None else models
    rows = {}
    for name, m in models.items():
        yhat = np.exp(m.fittedvalues.values) if name in LOGY else m.fittedvalues.values
        r = metrics(d.rasq.values, yhat, int(m.df_model) + 1)
        r.update({"R2_model": m.rsquared, "R2adj_model": m.rsquared_adj,
                  "F": m.fvalue, "p_F": m.f_pvalue, "s2_resid": m.mse_resid,
                  "AIC": m.aic, "n_sig_x": int((m.pvalues.drop("Intercept") < 0.05).sum()),
                  "k_x": int(m.df_model)})
        rows[name] = r
    return pd.DataFrame(rows).T


def coef_table(m):
    ci = m.conf_int()
    return pd.DataFrame({"coef": m.params, "se": m.bse, "t": m.tvalues, "p": m.pvalues,
                         "ci_low": ci[0], "ci_high": ci[1]})


def main():
    os.makedirs(TAB, exist_ok=True)
    d = data()
    ms = fit_models(d)
    c = compare(d, ms)
    c.to_csv(os.path.join(TAB, "03_сравнение_моделей.csv"))
    for k, m in ms.items():
        coef_table(m).to_csv(os.path.join(TAB, f"03_коэффициенты_{k}.csv"))
    pd.set_option("display.width", 220)
    print(c.round(4).to_string())
    for k in ("lin", "loglog", "loglog_lnN"):
        print(k); print(coef_table(ms[k]).round(4).to_string())


if __name__ == "__main__":
    main()
