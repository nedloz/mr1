"""Этап 6: Ridge / Lasso / Elastic Net с кросс-валидацией на train; сравнение с OLS на одном test."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mr1_data as D
import mr1_ols as O
import mr1_resid as R

TAB = os.path.join(D.OUT_DIR, "таблицы")
FEATURES = ["ln_doxodn", "chlico", "chdet", "chrab", "mest01"]
SEED = 42


def _metrics(y_ln, yhat_ln):
    """Ошибки по ln y и в исходных единицах (exp), вручную."""
    y, yhat = np.exp(y_ln), np.exp(yhat_ln)
    e_ln, e = y_ln - yhat_ln, y - yhat
    return {"n_test": len(y_ln),
            "R2_ln": 1 - (e_ln ** 2).sum() / ((y_ln - y_ln.mean()) ** 2).sum(),
            "RMSE_ln": np.sqrt((e_ln ** 2).mean()),
            "R2_orig": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "RMSE": np.sqrt((e ** 2).mean()), "MAE": np.abs(e).mean(),
            "MAPE_%": 100 * np.abs(e / y).mean()}


def run():
    d = D.load_clean(save=False)
    X, y = d[FEATURES].values, d.ln_rasq.values
    idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
    cv = KFold(5, shuffle=True, random_state=SEED)
    models = {
        "OLS": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 100), cv=cv),
        "Lasso": LassoCV(cv=cv, max_iter=100000, random_state=SEED),
        "ElasticNet": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1], cv=cv, max_iter=100000, random_state=SEED),
    }
    rows, coefs, params, fits = {}, {}, {}, {}
    for name, est in models.items():
        pipe = Pipeline([("sc", StandardScaler()), ("m", est)]).fit(X[tr], y[tr])
        sc, m = pipe.named_steps["sc"], pipe.named_steps["m"]
        rows[name] = _metrics(y[te], pipe.predict(X[te]))
        rows[name]["R2_ln_train"] = pipe.score(X[tr], y[tr])
        b_std = np.asarray(m.coef_).ravel()
        b = b_std / sc.scale_  # в исходном масштабе X
        coefs[name] = pd.Series(np.r_[m.intercept_ - (b * sc.mean_).sum(), b], index=["const"] + FEATURES)
        coefs[name + "_std"] = pd.Series(b_std, index=FEATURES)
        rows[name]["n_nonzero"] = int((np.abs(b_std) > 1e-12).sum())
        p = {}
        if name == "Ridge": p["alpha"] = float(m.alpha_)
        if name == "Lasso": p["alpha"] = float(m.alpha_)
        if name == "ElasticNet": p["alpha"] = float(m.alpha_); p["l1_ratio"] = float(m.l1_ratio_)
        params[name] = p
        fits[name] = pipe
        fixed = {"OLS": LinearRegression(), "Ridge": Ridge(alpha=p.get("alpha", 1)),
                 "Lasso": Lasso(alpha=p.get("alpha", 1), max_iter=100000),
                 "ElasticNet": ElasticNet(alpha=p.get("alpha", 1), l1_ratio=p.get("l1_ratio", .5), max_iter=100000)}[name]
        cvs = cross_val_score(Pipeline([("sc", StandardScaler()), ("m", fixed)]), X[tr], y[tr], cv=cv,
                              scoring="neg_mean_squared_error")
        rows[name]["CV_MSE"] = float(-cvs.mean())  # честная CV-ошибка на train с выбранным гиперпараметром
    return {"metrics": pd.DataFrame(rows).T, "coefs": pd.DataFrame(coefs), "params": params,
            "idx_train": tr, "idx_test": te, "scaler_mean": StandardScaler().fit(X[tr]).mean_,
            "fits": fits, "X": X, "y": y}


def residuals_selected(r, name):
    """Остатки выбранной модели на train; рычаги — OLS-рычаги по тем же X (приближение)."""
    tr = r["idx_train"]
    Xtr, ytr = r["X"][tr], r["y"][tr]
    pipe = r["fits"][name]
    fitted = pipe.predict(Xtr)
    Xd = np.column_stack([np.ones(len(tr)), StandardScaler().fit_transform(Xtr)])
    h = np.diag(Xd @ np.linalg.inv(Xd.T @ Xd) @ Xd.T)
    nz = int((np.abs(r["coefs"][name + "_std"]) > 1e-12).sum())
    return ytr - fitted, fitted, h, nz + 1


def select_regularized(m):
    """Минимум CV_MSE на train; в пределах 1e-4 от минимума — модель с меньшим числом ненулевых, затем Lasso < EN < Ridge."""
    c = m.loc[["Lasso", "ElasticNet", "Ridge"]]
    ok = c[c.CV_MSE <= c.CV_MSE.min() + 1e-4]
    return ok.sort_values("n_nonzero", kind="stable").index[0]


def main():
    os.makedirs(TAB, exist_ok=True)
    r = run()
    pd.set_option("display.width", 220)
    r["metrics"].round(4).to_csv(os.path.join(TAB, "06_регуляризация_метрики.csv"))
    r["coefs"].round(5).to_csv(os.path.join(TAB, "06_регуляризация_коэффициенты.csv"))
    print(r["metrics"].round(4).to_string()); print(r["coefs"].round(4).to_string()); print(r["params"])
    best = select_regularized(r["metrics"])
    print("selected regularized (CV_MSE within 1e-4 of min, fewest non-zero, Lasso first):", best)
    e, f, h, k = residuals_selected(r, best)
    print(R.resid_plots(e, f, h, k, f"reg_{best}", f"{best}: остатки на train"))


if __name__ == "__main__":
    main()
