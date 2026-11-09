"""Проверки ключевых чисел МР 1. Запуск: .venv/bin/python -I -m pytest или напрямую (python test_mr1.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mr1_data as D


def test_load_clean_shape():
    df = D.load_clean()
    assert len(df) == 480 - 16
    assert df.bud.is_unique
    assert (df[["rasq", "doxodn"]] > 0).all().all()
    assert set(df.mest01) == {0, 1}
    assert (df.chrab <= df.chlico).all()
    assert np.allclose(df.ln_rasq, np.log(df.rasq))


def test_vif_chlico_chdet_high():
    import mr1_eda as E
    v = E.vif()
    assert "const" not in v.index
    assert v["chlico"] > 4 and v["chdet"] > 4


def test_corr_matches_manual():
    import mr1_eda as E
    df = D.load_clean(save=False)
    c = E.corr().loc["rasq", "doxodn"]
    x, y = df.doxodn.values, df.rasq.values
    manual = ((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std())
    assert abs(c - manual) < 1e-9


def test_r2_backtransform():
    import mr1_ols as O
    m = O.fit_models()
    cmp_ = O.compare()
    df = D.load_clean(save=False)
    y = df.rasq.values
    yhat = np.exp(m["loglog"].fittedvalues.values)
    manual = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert abs(cmp_.loc["loglog", "R2_orig"] - manual) < 1e-9
    assert cmp_.loc["loglog", "R2_orig"] < 1
    # для линейной модели R2 в исходных единицах совпадает со стандартным
    assert abs(cmp_.loc["lin", "R2_orig"] - m["lin"].rsquared) < 1e-9


def test_robust_same_params():
    import mr1_ols as O
    import mr1_hetero as H
    m = O.fit_models()["loglog"]
    r = H.robust(m)
    assert np.allclose(r.params, m.params)
    assert not np.allclose(r.bse, m.bse)
    t = H.tests_table(m)
    assert set(t.index) == {"Breusch-Pagan", "White", "Goldfeld-Quandt"}
    assert ((t["p"] >= 0) & (t["p"] <= 1)).all()


def test_studentized_external():
    import mr1_ols as O
    import mr1_resid as R
    m = O.fit_models()["loglog"]
    t = R.influence_table(m)
    inf = m.get_influence()
    assert np.allclose(t["stud_ext"], inf.resid_studentized_external)
    # ручная формула: e_i / (s_(-i) * sqrt(1 - h_ii))
    e, h, n, k = m.resid.values, inf.hat_matrix_diag, int(m.nobs), int(m.df_model) + 1
    s2 = (e ** 2).sum() / (n - k)
    s2_i = ((n - k) * s2 - e ** 2 / (1 - h)) / (n - k - 1)
    manual = e / np.sqrt(s2_i * (1 - h))
    assert np.allclose(t["stud_ext"], manual)


def test_cv_only_on_train():
    import mr1_reg as G
    r = G.run()
    tr, te = r["idx_train"], r["idx_test"]
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == 464
    df = D.load_clean(save=False)
    assert abs(r["scaler_mean"][0] - df.iloc[tr].ln_doxodn.mean()) < 1e-9
    assert abs(r["scaler_mean"][0] - df.ln_doxodn.mean()) > 1e-6  # не по всей выборке
    m = r["metrics"]
    assert {"OLS", "Ridge", "Lasso", "ElasticNet"} <= set(m.index)
    # все модели оценены на одном и том же test
    assert (m["n_test"] == len(te)).all()


def test_pca_cumvar():
    import mr1_pca as P
    r = P.run()
    cum = np.cumsum(r["eigvals"]) / r["eigvals"].sum()
    k = r["k"]
    assert cum[k - 1] >= 0.70 and (k == 1 or cum[k - 2] < 0.70)
    Z = r["scores"]
    assert np.allclose(np.corrcoef(Z.T) - np.eye(Z.shape[1]), 0, atol=1e-8)
    df = D.load_clean(save=False)
    corr = np.corrcoef(df[P.FEATURES].values.T)
    assert np.allclose(np.sort(np.linalg.eigvalsh(corr))[::-1], r["eigvals"])
    # PC-коэффициенты в пространстве X: beta_x = V_k gamma
    assert np.allclose(r["beta_std"], r["V"][:, :k] @ r["gamma"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
