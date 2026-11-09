"""ДЗ 2: регион ter=4, y=PROD, x1=DOXODN, x2=CHDET, 150 наблюдений.
Запуск: .venv/bin/python "ДЗ/ДЗ_2/расчёт_дз2.py"
Регрессия считается вручную (numpy, МНК через нормальные уравнения);
statsmodels используется только для сверки."""
import os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

D = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(D, "..", "..", "данные", "Доходы домохозяйств_2023.xlsx")
CACHE = os.path.join(D, "выборка_ter04_150.csv")
ALPHA, SEED, N = 0.05, 2026, 150

if os.path.exists(CACHE):
    s = pd.read_csv(CACHE)
else:
    df = pd.read_excel(SRC)
    r = df[df.ter == 4][["ter", "doxodn", "chdet", "prod"]].reset_index(drop=True)
    print("домохозяйств в регионе:", len(r))
    s = r.sample(N, random_state=SEED); s.to_csv(CACHE, index=False)
s = s.reset_index(drop=True)
s.loc[s["prod"] == 0, "prod"] = 1  # нули y -> 1 (в выборке нулей нет)

def ols(X, y):
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    yh = X @ b; e = y - yh
    ess = e @ e; tss = ((y - y.mean())**2).sum()
    r2 = 1 - ess/tss; r2a = 1 - (ess/(n-k))/(tss/(n-1))
    F = ((tss-ess)/(k-1))/(ess/(n-k)); pF = stats.f.sf(F, k-1, n-k)
    se = np.sqrt(np.diag(XtXi)*ess/(n-k)); t = b/se; pt = 2*stats.t.sf(abs(t), n-k)
    tc = stats.t.ppf(1-ALPHA/2, n-k)
    m = dict(n=n, k=k, b=b, se=se, t=t, pt=pt, tcrit=tc, F=F, pF=pF,
             Fcrit=stats.f.ppf(1-ALPHA, k-1, n-k), r2=r2, r2a=r2a, ess=ess, tss=tss)
    return m, yh

def fit(d):
    y = d["prod"].values.astype(float); x1 = d["doxodn"].values.astype(float); x2 = d["chdet"].values.astype(float)
    X = np.column_stack([np.ones(len(y)), x1, x2])
    m, yh = ols(X, y)
    sm_ = sm.OLS(y, X).fit()  # сверка
    assert np.allclose(m["b"], sm_.params) and np.isclose(m["r2"], sm_.rsquared) and np.isclose(m["F"], sm_.fvalue)
    return m, y, yh

def iqr_mask(d, cols):
    ok = pd.Series(True, index=d.index); lim = {}
    for c in cols:
        q1, q3 = d[c].quantile([.25, .75]); i = q3-q1
        lo, hi = q1-1.5*i, q3+1.5*i; lim[c] = (q1, q3, lo, hi)
        ok &= d[c].between(lo, hi)
    return ok, lim

cols = ["doxodn", "chdet", "prod"]
ok, lim = iqr_mask(s, cols)
clean = s[ok].reset_index(drop=True)

out = []
P = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))
P("Описательная статистика (150 набл.):"); P(s[cols].describe().round(2).to_string())
P("\nГраницы IQR (Q1, Q3, нижняя, верхняя):")
for c, v in lim.items():
    P(f"  {c}: " + ", ".join(f"{x:.2f}" for x in v) + f";  выбросов: {int((~s[c].between(v[2], v[3])).sum())}")
P(f"Удалено строк: {int((~ok).sum())}; осталось {len(clean)}")
P("chdet в выборке:", s.chdet.value_counts().sort_index().to_dict())

res = {}
for name, d in [("с выбросами", s), ("без выбросов", clean)]:
    m, y, yh = fit(d); res[name] = m
    P(f"\n=== {name}, n={m['n']} ===")
    for i, nm in enumerate(["b0 (const)", "b1 (doxodn)", "b2 (chdet)"]):
        P(f"{nm}: {m['b'][i]:.4f}  se={m['se'][i]:.4f}  t={m['t'][i]:.3f}  p={m['pt'][i]:.4g}  {'значим' if m['pt'][i] < ALPHA else 'НЕзначим'}")
    P(f"t_крит={m['tcrit']:.3f}; F={m['F']:.3f}, F_крит={m['Fcrit']:.3f}, p={m['pF']:.4g}")
    P(f"R2={m['r2']:.4f}, R2adj={m['r2a']:.4f}")

# рисунки
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#3b6ea5", "#d9822b", "#888888"
names = {"doxodn": "Доход, руб.", "chdet": "Число детей", "prod": "Расходы на продукты, руб."}

# 1. ящики с усами до и после очистки
fig, ax = plt.subplots(2, 3, figsize=(11, 6))
for j, c in enumerate(cols):
    for i, (d, t) in enumerate([(s, "до очистки"), (clean, "после очистки")]):
        ax[i, j].boxplot(d[c], vert=False, patch_artist=True, boxprops=dict(facecolor=C1, alpha=.5),
                         flierprops=dict(marker="o", markerfacecolor=C2, markeredgecolor=C2))
        ax[i, j].set_title(f"{names[c]} — {t}", fontsize=9); ax[i, j].set_yticks([])
        if c != "chdet":
            ax[i, j].xaxis.set_major_locator(plt.MaxNLocator(5))
            ax[i, j].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f} тыс."))
        else:
            ax[i, j].xaxis.set_major_locator(plt.MultipleLocator(1))
plt.tight_layout(); plt.savefig(os.path.join(D, "boxplots.png"), dpi=130); plt.close()

# 2. диаграмма рассеяния и линии регрессии
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
xs = np.linspace(s.doxodn.min(), s.doxodn.max(), 100)
for a_, d, nm, t in [(ax[0], s, "с выбросами", "до очистки"), (ax[1], clean, "без выбросов", "после очистки")]:
    out_ = s[~ok]
    a_.scatter(d.doxodn, d["prod"], c=C1, alpha=.6, s=22, label="наблюдения")
    if nm == "с выбросами": a_.scatter(out_.doxodn, out_["prod"], c=C2, s=26, label="будут удалены как выбросы")
    b = res[nm]["b"]; a_.plot(xs, b[0]+b[1]*xs+b[2]*d.chdet.mean(), c="k", lw=2, label=f"регрессия ($R^2$={res[nm]['r2']:.2f})")
    a_.set_title(f"{t}, n={len(d)}"); a_.set_xlabel(names["doxodn"]); a_.legend(fontsize=8, frameon=False)
ax[0].set_ylabel(names["prod"]); plt.tight_layout(); plt.savefig(os.path.join(D, "scatter_regressia.png"), dpi=130); plt.close()

# 3. расходы по числу детей
fig, a_ = plt.subplots(figsize=(6, 4))
g = sorted(s.chdet.unique()); a_.boxplot([s[s.chdet == v]["prod"] for v in g], tick_labels=[f"{int(v)}\n(n={int((s.chdet==v).sum())})" for v in g],
                                         patch_artist=True, boxprops=dict(facecolor=C1, alpha=.5), medianprops=dict(color="k"))
a_.set_xlabel("Число детей до 16 лет"); a_.set_ylabel(names["prod"]); a_.set_title("Расходы на продукты и число детей")
plt.tight_layout(); plt.savefig(os.path.join(D, "prod_po_detyam.png"), dpi=130); plt.close()

# 4. сравнение R2 и коэффициентов
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
lab = ["с выбросами", "без выбросов"]
for a_, val, t in [(ax[0], [res[l]["r2"] for l in lab], "$R^2$"), (ax[1], [res[l]["b"][1] for l in lab], "$b_1$: руб. расходов на 1 руб. дохода"),
                   (ax[2], [res[l]["b"][2] for l in lab], "$b_2$: руб. на 1 ребёнка")]:
    bars = a_.bar(lab, val, color=[C3, C1]); a_.set_title(t, fontsize=10)
    for r_, v in zip(bars, val): a_.text(r_.get_x()+r_.get_width()/2, v, f"{v:.3g}" if v < 1 else f"{v:,.0f}".replace(",", " "), ha="center", va="bottom")
plt.tight_layout(); plt.savefig(os.path.join(D, "sravnenie_modeley.png"), dpi=130); plt.close()
open(os.path.join(D, "результаты_расчёта.txt"), "w", encoding="utf-8").write("\n".join(out))
