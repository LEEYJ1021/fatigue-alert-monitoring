import numpy as np, pandas as pd, time, os, json
from scipy.stats import norm
from scipy.special import expit

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = str(ROOT / 'results')          # all CSV/JSON outputs
os.makedirs(OUT, exist_ok=True)
GAMMA = 0.02

# ---------------------------------------------------------------- helpers
def starts_from(cl):
    return np.flatnonzero(np.r_[True, cl[1:] != cl[:-1]])

def ols_cl(X, y, st):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    S = np.add.reduceat(X * e[:, None], st, axis=0)
    G = len(st)
    V = XtXi @ (S.T @ S) @ XtXi * G / (G - 1)
    return b, np.sqrt(np.diag(V))

def tsls_cl(y, X, Z, st):
    Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)          # fitted X
    A = np.linalg.inv(Pz.T @ X)
    b = A @ (Pz.T @ y)
    e = y - X @ b
    S = np.add.reduceat(Pz * e[:, None], st, axis=0)
    G = len(st)
    V = A @ (S.T @ S) @ A * G / (G - 1)
    return b, np.sqrt(np.diag(V))

def within(v, st, Tc):
    m = np.add.reduceat(v, st, axis=0) / Tc
    return v - np.repeat(m, Tc, axis=0)

# ---------------------------------------------------------------- DGP (Eqs. 1-5 of the manuscript, stylised)
def gen_panel(rng, N, p0=0.7, kappa=0.0, delta=0.0, phi_u=0.0, kappa_m=0.0,
              decay=0.0, b0=-0.36, b1=0.5):
    Tm = 7
    Tc = rng.choice([4, 5, 6, 7], size=N, p=[0.4, 0.3, 0.2, 0.1])
    m = rng.normal(1.2, 0.6, N)                              # latent persistent complexity
    logx = m[:, None] + 0.3 * rng.standard_normal((N, Tm))   # mention-level complexity
    u = rng.normal(0, 1, N)                                  # latent alert propensity (documentation quality)
    A = np.zeros((N, Tm)); C = np.zeros((N, Tm)); D = np.zeros((N, Tm)); AD = np.zeros((N, Tm))
    cumC = np.zeros(N); cumD = np.zeros(N); cumdec = np.zeros(N)
    for t in range(Tm):
        if t >= 1:
            lag = logx[:, max(t - 2, 0)]
            a = (rng.random(N) < expit(b0 - b1 * (lag - 1.2) + u)).astype(float)   # Eq. 1 (2nd+ mentions only)
        else:
            a = np.zeros(N)
        cumC += a                                            # Eq. 3
        pdec = p0 * np.exp(-kappa * np.maximum(cumC - 1, 0)) # alert-fatigue: repeated alerts lower response
        dec = rng.random(N) < pdec
        cumD += np.exp(-decay * (t + 1)) * a * (~dec)        # Eq. 4 (decay=0 -> manuscript form)
        A[:, t] = a; C[:, t] = cumC; D[:, t] = cumD
        AD[:, t] = a * dec
    tt = np.arange(1, Tm + 1)[None, :]
    treated = (C[:, 2] == 2)                                 # high exposure at close of period 3
    eta = rng.normal(0, 0.02, N)
    Y = (0.85 - 0.01 * (logx - 1.2) + eta[:, None] + phi_u * u[:, None]
         + kappa_m * (m - 1.2)[:, None] - GAMMA * D                     # Eq. 5
         + delta * (tt - 1) * treated[:, None]
         + rng.normal(0, 0.02, (N, Tm)))
    mask = tt <= Tc[:, None]
    z = np.full((N, Tm), np.nan)
    for t in range(2, Tm):
        z[:, t] = logx[:, t - 2] - 1.2
    fl = lambda M: M[mask]
    P = dict(t=fl(np.broadcast_to(tt, (N, Tm))).astype(float),
             tr=fl(np.broadcast_to(treated[:, None], (N, Tm))).astype(float),
             C=fl(C), D=fl(D), Y=fl(Y), lx=fl(logx) - 1.2, z=fl(z),
             cl=np.repeat(np.arange(N), Tc), Tc=Tc, treated=treated,
             alert_rate=fl(A)[fl(np.broadcast_to(tt, (N, Tm))) >= 2].mean(),
             eff_p=(AD.sum() / max(A.sum(), 1)))
    P['st'] = starts_from(P['cl'])
    return P

# ---------------------------------------------------------------- estimators
def run_estimators(P, p0):
    out = {}
    st, t, tr, C, D, Y, lx = P['st'], P['t'], P['tr'], P['C'], P['D'], P['Y'], P['lx']
    # ---- DiD (Eq. 6)
    post = (t >= 4).astype(float)
    X = np.c_[np.ones_like(Y), tr, post, tr * post, lx]
    b, se = ols_cl(X, Y, st)
    def gm(mask): return C[mask].mean()
    dC = (gm((tr == 1) & (post == 1)) - gm((tr == 1) & (post == 0))) - \
         (gm((tr == 0) & (post == 1)) - gm((tr == 0) & (post == 0)))
    target = -GAMMA * (1 - p0) * dC
    out['DiD'] = (b[3], se[3], target)
    out['DiD implied p-hat'] = (1 + b[3] / (GAMMA * dC), se[3] / (GAMMA * abs(dC)), p0)
    # ---- Mediation, pooled
    ones = np.ones_like(Y)
    ba, sea = ols_cl(np.c_[ones, C, lx], D, st)
    bb, seb = ols_cl(np.c_[ones, D, C, lx], Y, st)
    a, sa = ba[1], sea[1]; bcoef, sb = bb[1], seb[1]; cprime, scp = bb[2], seb[2]
    ind = a * bcoef; sind = np.sqrt(a**2 * sb**2 + bcoef**2 * sa**2)
    out['Path a (pooled)'] = (a, sa, 1 - p0)
    out['Indirect a*b (pooled)'] = (ind, sind, -GAMMA * (1 - p0))
    out['Direct c\' (pooled)'] = (cprime, scp, 0.0)
    # ---- Mediation, CUI fixed effects
    Tc = P['Tc']
    Dw, Cw, Yw, Lw = [within(v, st, Tc) for v in (D, C, Y, lx)]
    ba, sea = ols_cl(np.c_[Cw, Lw], Dw, st)
    bb, seb = ols_cl(np.c_[Dw, Cw, Lw], Yw, st)
    ind = ba[0] * bb[0]; sind = np.sqrt(ba[0]**2 * seb[0]**2 + bb[0]**2 * sea[0]**2)
    out['Indirect a*b (FE)'] = (ind, sind, -GAMMA * (1 - p0))
    out['Direct c\' (FE)'] = (bb[1], seb[1], 0.0)
    # ---- IV (lag-2 complexity instrument), rows with t>=3
    k = t >= 3
    st2 = starts_from(P['cl'][k])
    I = (C / t)[k]; y = Y[k]; l2 = lx[k]; z = P['z'][k]; o = np.ones_like(y)
    Xn = np.c_[o, I, l2]; Zn = np.c_[o, z, l2]
    b, se = tsls_cl(y, Xn, Zn, st2)
    tgt = np.mean(-GAMMA * (1 - p0) * t[k])
    out['IV 2SLS'] = (b[1], se[1], tgt)
    b, se = ols_cl(Xn, y, st2)
    out['IV naive OLS'] = (b[1], se[1], tgt)
    return out

SCEN = {
    'S0 baseline (p=.70)':            dict(),
    'S1 true p=.50':                  dict(p0=0.5),
    'S2 true p=.90':                  dict(p0=0.9),
    'S3 pre-trend d=-.002':           dict(delta=-0.002),
    'S4 pre-trend d=-.005':           dict(delta=-0.005),
    'S5 fatigue k=.15':               dict(kappa=0.15),
    'S6 fatigue k=.30':               dict(kappa=0.30),
    'S7 unobs. confounder':           dict(phi_u=-0.03),
    'S8 IV exclusion violated':       dict(kappa_m=-0.02),
    'S9 drift decay .1 (hypothesis)': dict(decay=0.1),
}

def monte_carlo(R=int(os.environ.get('MC_R', 400)), N=2500, seed=2026):
    rows = []; meta = []
    for si, (name, kw) in enumerate(SCEN.items()):
        rng = np.random.default_rng(seed + si)
        nominal_p = kw.get('p0', 0.7)
        for r in range(R):
            P = gen_panel(rng, N, **kw)
            res = run_estimators(P, nominal_p)
            for est, (b, se, tg) in res.items():
                rows.append((name, est, b, se, tg))
            if r == 0 or r == R - 1:
                meta.append((name, P['alert_rate'], P['treated'].mean(), P['eff_p']))
        print(name, 'done', flush=True)
    df = pd.DataFrame(rows, columns=['scenario', 'estimator', 'est', 'se', 'target'])
    return df, pd.DataFrame(meta, columns=['scenario', 'alert_rate', 'share_treated', 'effective_p']).drop_duplicates('scenario')

def summarise(df):
    g = df.assign(err=df.est - df.target,
                  cover=(np.abs(df.est - df.target) <= 1.96 * df.se).astype(float),
                  reject0=(np.abs(df.est) > 1.96 * df.se).astype(float))
    s = g.groupby(['scenario', 'estimator'], sort=False).agg(
        mean_est=('est', 'mean'), target=('target', 'mean'), bias=('err', 'mean'),
        rmse=('err', lambda e: np.sqrt(np.mean(e**2))), mc_sd=('est', 'std'),
        mean_se=('se', 'mean'), coverage=('cover', 'mean'), reject_zero=('reject0', 'mean')).reset_index()
    s['rel_bias_pct'] = 100 * s.bias / s.target.abs().replace(0, np.nan)
    return s

# ---------------------------------------------------------------- Part 2: threshold policy with alert fatigue
N_TOT = 128945; CFP, CFN = 1.0, 5.0
NPOS = 39205 / CFN; NNEG = N_TOT - NPOS
z1, z2 = norm.ppf(1 - 0.078), norm.ppf(1 - 0.011)
S0 = (0.70 - 0.105) / (z2 - z1); MU0 = 0.105 - z1 * S0
q1, q2 = norm.ppf(0.954), norm.ppf(0.343)
S1 = (0.70 - 0.105) / (q1 - q2); MU1 = 0.105 + q1 * S1
AUC = norm.cdf((MU1 - MU0) / np.hypot(S0, S1))
FPR = lambda tau: 1 - norm.cdf((tau - MU0) / S0)
TPR = lambda tau: norm.cdf((MU1 - tau) / S1)
VOL = lambda tau: NNEG * FPR(tau) + NPOS * TPR(tau)
VREF = VOL(0.105)
LAM_CAL = -np.log(0.70)          # lambda at which p(V_ref)=0.70 (manuscript's response probability at its optimal volume)

def p_of_V(V, lam):
    """response probability falls exponentially with alert volume: p = exp(-lam * V / V_ref); lam=0 -> p=1 (Eq.7 literal)"""
    return np.exp(-lam * V / VREF)

def cost(tau, lam):
    fp = NNEG * FPR(tau); tp = NPOS * TPR(tau)
    p = 0.70 if lam == 'const' else p_of_V(VOL(tau), lam)
    return CFP * fp + CFN * (NPOS - p * tp)

GRID = np.arange(0, 1.0005, 0.0005)
MODELS = [('Eq.7 literal (p=1, lam=0)', 0.0), ('constant p=.70', 'const'),
          ('fatigue lam=0.357 (p=.70 at V_ref)', LAM_CAL), ('fatigue lam=0.70', 0.70),
          ('fatigue lam=1.00', 1.0), ('fatigue lam=1.50', 1.5), ('fatigue lam=2.00', 2.0)]

def threshold_table():
    base = CFN * NPOS
    rows = []
    for name, lam in MODELS:
        c = cost(GRID, lam); i = c.argmin(); tau = GRID[i]
        pv = 0.70 if lam == 'const' else float(p_of_V(VOL(tau), lam))
        rows.append(dict(model=name, lam=lam, tau_opt=tau, cost_opt=c[i], saving_pct=100 * (1 - c[i] / base),
                         alerts_per_quarter=VOL(tau), resp_prob_at_opt=pv,
                         cost_at_0p70=float(cost(0.70, lam)), change_at_0p70_pct=100 * (float(cost(0.70, lam)) / base - 1)))
    df = pd.DataFrame(rows)
    tau_eq7 = GRID[cost(GRID, 0.0).argmin()]; tau_c = GRID[cost(GRID, 'const').argmin()]
    reg = []
    for name, lam in MODELS[2:]:
        ct = cost(GRID, lam); opt = ct.min()
        reg.append(dict(true_model=name, tau_true=GRID[ct.argmin()],
                        tau_if_tuned_on_Eq7=tau_eq7, regret_Eq7_tuner_pct=100 * (float(cost(tau_eq7, lam)) / opt - 1),
                        tau_if_tuned_on_const_p=tau_c, regret_constp_tuner_pct=100 * (float(cost(tau_c, lam)) / opt - 1)))
    return df, pd.DataFrame(reg), GRID, base

def foc_check(lam):
    """Proposition 3: LR* = f1/f0 = Nneg (CFP + CFN Npos TPR |p'|) / (CFN Npos (p - Npos TPR |p'|)) at the numerical optimum."""
    c = cost(GRID, lam); tau = GRID[c.argmin()]
    f0 = norm.pdf((tau - MU0) / S0) / S0; f1 = norm.pdf((MU1 - tau) / S1) / S1
    V = VOL(tau); p = float(p_of_V(V, lam)); dp = lam * p / VREF
    lhs = f1 / f0
    rhs = NNEG * (CFP + CFN * NPOS * TPR(tau) * dp) / (CFN * NPOS * (p - NPOS * TPR(tau) * dp))
    return tau, lhs, rhs

if __name__ == '__main__':
    t0 = time.time()
    raw, meta = monte_carlo()
    summ = summarise(raw)
    raw.to_csv(f'{OUT}/mc_raw.csv', index=False)
    summ.to_csv(f'{OUT}/mc_estimator_performance.csv', index=False)
    meta.to_csv(f'{OUT}/mc_scenario_diagnostics.csv', index=False)
    tt, reg, grid, base = threshold_table()
    tt.to_csv(f'{OUT}/threshold_fatigue_table.csv', index=False)
    reg.to_csv(f'{OUT}/threshold_regret_table.csv', index=False)
    foc = pd.DataFrame([dict(lam=k, tau_opt=a, LR_at_opt=b, LR_formula=c_) for k in (0.0, LAM_CAL, 0.7, 1.0, 1.5)
                        for a, b, c_ in [foc_check(k)]])
    foc.to_csv(f'{OUT}/threshold_foc_check.csv', index=False)
    info = dict(AUC_binormal=AUC, mu0=MU0, s0=S0, mu1=MU1, s1=S1, Npos=NPOS, Nneg=NNEG, Vref=VREF,
                lam_crit_at_ref=VREF / (NPOS * TPR(0.105)), lam_cal=LAM_CAL,
                eq7_cost_at_0p70=float(cost(0.70, 0.0)), eq7_change_at_0p70_pct=100 * (float(cost(0.70, 0.0)) / base - 1),
                eq7_cost_at_0p105=float(cost(0.105, 0.0)))
    json.dump(info, open(f'{OUT}/threshold_model_calibration.json', 'w'), indent=2)
    print(json.dumps(info, indent=2)); print(tt.round(3).to_string()); print(reg.round(3).to_string()); print(foc.round(4).to_string())
    print(meta.round(3).to_string())
    print('elapsed', time.time() - t0)
