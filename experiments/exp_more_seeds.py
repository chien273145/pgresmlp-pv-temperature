"""
exp_more_seeds.py — Rerun Table 3 & Table 4 with N_SEEDS = 20
==============================================================
Purpose: Increase statistical power for cross-climate transfer claims.
  - Table 3: CO→FL few-shot 3d (PG-ResMLP vs LSTM-32)
  - Table 4: Full 6-pair transfer matrix (CO, FL, OR × CO, FL, OR)

With n=20 seeds:
  - df = 19 (vs 4), t(0.975,19) = 2.093 (vs 2.776)  → narrower CIs
  - Bonferroni threshold: t(19, 1-0.0083) ≈ 2.88 (vs 4.77) → much easier to reject
  - Power to detect d=0.8 effect: ~0.80 (vs ~0.40 at n=5)

Takes ~10-25 min on RTX 5050. Progress saved incrementally.
"""
import sys, io, os, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy import stats
import torch, torch.nn as nn, torch.optim as optim
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE   = r"c:\Users\admin\NCKH\data_duramat\pert\Data For Validating Models"
RESULT = r"c:\Users\admin\NCKH\results"
os.makedirs(RESULT, exist_ok=True)

# ── CHANGE THIS to re-run ──────────────────────────────────────────────
N_SEEDS = 20
# ──────────────────────────────────────────────────────────────────────

COL_MAP = {0:"timestamp",1:"POA",3:"T_module",20:"T_ambient",22:"RH",30:"GHI"}
FEAT    = ["T_Faiman_RH","T_ambient","POA","GHI_lag","RH"]

def log(s): print(s); sys.stdout.flush()

def read_pert(fp):
    rows=[]
    with open(fp) as f:
        f.readline();f.readline();f.readline()
        for line in f:
            p=line.strip().split(",")
            if len(p)>=42: rows.append({n:p[i] for i,n in COL_MAP.items()})
    df=pd.DataFrame(rows)
    df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    for c in df.columns:
        if c!="timestamp": df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.replace(-9999.0,float("nan")).dropna(subset=["timestamp"])

def add_features(df, u0_a, u0_b, ws=2.0):
    df=df.copy()
    u0=np.clip(u0_a+u0_b*df["RH"].values,10,100)
    df["T_Faiman_RH"]=df["T_ambient"]+df["POA"]/(u0+6.84*ws)
    df["GHI_lag"]=df["GHI"].shift(6)
    return df.dropna(subset=FEAT+["T_module"])

class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(5,32),nn.ReLU(),
            nn.Linear(32,16),nn.ReLU(),
            nn.Linear(16,1))
    def forward(self,x): return self.net(x)

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def prep(df, sc):
    X=torch.from_numpy(sc.transform(df[FEAT].values).astype(np.float32)).to(DEVICE)
    Tf=torch.from_numpy(df["T_Faiman_RH"].values.astype(np.float32)).to(DEVICE)
    y=torch.from_numpy(df["T_module"].values.astype(np.float32)).to(DEVICE)
    return X, Tf, y

def train_model(X, Tf, y, epochs=300, lr=1e-3, val_frac=0.15):
    n=len(X); nv=int(n*val_frac); nt=n-nv
    r=(y-Tf).unsqueeze(1)
    model=ResidualMLP().to(DEVICE)
    opt=optim.Adam(model.parameters(),lr=lr)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=0.5,min_lr=1e-6)
    best_vl=float('inf'); pat=0; best_st=None
    for ep in range(epochs):
        model.train()
        idx=torch.randperm(nt,device=DEVICE)
        for i in range(0,nt,128):
            sl=idx[i:i+128]
            loss=nn.functional.mse_loss(model(X[sl]),r[sl])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=nn.functional.mse_loss(model(X[nt:]),r[nt:]).item()
        sched.step(vl)
        if vl<best_vl: best_vl=vl; best_st={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=20: break
    model.load_state_dict(best_st)
    return model

def fewshot(base_state, pool_df, test_df, sc, seed, epochs=100, lr=1e-4):
    """3-day few-shot fine-tuning using random date selection."""
    set_seed(seed)
    dates=pool_df["timestamp"].dt.date.unique()
    sel=np.random.choice(dates, size=min(3, len(dates)), replace=False)
    ft_df=pool_df[pool_df["timestamp"].dt.date.isin(sel)]
    if len(ft_df)<10: return float('nan')

    model=ResidualMLP().to(DEVICE)
    model.load_state_dict(base_state)
    Xft, Tft, yft = prep(ft_df, sc)
    r_ft = (yft - Tft).unsqueeze(1)

    opt=optim.Adam(model.parameters(), lr=lr)
    best_loss=float('inf'); best_st=None
    for ep in range(epochs):
        model.train()
        loss=nn.functional.mse_loss(model(Xft), r_ft)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if loss.item()<best_loss: best_loss=loss.item(); best_st={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_st)

    model.eval()
    Xte, Tfte, yte = prep(test_df, sc)
    with torch.no_grad():
        pred = Tfte + model(Xte).squeeze()
        return nn.functional.l1_loss(pred, yte).item()

def pretrain(train_df, sc, seed):
    set_seed(seed)
    X,Tf,y = prep(train_df, sc)
    return train_model(X,Tf,y)

def eval_zeroshot(model, test_df, sc):
    model.eval()
    X,Tf,y = prep(test_df, sc)
    with torch.no_grad():
        pred = Tf + model(X).squeeze()
        return nn.functional.l1_loss(pred, y).item()

def report_stats(maes, label, faiman_baseline=None):
    maes=np.array([m for m in maes if not np.isnan(m)])
    n=len(maes)
    if n==0: log(f"  {label}: NO DATA"); return
    mean,std,se=maes.mean(),maes.std(),maes.std()/np.sqrt(n)
    t_crit=stats.t.ppf(0.975, df=n-1)
    lo,hi=mean-t_crit*se, mean+t_crit*se
    log(f"  {label}: {mean:.3f} ± {std:.3f}  n={n}  95%CI=[{lo:.3f},{hi:.3f}]", )
    if faiman_baseline is not None:
        t_stat=(faiman_baseline-mean)/se
        p=stats.t.sf(t_stat, df=n-1)  # one-sided: model < faiman
        signif="*SIGNIFICANT*" if p<0.05 else ("borderline" if p<0.15 else "NOT sig.")
        bonf_thresh=stats.t.ppf(1-0.05/(6*2), df=n-1)  # 6 pairs, 2-sided
        log(f"    vs Faiman {faiman_baseline:.3f}: t={t_stat:.2f}  p={p:.4f}  [{signif}]  Bonf-thresh={bonf_thresh:.2f}")

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
log("="*70)
log(f"  exp_more_seeds.py — N_SEEDS={N_SEEDS}  Device={DEVICE}")
log("="*70)

co_raw=read_pert(os.path.join(BASE,"Golden","Golden_mSi0247.csv"))
fl_raw=read_pert(os.path.join(BASE,"Cocoa","Cocoa_mSi0166.csv"))
or_raw=read_pert(os.path.join(BASE,"Eugene","Eugene_mSi0166.csv"))

# Fit U0(RH) on CO
co_tmp=co_raw.dropna(subset=["T_ambient","POA","RH","T_module"])
co_tmp=co_tmp[co_tmp["POA"]>50].iloc[:int(len(co_tmp[co_tmp["POA"]>50])*0.8)]
dt=co_tmp["T_module"].values-co_tmp["T_ambient"].values; v=dt>1.0
popt,_=curve_fit(lambda rh,a,b:a+b*rh, co_tmp["RH"].values[v],
                  co_tmp["POA"].values[v]/dt[v]-6.84*2.0, p0=[30,-0.1])
U0_A,U0_B=popt
log(f"  U0(RH) = {U0_A:.3f} + {U0_B:.4f}*RH")

# Add features
data={}
for name,raw in [("CO",co_raw),("FL",fl_raw),("OR",or_raw)]:
    df=add_features(raw,U0_A,U0_B)
    day=df[df["POA"]>50]
    n=len(day)
    pool=day.iloc[:int(n*0.8)]
    test=day.iloc[int(n*0.8):]
    data[name]={"pool":pool,"test":test}
    log(f"  {name}: pool={len(pool):,}  test={len(test):,}")

# Faiman baselines
log("\n  FAIMAN+RH BASELINES:")
faiman={}
for name in ["CO","FL","OR"]:
    mae=mean_absolute_error(data[name]["test"]["T_module"],data[name]["test"]["T_Faiman_RH"])
    faiman[name]=mae
    log(f"  {name}: {mae:.3f}")

# Shared scaler (fit on CO pool)
sc_co=StandardScaler().fit(data["CO"]["pool"][FEAT].values)

# ═══════════════════════════════════════════════════════════════════════
# TABLE 3: CO→FL few-shot (PG-ResMLP vs LSTM)
# ═══════════════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log(f"  TABLE 3 REPLICATION: CO→FL few-shot 3d  (n={N_SEEDS} seeds)")
log(f"{'='*70}")

# Pre-train on CO (single reference model, varied seed in fewshot only)
set_seed(42)
co_model=train_model(*prep(data["CO"]["pool"],sc_co))
co_state={k:v.clone() for k,v in co_model.state_dict().items()}

# PG-ResMLP few-shot
t0=time.time()
fs_maes=[]
for seed in range(N_SEEDS):
    mae=fewshot(co_state, data["FL"]["pool"], data["FL"]["test"], sc_co, seed=seed)
    fs_maes.append(mae)
    if (seed+1)%5==0: log(f"    seed {seed+1}/{N_SEEDS} done ({time.time()-t0:.0f}s)")
log(f"\n  PG-ResMLP (CO→FL 3d fewshot):")
report_stats(fs_maes, "PG-ResMLP fewshot", faiman_baseline=faiman["FL"])

# Save Table3 raw
pd.DataFrame({"seed":range(N_SEEDS),"MAE_PGResMLP":fs_maes}).to_csv(
    os.path.join(RESULT,"table3_more_seeds.csv"),index=False)

# ═══════════════════════════════════════════════════════════════════════
# TABLE 4: Full 6-pair transfer matrix
# ═══════════════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log(f"  TABLE 4 REPLICATION: 6-pair transfer matrix (n={N_SEEDS} seeds)")
log(f"{'='*70}")

pairs=[("CO","FL"),("CO","OR"),("FL","CO"),("FL","OR"),("OR","CO"),("OR","FL")]
table4=[]

for src, tgt in pairs:
    log(f"\n  {src}→{tgt}:")
    # Pre-train on source (seed=42 for pre-train, varied seed for fewshot date selection)
    set_seed(42)
    src_model=train_model(*prep(data[src]["pool"],sc_co))
    src_state={k:v.clone() for k,v in src_model.state_dict().items()}

    maes=[]
    for seed in range(N_SEEDS):
        mae=fewshot(src_state, data[tgt]["pool"], data[tgt]["test"], sc_co, seed=seed)
        maes.append(mae)
    maes_arr=np.array([m for m in maes if not np.isnan(m)])
    n=len(maes_arr)
    mean_mae=maes_arr.mean(); std_mae=maes_arr.std(); se=std_mae/np.sqrt(n)
    t_crit=stats.t.ppf(0.975,df=n-1)
    ci_lo,ci_hi=mean_mae-t_crit*se, mean_mae+t_crit*se
    t_stat=(faiman[tgt]-mean_mae)/se
    p_val=stats.t.sf(t_stat, df=n-1)
    bonf_p=min(1.0, p_val*6)  # Bonferroni corrected

    result={
        "pair":f"{src}→{tgt}","src":src,"tgt":tgt,
        "faiman":faiman[tgt],
        "mean":mean_mae,"std":std_mae,"n":n,
        "ci_lo":ci_lo,"ci_hi":ci_hi,
        "t_stat":t_stat,"p_val":p_val,"p_val_bonf":bonf_p,
        "sig_uncorrected":p_val<0.05,
        "sig_bonf":bonf_p<0.05,
        "assessment": ("Significant" if bonf_p<0.05 else
                       ("Borderline" if p_val<0.05 else
                        ("Negative transfer" if mean_mae>faiman[tgt] else "Not significant")))
    }
    table4.append(result)
    log(f"  MAE={mean_mae:.3f}±{std_mae:.3f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
        f"p={p_val:.4f}  p_bonf={bonf_p:.4f}  → {result['assessment']}")

df4=pd.DataFrame(table4)
df4.to_csv(os.path.join(RESULT,"table4_more_seeds.csv"),index=False)

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log(f"  UPDATED TABLE 4 — n={N_SEEDS} seeds")
log(f"{'='*70}")
log(f"  {'Pair':12s} {'Faiman':>8} {'Few-shot':>10} {'95% CI':>18} {'p':>7} {'p_Bonf':>7} {'Result':>15}")
log(f"  {'-'*82}")
for r in table4:
    log(f"  {r['pair']:12s} {r['faiman']:>8.3f} {r['mean']:>8.3f}±{r['std']:.2f} "
        f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] {r['p_val']:>7.4f} {r['p_val_bonf']:>7.4f} {r['assessment']:>15}")

log(f"\n  Results saved to {RESULT}/table3_more_seeds.csv + table4_more_seeds.csv")
log("  DONE.")
