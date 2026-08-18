"""
Experiment B + C: Cross-Domain Transfer Matrix & Ablation Study
================================================================
B: Train on site X → test site Y (zero-shot + few-shot 3d)
C: Ablation — what matters? (physics prior, pre-train, residual)
All on DuraMAT REAL data, PyTorch GPU.
"""
import sys, io, os, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import torch, torch.nn as nn, torch.optim as optim
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = r"c:\Users\admin\NCKH\data_duramat\pert\Data For Validating Models"
RESULT_DIR = r"c:\Users\admin\NCKH\results"
os.makedirs(RESULT_DIR, exist_ok=True)
COL_MAP = {0:"timestamp",1:"POA",3:"T_module",20:"T_ambient",22:"RH",30:"GHI"}
FEAT = ["T_Faiman_RH","T_ambient","POA","GHI_lag","RH"]
FEAT_DIRECT = ["T_ambient","POA","GHI_lag","RH"]  # no physics baseline
N_SEEDS = 20   # nâng từ 5 lên 20 (2026-08-22) — tái sinh Table 5 chuẩn n=20

def log(s): print(s); sys.stdout.flush()

def read_pert(fp):
    rows=[]
    with open(fp) as f:
        f.readline();f.readline();f.readline()
        for line in f:
            p=line.strip().split(",")
            if len(p)>=42: rows.append({n:p[i] for i,n in COL_MAP.items()})
    df=pd.DataFrame(rows); df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    for c in df.columns:
        if c!="timestamp": df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.replace(-9999.0,float("nan")).dropna(subset=["timestamp"])

def add_features(df, u0a, u0b):
    df=df.copy()
    u0=np.clip(u0a+u0b*df["RH"].values,10,100)
    df["T_Faiman_RH"]=df["T_ambient"]+df["POA"]/(u0+6.84*2.0)
    df["GHI_lag"]=df["GHI"].shift(6)
    df=df.dropna(subset=FEAT+["T_module"])
    return df[df["POA"]>50].copy()

class ResidualMLP(nn.Module):
    def __init__(self, n_in=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in,32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,1))
    def forward(self, x): return self.net(x)
    def count(self): return sum(p.numel() for p in self.parameters())

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# =====================================================================
log("="*70)
log(f"  EXPERIMENTS B+C (Device: {DEVICE})")
log("="*70)

# Load all sites
co_raw = read_pert(os.path.join(BASE,"Golden","Golden_mSi0247.csv"))
fl_raw = read_pert(os.path.join(BASE,"Cocoa","Cocoa_mSi0166.csv"))
or_raw = read_pert(os.path.join(BASE,"Eugene","Eugene_mSi0166.csv"))

# Fit U0
co_tmp = co_raw.dropna(subset=["T_ambient","POA","RH","T_module"])
co_tmp = co_tmp[co_tmp["POA"]>50].iloc[:int(len(co_tmp[co_tmp["POA"]>50])*0.8)]
dt=co_tmp["T_module"].values-co_tmp["T_ambient"].values; v=dt>1
popt,_=curve_fit(lambda rh,a,b:a+b*rh,co_tmp["RH"].values[v],
                 co_tmp["POA"].values[v]/dt[v]-6.84*2,p0=[30,-0.1])
U0_A,U0_B=popt

co = add_features(co_raw, U0_A, U0_B)
fl = add_features(fl_raw, U0_A, U0_B)
or_ = add_features(or_raw, U0_A, U0_B)

# Splits: 80% train/pool, 20% test
sites = {
    "CO": (co.iloc[:int(len(co)*0.8)], co.iloc[int(len(co)*0.8):]),
    "FL": (fl.iloc[:int(len(fl)*0.8)], fl.iloc[int(len(fl)*0.8):]),
    "OR": (or_.iloc[:int(len(or_)*0.8)], or_.iloc[int(len(or_)*0.8):]),
}

for name, (pool, te) in sites.items():
    log(f"  {name}: pool={len(pool):,}  test={len(te):,}  dates={pool['timestamp'].dt.date.nunique()}")

# =====================================================================
# EXPERIMENT B: Cross-Domain Transfer Matrix
# =====================================================================
log(f"\n{'='*70}")
log("  EXPERIMENT B: CROSS-DOMAIN TRANSFER MATRIX")
log(f"{'='*70}")

def pretrain(train_df, sc, epochs=300, seed=42):
    """Pre-train ResidualMLP on source data."""
    set_seed(seed)
    X = torch.from_numpy(sc.transform(train_df[FEAT].values).astype(np.float32)).to(DEVICE)
    r = torch.from_numpy((train_df["T_module"].values-train_df["T_Faiman_RH"].values).astype(np.float32)).unsqueeze(1).to(DEVICE)
    n = int(len(X)*0.85)
    model = ResidualMLP().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)
    best_vl, pat, best_st = float('inf'), 0, None
    for ep in range(epochs):
        model.train(); idx=torch.randperm(n,device=DEVICE)
        for i in range(0,n,128):
            sl=idx[i:min(i+128,n)]
            loss=nn.functional.mse_loss(model(X[sl]),r[sl])
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad(): vl=nn.functional.mse_loss(model(X[n:]),r[n:]).item()
        sched.step(vl)
        if vl<best_vl: best_vl=vl;best_st={k:v.clone() for k,v in model.state_dict().items()};pat=0
        else:
            pat+=1
            if pat>=20:break
    model.load_state_dict(best_st)
    return {k:v.clone() for k,v in model.state_dict().items()}

def eval_zeroshot(base_state, test_df, sc):
    model = ResidualMLP().to(DEVICE)
    model.load_state_dict(base_state)
    model.eval()
    X = torch.from_numpy(sc.transform(test_df[FEAT].values).astype(np.float32)).to(DEVICE)
    Tf = torch.from_numpy(test_df["T_Faiman_RH"].values.astype(np.float32)).to(DEVICE)
    y = torch.from_numpy(test_df["T_module"].values.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        pred = Tf + model(X).squeeze()
        return nn.functional.l1_loss(pred, y).item()

def fewshot_3d(base_state, pool_df, test_df, sc, n_seeds=N_SEEDS):
    """Few-shot 3d with multiple random date selections."""
    pool_dates = pool_df["timestamp"].dt.date.unique()
    maes = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        sel = np.random.choice(pool_dates, size=min(3, len(pool_dates)), replace=False)
        ft_df = pool_df[pool_df["timestamp"].dt.date.isin(sel)]
        X_ft = torch.from_numpy(sc.transform(ft_df[FEAT].values).astype(np.float32)).to(DEVICE)
        r_ft = torch.from_numpy((ft_df["T_module"].values-ft_df["T_Faiman_RH"].values).astype(np.float32)).unsqueeze(1).to(DEVICE)
        X_te = torch.from_numpy(sc.transform(test_df[FEAT].values).astype(np.float32)).to(DEVICE)
        Tf_te = torch.from_numpy(test_df["T_Faiman_RH"].values.astype(np.float32)).to(DEVICE)
        y_te = torch.from_numpy(test_df["T_module"].values.astype(np.float32)).to(DEVICE)
        
        n = len(X_ft); n_tr = max(int(n*0.85),1)
        set_seed(seed)
        model = ResidualMLP().to(DEVICE)
        model.load_state_dict(base_state)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5, min_lr=1e-6)
        best_mae = float('inf'); best_s = None
        for ep in range(150):
            model.train(); idx=torch.randperm(n_tr,device=DEVICE)
            for i in range(0,n_tr,min(64,n_tr)):
                sl=idx[i:min(i+64,n_tr)]
                loss=nn.functional.mse_loss(model(X_ft[sl]),r_ft[sl])
                opt.zero_grad();loss.backward();opt.step()
            model.eval()
            with torch.no_grad():
                vl=nn.functional.mse_loss(model(X_ft[n_tr:] if n>n_tr else X_ft),
                                           r_ft[n_tr:] if n>n_tr else r_ft).item()
                sched.step(vl)
                pred=Tf_te+model(X_te).squeeze()
                mae=nn.functional.l1_loss(pred,y_te).item()
            if mae<best_mae: best_mae=mae; best_s={k:v.clone() for k,v in model.state_dict().items()}
        maes.append(best_mae)
    return np.mean(maes), np.std(maes)

# Scaler: fit on CO
sc_co = StandardScaler().fit(sites["CO"][0][FEAT].values)

# Pre-train on each source
log("\n  Pre-training on each source site...")
pretrained = {}
for src in ["CO","FL","OR"]:
    t0 = time.time()
    pretrained[src] = pretrain(sites[src][0], sc_co)
    log(f"    {src} pretrained ({time.time()-t0:.1f}s)")

# Physics baseline
log("\n  Physics baselines:")
for tgt in ["CO","FL","OR"]:
    te = sites[tgt][1]
    mae_frh = mean_absolute_error(te["T_module"], te["T_Faiman_RH"])
    log(f"    {tgt} Faiman+RH: {mae_frh:.3f}")

# Transfer matrix
log("\n  Cross-domain transfer:")
results_B = []
log(f"\n  {'Train→Test':<12} {'Faiman+RH':>10} {'Zero-shot':>10} {'Few-shot 3d':>12} {'Improvement':>12}")
log(f"  {'-'*58}")

for src in ["CO","FL","OR"]:
    for tgt in ["CO","FL","OR"]:
        te = sites[tgt][1]
        pool = sites[tgt][0]
        
        # Physics baseline
        mae_physics = mean_absolute_error(te["T_module"], te["T_Faiman_RH"])
        
        if src == tgt:
            # Same site — full data training result
            zs = eval_zeroshot(pretrained[src], te, sc_co)
            results_B.append({"Source":src,"Target":tgt,"Faiman_RH":mae_physics,
                            "ZeroShot":zs,"FewShot3d":zs,"Improvement":mae_physics-zs})
            log(f"  {src}→{tgt} (self) {mae_physics:>10.3f} {zs:>10.3f} {'    (same)':>12} {mae_physics-zs:>+12.3f}")
        else:
            # Cross-domain
            zs = eval_zeroshot(pretrained[src], te, sc_co)
            fs_mean, fs_std = fewshot_3d(pretrained[src], pool, te, sc_co)
            imp = mae_physics - fs_mean
            results_B.append({"Source":src,"Target":tgt,"Faiman_RH":mae_physics,
                            "ZeroShot":zs,"FewShot3d":fs_mean,"FewShot_std":fs_std,
                            "Improvement":imp})
            log(f"  {src}→{tgt}       {mae_physics:>10.3f} {zs:>10.3f} {fs_mean:>7.3f}±{fs_std:.3f} {imp:>+12.3f}")

pd.DataFrame(results_B).to_csv(os.path.join(RESULT_DIR,"expB_crossdomain.csv"), index=False)

# Summary
log(f"\n  SUMMARY:")
cross = [r for r in results_B if r['Source']!=r['Target']]
improved = [r for r in cross if r['Improvement']>0]
log(f"  Cross-domain pairs: {len(cross)}")
log(f"  Few-shot beats Faiman: {len(improved)}/{len(cross)}")
avg_imp = np.mean([r['Improvement'] for r in improved]) if improved else 0
log(f"  Average improvement: {avg_imp:+.3f}°C")

# =====================================================================
# EXPERIMENT C: ABLATION STUDY
# =====================================================================
log(f"\n{'='*70}")
log("  EXPERIMENT C: ABLATION STUDY — What makes it work?")
log(f"{'='*70}")

results_C = []

# Helper: train from scratch (no pretrain)
def train_scratch(train_df, test_df, sc, n_in=5, feat=FEAT, residual=True, n_seeds=N_SEEDS):
    maes = []
    for seed in range(n_seeds):
        set_seed(seed)
        X_tr = torch.from_numpy(sc.transform(train_df[feat].values).astype(np.float32)).to(DEVICE)
        X_te = torch.from_numpy(sc.transform(test_df[feat].values).astype(np.float32)).to(DEVICE)
        Tf_te = torch.from_numpy(test_df["T_Faiman_RH"].values.astype(np.float32)).to(DEVICE)
        y_te = torch.from_numpy(test_df["T_module"].values.astype(np.float32)).to(DEVICE)
        
        if residual:
            y_tr = torch.from_numpy((train_df["T_module"].values-train_df["T_Faiman_RH"].values).astype(np.float32)).unsqueeze(1).to(DEVICE)
        else:
            y_tr = torch.from_numpy(train_df["T_module"].values.astype(np.float32)).unsqueeze(1).to(DEVICE)
        
        n = int(len(X_tr)*0.85)
        model = ResidualMLP(n_in).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)
        best_vl,pat,best_st=float('inf'),0,None
        for ep in range(200):
            model.train(); idx=torch.randperm(n,device=DEVICE)
            for i in range(0,n,128):
                sl=idx[i:min(i+128,n)]
                loss=nn.functional.mse_loss(model(X_tr[sl]),y_tr[sl])
                opt.zero_grad();loss.backward();opt.step()
            model.eval()
            with torch.no_grad(): vl=nn.functional.mse_loss(model(X_tr[n:]),y_tr[n:]).item()
            sched.step(vl)
            if vl<best_vl: best_vl=vl;best_st={k:v.clone() for k,v in model.state_dict().items()};pat=0
            else:
                pat+=1
                if pat>=20:break
        model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            if residual:
                pred = Tf_te + model(X_te).squeeze()
            else:
                pred = model(X_te).squeeze()
            mae = nn.functional.l1_loss(pred, y_te).item()
        maes.append(mae)
    return np.mean(maes), np.std(maes)

# Test on FL and OR
for tgt_name, tgt_pool, tgt_te in [("FL", sites["FL"][0], sites["FL"][1]),
                                     ("OR", sites["OR"][0], sites["OR"][1])]:
    log(f"\n  --- Target: {tgt_name} ---")
    
    # Physics baselines
    mae_faiman = mean_absolute_error(tgt_te["T_module"], tgt_te["T_Faiman_RH"])
    log(f"  Faiman+RH:           {mae_faiman:.3f}")
    
    # A1: Direct MLP (no physics baseline, predict T_module directly)
    log(f"  Training ablation variants...")
    sc_direct = StandardScaler().fit(sites["CO"][0][FEAT_DIRECT].values)
    m1, s1 = train_scratch(tgt_pool, tgt_te, sc_direct, n_in=4, feat=FEAT_DIRECT, residual=False)
    log(f"  A1 Direct MLP (no physics):   {m1:.3f}±{s1:.3f}")
    results_C.append({"Target":tgt_name,"Variant":"A1: Direct MLP (no physics)","MAE":m1,"std":s1})
    
    # A2: Residual MLP, trained from scratch on target (no pre-train)
    m2, s2 = train_scratch(tgt_pool, tgt_te, sc_co, residual=True)
    log(f"  A2 Residual scratch (no CO):  {m2:.3f}±{s2:.3f}")
    results_C.append({"Target":tgt_name,"Variant":"A2: Residual scratch (no pretrain)","MAE":m2,"std":s2})
    
    # A3: CO pre-trained, zero-shot (no adaptation)
    zs_mae = eval_zeroshot(pretrained["CO"], tgt_te, sc_co) 
    log(f"  A3 CO pretrained zero-shot:   {zs_mae:.3f}")
    results_C.append({"Target":tgt_name,"Variant":"A3: CO pretrained zero-shot","MAE":zs_mae,"std":0})
    
    # A4: CO pre-trained + few-shot 3d (FULL METHOD)
    fs_mean, fs_std = fewshot_3d(pretrained["CO"], tgt_pool, tgt_te, sc_co)
    log(f"  A4 CO pretrain + FS 3d (FULL):{fs_mean:.3f}±{fs_std:.3f}")
    results_C.append({"Target":tgt_name,"Variant":"A4: Full method (pretrain+fewshot)","MAE":fs_mean,"std":fs_std})
    
    # A5: Residual MLP, scratch, only 3 days target data (no pre-train, few data)
    pool_dates = tgt_pool["timestamp"].dt.date.unique()
    maes_a5 = []
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        sel = np.random.choice(pool_dates, size=3, replace=False)
        ft_df = tgt_pool[tgt_pool["timestamp"].dt.date.isin(sel)]
        m5, _ = train_scratch(ft_df, tgt_te, sc_co, residual=True, n_seeds=1)
        maes_a5.append(m5)
    m5_mean = np.mean(maes_a5); m5_std = np.std(maes_a5)
    log(f"  A5 Scratch 3d (no pretrain):  {m5_mean:.3f}±{m5_std:.3f}")
    results_C.append({"Target":tgt_name,"Variant":"A5: Scratch 3d (no pretrain)","MAE":m5_mean,"std":m5_std})
    
    # A6: Physics baseline only
    results_C.append({"Target":tgt_name,"Variant":"A0: Faiman+RH (physics only)","MAE":mae_faiman,"std":0})

pd.DataFrame(results_C).to_csv(os.path.join(RESULT_DIR,"expC_ablation.csv"), index=False)

# =====================================================================
# FINAL TABLES
# =====================================================================
log(f"\n{'='*70}")
log("  FINAL ABLATION TABLE")
log(f"{'='*70}")

log(f"\n  {'Variant':<45} {'FL MAE':>8} {'OR MAE':>8} {'Avg':>8}")
log(f"  {'-'*72}")

variants_order = [
    "A0: Faiman+RH (physics only)",
    "A1: Direct MLP (no physics)",
    "A2: Residual scratch (no pretrain)",
    "A5: Scratch 3d (no pretrain)",
    "A3: CO pretrained zero-shot",
    "A4: Full method (pretrain+fewshot)",
]

for var in variants_order:
    fl_r = next((r for r in results_C if r['Target']=='FL' and r['Variant']==var), None)
    or_r = next((r for r in results_C if r['Target']=='OR' and r['Variant']==var), None)
    fl_m = fl_r['MAE'] if fl_r else 0
    or_m = or_r['MAE'] if or_r else 0
    avg = (fl_m + or_m) / 2
    marker = " ← OURS" if "Full method" in var else ""
    log(f"  {var:<45} {fl_m:>8.3f} {or_m:>8.3f} {avg:>8.3f}{marker}")

# Relative improvements
log(f"\n  CONTRIBUTION OF EACH COMPONENT:")
for tgt in ["FL","OR"]:
    log(f"\n  {tgt}:")
    a0 = next(r['MAE'] for r in results_C if r['Target']==tgt and 'A0' in r['Variant'])
    a1 = next(r['MAE'] for r in results_C if r['Target']==tgt and 'A1' in r['Variant'])
    a2 = next(r['MAE'] for r in results_C if r['Target']==tgt and 'A2' in r['Variant'])
    a3 = next(r['MAE'] for r in results_C if r['Target']==tgt and 'A3' in r['Variant'])
    a4 = next(r['MAE'] for r in results_C if r['Target']==tgt and 'A4' in r['Variant'])
    a5 = next(r['MAE'] for r in results_C if r['Target']==tgt and 'A5' in r['Variant'])
    
    log(f"    Physics baseline (Faiman+RH):  {a0:.3f}")
    log(f"    + Residual learning:           {a2:.3f} ({a0-a2:+.3f})")
    log(f"    + CO pre-training:             {a3:.3f} ({a2-a3:+.3f} vs scratch)")
    log(f"    + Few-shot 3d:                 {a4:.3f} ({a3-a4:+.3f} vs zero-shot)")
    log(f"    Total improvement:             {a0-a4:+.3f} ({(a0-a4)/a0*100:.1f}%)")
    log(f"")
    log(f"    WITHOUT pre-train (scratch 3d): {a5:.3f}")
    log(f"    Value of pre-training:          {a5-a4:+.3f} MAE reduction")

log(f"\n{'='*70}")
log("  EXPERIMENTS B+C COMPLETE")
log(f"{'='*70}")
