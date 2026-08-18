"""
Experiments A+B+C FIXED v2 — Correct bugs
==========================================
ACTUAL Fixes applied:
  1. LSTM/GRU with proper sliding window (seq_len=12 = 3h @ 15min)
  2. Scaler fit on TRAINING data (FL scaler for FL full-data)
  3. Early stopping on VALIDATION loss only, test evaluated ONCE at end
  
NOTE: DuraMAT has NO wind speed column → WS=2.0 is correct assumption.
"""
import sys, io, os, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
import torch, torch.nn as nn, torch.optim as optim
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = r"c:\Users\admin\NCKH\data_duramat\pert\Data For Validating Models"
RESULT_DIR = r"c:\Users\admin\NCKH\results"
os.makedirs(RESULT_DIR, exist_ok=True)

COL_MAP = {0:"timestamp",1:"POA",3:"T_module",20:"T_ambient",22:"RH",30:"GHI"}
FEAT = ["T_Faiman_RH","T_ambient","POA","GHI_lag","RH"]
N_SEEDS = 5
SEQ_LEN = 12  # 12 x 15min = 3 hours

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
    df["T_Faiman_RH"]=df["T_ambient"]+df["POA"]/(u0+6.84*2.0)  # WS=2 (no WS in data)
    df["T_SAPM"]=df["POA"]*np.exp(-3.56-0.075*2.0)+df["T_ambient"]
    df["GHI_lag"]=df["GHI"].shift(6)   # 15-min resolution => lag = 90 min (paper Sec 3.2)
    df=df.dropna(subset=FEAT+["T_module"])
    return df[df["POA"]>50].copy()

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# Models
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.net(x)
    def param_count(self): return sum(p.numel() for p in self.parameters())

class LSTMSeq(nn.Module):
    """LSTM with proper sequence input."""
    def __init__(self, input_size=5, hidden=32):
        super().__init__()
        self.lstm=nn.LSTM(input_size,hidden,1,batch_first=True)
        self.fc=nn.Linear(hidden,1)
    def forward(self,x):  # x: (batch, seq_len, features)
        out,_=self.lstm(x)
        return self.fc(out[:,-1,:])
    def param_count(self): return sum(p.numel() for p in self.parameters())

class GRUSeq(nn.Module):
    def __init__(self, input_size=5, hidden=32):
        super().__init__()
        self.gru=nn.GRU(input_size,hidden,1,batch_first=True)
        self.fc=nn.Linear(hidden,1)
    def forward(self,x):
        out,_=self.gru(x)
        return self.fc(out[:,-1,:])
    def param_count(self): return sum(p.numel() for p in self.parameters())

def make_sequences(X, seq_len):
    """Sliding window: (N, feat) → (N-seq_len, seq_len, feat)"""
    return torch.stack([X[i-seq_len:i] for i in range(seq_len, len(X))])

# FIX #3: Train with val-only stopping, test eval ONCE
def train_fixed(model, X_tr, y_tr, X_te, Tf_te, y_te,
                epochs=200, lr=1e-3, bs=128, patience=20):
    n=len(X_tr); n_t=int(n*0.85)
    opt=optim.Adam(model.parameters(),lr=lr)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=0.5,min_lr=1e-6)
    best_vl,pat,best_st=float('inf'),0,None
    for ep in range(epochs):
        model.train(); idx=torch.randperm(n_t,device=DEVICE)
        for i in range(0,n_t,bs):
            sl=idx[i:min(i+bs,n_t)]
            loss=nn.functional.mse_loss(model(X_tr[sl]),y_tr[sl])
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            vl=nn.functional.mse_loss(model(X_tr[n_t:]),y_tr[n_t:]).item()
        sched.step(vl)
        if vl<best_vl: best_vl=vl;best_st={k:v.clone() for k,v in model.state_dict().items()};pat=0
        else:
            pat+=1
            if pat>=patience: break
    if best_st: model.load_state_dict(best_st)
    # TEST: evaluate ONCE
    model.eval()
    with torch.no_grad():
        pred=Tf_te+model(X_te).squeeze()
        mae=nn.functional.l1_loss(pred,y_te).item()
        rmse=torch.sqrt(nn.functional.mse_loss(pred,y_te)).item()
    return mae, rmse

# =====================================================================
log("="*70)
log(f"  EXPERIMENTS A+B+C FIXED v2 (Device: {DEVICE})")
log(f"  Fixes: LSTM seq={SEQ_LEN}, FL scaler for FL, val-only stopping")
log(f"  Note: WS=2.0 (DuraMAT has no wind speed column)")
log("="*70)

co_raw=read_pert(os.path.join(BASE,"Golden","Golden_mSi0247.csv"))
fl_raw=read_pert(os.path.join(BASE,"Cocoa","Cocoa_mSi0166.csv"))
or_raw=read_pert(os.path.join(BASE,"Eugene","Eugene_mSi0166.csv"))

co_tmp=co_raw.dropna(subset=["T_ambient","POA","RH","T_module"])
co_tmp=co_tmp[co_tmp["POA"]>50].iloc[:int(len(co_tmp[co_tmp["POA"]>50])*0.8)]
dt=co_tmp["T_module"].values-co_tmp["T_ambient"].values; v=dt>1
popt,_=curve_fit(lambda rh,a,b:a+b*rh,co_tmp["RH"].values[v],
                 co_tmp["POA"].values[v]/dt[v]-6.84*2.0,p0=[30,-0.1])
U0_A,U0_B=popt
log(f"  U0(RH) = {U0_A:.3f} + {U0_B:.4f}*RH")

co=add_features(co_raw,U0_A,U0_B)
fl=add_features(fl_raw,U0_A,U0_B)
or_=add_features(or_raw,U0_A,U0_B)

co_tr=co.iloc[:int(len(co)*0.8)]
fl_pool=fl.iloc[:int(len(fl)*0.8)]
fl_te=fl.iloc[int(len(fl)*0.8):]

# FIX #2: Separate scalers
sc_co=StandardScaler().fit(co_tr[FEAT].values)
sc_fl=StandardScaler().fit(fl_pool[FEAT].values)

fl_dates=fl_pool["timestamp"].dt.date.unique()
y_te_np=fl_te["T_module"].values
log(f"  CO:{len(co_tr):,}  FL pool:{len(fl_pool):,}  FL test:{len(fl_te):,}")

# =====================================================================
# EXP A — SCENARIO 1: Full-data FL
# =====================================================================
log(f"\n{'='*70}")
log("  EXP A — SCENARIO 1: Full-data FL (FIX: FL scaler)")
log(f"{'='*70}")

# FIX #2: FL scaler for FL training
X_fl=torch.from_numpy(sc_fl.transform(fl_pool[FEAT].values).astype(np.float32)).to(DEVICE)
r_fl=torch.from_numpy((fl_pool["T_module"].values-fl_pool["T_Faiman_RH"].values).astype(np.float32)).unsqueeze(1).to(DEVICE)
X_fl_te=torch.from_numpy(sc_fl.transform(fl_te[FEAT].values).astype(np.float32)).to(DEVICE)
Tf_fl_te=torch.from_numpy(fl_te["T_Faiman_RH"].values.astype(np.float32)).to(DEVICE)
y_fl_te=torch.from_numpy(fl_te["T_module"].values.astype(np.float32)).to(DEVICE)

# Sequences for LSTM/GRU
X_fl_seq=make_sequences(X_fl, SEQ_LEN)
r_fl_seq=r_fl[SEQ_LEN:]
X_fl_te_seq=make_sequences(X_fl_te, SEQ_LEN)
Tf_fl_te_seq=Tf_fl_te[SEQ_LEN:]
y_fl_te_seq=y_fl_te[SEQ_LEN:]

results_A = []

# Physics
for nm, pv in [("Faiman+RH", fl_te["T_Faiman_RH"].values), ("SAPM", fl_te["T_SAPM"].values)]:
    m=mean_absolute_error(y_te_np,pv); r2=np.sqrt(mean_squared_error(y_te_np,pv))
    results_A.append({"Model":nm,"Params":"3","MAE":m,"RMSE":r2,"Type":"Physics"})

# Classical ML
X_tr_sk=sc_fl.transform(fl_pool[FEAT].values)
r_tr_sk=fl_pool["T_module"].values-fl_pool["T_Faiman_RH"].values
X_te_sk=sc_fl.transform(fl_te[FEAT].values)

for nm,clf,pr in [("Linear Regression",LinearRegression(),6),
                  ("Ridge",Ridge(alpha=1.0),6),
                  ("Random Forest",RandomForestRegressor(200,max_depth=10,random_state=42,n_jobs=-1),"~50K")]:
    clf.fit(X_tr_sk,r_tr_sk)
    p=fl_te["T_Faiman_RH"].values+clf.predict(X_te_sk)
    m=mean_absolute_error(y_te_np,p); r2=np.sqrt(mean_squared_error(y_te_np,p))
    results_A.append({"Model":nm,"Params":pr,"MAE":m,"RMSE":r2,"Type":"ML"})
    log(f"  {nm:30s} MAE={m:.3f}")

try:
    from xgboost import XGBRegressor
    xg=XGBRegressor(200,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
    xg.fit(X_tr_sk,r_tr_sk)
    p=fl_te["T_Faiman_RH"].values+xg.predict(X_te_sk)
    m=mean_absolute_error(y_te_np,p); r2=np.sqrt(mean_squared_error(y_te_np,p))
    results_A.append({"Model":"XGBoost","Params":"~5K","MAE":m,"RMSE":r2,"Type":"ML"})
    log(f"  {'XGBoost':30s} MAE={m:.3f}")
except: pass

# Neural — point-wise models
log(f"\n  Neural — ResidualMLP (5 seeds):")
mlp_m=[]
for s in range(N_SEEDS):
    set_seed(s); mdl=ResidualMLP().to(DEVICE)
    m,r2=train_fixed(mdl,X_fl,r_fl,X_fl_te,Tf_fl_te,y_fl_te); mlp_m.append(m)
sm=np.array(mlp_m)
results_A.append({"Model":"ResidualMLP (ours)","Params":737,"MAE":sm.mean(),"MAE_std":sm.std(),"RMSE":0,"Type":"Neural"})
log(f"    MAE={sm.mean():.3f}±{sm.std():.3f}")

# FIX #1: LSTM with proper sequences
log(f"  Neural — LSTM-32 (seq={SEQ_LEN}, 5 seeds):")
lstm_m=[]
for s in range(N_SEEDS):
    set_seed(s); mdl=LSTMSeq().to(DEVICE)
    m,r2=train_fixed(mdl,X_fl_seq,r_fl_seq,X_fl_te_seq,Tf_fl_te_seq,y_fl_te_seq); lstm_m.append(m)
sm=np.array(lstm_m)
results_A.append({"Model":f"LSTM-32 (seq={SEQ_LEN})","Params":LSTMSeq().param_count(),"MAE":sm.mean(),"MAE_std":sm.std(),"RMSE":0,"Type":"Neural"})
log(f"    MAE={sm.mean():.3f}±{sm.std():.3f}")

log(f"  Neural — GRU-32 (seq={SEQ_LEN}, 5 seeds):")
gru_m=[]
for s in range(N_SEEDS):
    set_seed(s); mdl=GRUSeq().to(DEVICE)
    m,r2=train_fixed(mdl,X_fl_seq,r_fl_seq,X_fl_te_seq,Tf_fl_te_seq,y_fl_te_seq); gru_m.append(m)
sm=np.array(gru_m)
results_A.append({"Model":f"GRU-32 (seq={SEQ_LEN})","Params":GRUSeq().param_count(),"MAE":sm.mean(),"MAE_std":sm.std(),"RMSE":0,"Type":"Neural"})
log(f"    MAE={sm.mean():.3f}±{sm.std():.3f}")

log(f"\n  TABLE 1: Full-data FL (CORRECTED)")
log(f"  {'Model':<35} {'Params':>8} {'MAE':>8}")
log(f"  {'-'*55}")
for r in sorted(results_A, key=lambda x: x['MAE']):
    std=f"±{r.get('MAE_std',0):.3f}" if r.get('MAE_std',0)>0 else ""
    log(f"  {r['Model']:<35} {str(r['Params']):>8} {r['MAE']:>7.3f}{std}")

# =====================================================================
# EXP A — SCENARIO 2: CO pretrain → FS 3d FL (FIXED)
# =====================================================================
log(f"\n{'='*70}")
log("  EXP A — SCENARIO 2: CO pretrain → Few-shot 3d FL (FIXED)")
log(f"{'='*70}")

# Pretrain MLP on CO
Xco=torch.from_numpy(sc_co.transform(co_tr[FEAT].values).astype(np.float32)).to(DEVICE)
rco=torch.from_numpy((co_tr["T_module"].values-co_tr["T_Faiman_RH"].values).astype(np.float32)).unsqueeze(1).to(DEVICE)
X_fl_te_co=torch.from_numpy(sc_co.transform(fl_te[FEAT].values).astype(np.float32)).to(DEVICE)

set_seed(42)
base_mlp=ResidualMLP().to(DEVICE)
train_fixed(base_mlp,Xco,rco,X_fl_te_co,Tf_fl_te,y_fl_te,epochs=300)
base_mlp_st={k:v.clone() for k,v in base_mlp.state_dict().items()}

# Zero-shot MLP
base_mlp.eval()
with torch.no_grad():
    zs_mlp=nn.functional.l1_loss(Tf_fl_te+base_mlp(X_fl_te_co).squeeze(),y_fl_te).item()
log(f"  ResidualMLP ZS: {zs_mlp:.3f}")

# Pretrain LSTM on CO
Xco_seq=make_sequences(Xco,SEQ_LEN); rco_seq=rco[SEQ_LEN:]
X_fl_te_co_seq=make_sequences(X_fl_te_co,SEQ_LEN)
set_seed(42)
base_lstm=LSTMSeq().to(DEVICE)
train_fixed(base_lstm,Xco_seq,rco_seq,X_fl_te_co_seq,Tf_fl_te_seq,y_fl_te_seq,epochs=300)
base_lstm_st={k:v.clone() for k,v in base_lstm.state_dict().items()}

base_lstm.eval()
with torch.no_grad():
    zs_lstm=nn.functional.l1_loss(Tf_fl_te_seq+base_lstm(X_fl_te_co_seq).squeeze(),y_fl_te_seq).item()
log(f"  LSTM-32 ZS:     {zs_lstm:.3f}")

# Few-shot 3d
results_fs=[]

def fewshot_v2(base_st, ModelCls, pool, test_X, test_Tf, test_y, sc, 
               is_seq=False, seq_len=12):
    pool_dates=pool["timestamp"].dt.date.unique()
    maes=[]
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        sel=np.random.choice(pool_dates,size=3,replace=False)
        ft=pool[pool["timestamp"].dt.date.isin(sel)]
        X_ft=torch.from_numpy(sc.transform(ft[FEAT].values).astype(np.float32)).to(DEVICE)
        r_ft=torch.from_numpy((ft["T_module"].values-ft["T_Faiman_RH"].values).astype(np.float32)).unsqueeze(1).to(DEVICE)
        if is_seq:
            if len(X_ft)<=seq_len: continue
            X_ft=make_sequences(X_ft,seq_len); r_ft=r_ft[seq_len:]
        set_seed(seed)
        mdl=ModelCls().to(DEVICE); mdl.load_state_dict(base_st)
        m,_=train_fixed(mdl,X_ft,r_ft,test_X,test_Tf,test_y,epochs=150,lr=5e-4,patience=25)
        maes.append(m)
    return np.mean(maes),np.std(maes)

log(f"\n  Few-shot 3d (5 seeds, val-only stopping):")
fs_mlp,fs_mlp_s=fewshot_v2(base_mlp_st,ResidualMLP,fl_pool,X_fl_te_co,Tf_fl_te,y_fl_te,sc_co)
log(f"  ResidualMLP FS3d: {fs_mlp:.3f}±{fs_mlp_s:.3f}")
results_fs.append({"Model":"ResidualMLP (ours)","Params":737,"ZS":zs_mlp,"FS3d":fs_mlp,"std":fs_mlp_s})

fs_lstm,fs_lstm_s=fewshot_v2(base_lstm_st,LSTMSeq,fl_pool,X_fl_te_co_seq,Tf_fl_te_seq,y_fl_te_seq,sc_co,True,SEQ_LEN)
log(f"  LSTM-32 FS3d:     {fs_lstm:.3f}±{fs_lstm_s:.3f}")
results_fs.append({"Model":"LSTM-32","Params":LSTMSeq().param_count(),"ZS":zs_lstm,"FS3d":fs_lstm,"std":fs_lstm_s})

# Classical ML FS3d
X_te_co_sk=sc_co.transform(fl_te[FEAT].values)
for nm,clf in [("Random Forest",RandomForestRegressor(100,max_depth=6,random_state=42)),
               ("XGBoost",None)]:
    seed_m=[]
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        sel=np.random.choice(fl_dates,size=3,replace=False)
        ft=fl_pool[fl_pool["timestamp"].dt.date.isin(sel)]
        X_ft_sk=sc_co.transform(ft[FEAT].values)
        r_ft_sk=ft["T_module"].values-ft["T_Faiman_RH"].values
        if nm=="XGBoost":
            try:
                c=XGBRegressor(100,max_depth=4,learning_rate=0.1,random_state=seed,verbosity=0)
                c.fit(X_ft_sk,r_ft_sk)
                p=fl_te["T_Faiman_RH"].values+c.predict(X_te_co_sk)
            except: continue
        else:
            clf.fit(X_ft_sk,r_ft_sk)
            p=fl_te["T_Faiman_RH"].values+clf.predict(X_te_co_sk)
        seed_m.append(mean_absolute_error(y_te_np,p))
    if seed_m:
        sm=np.array(seed_m)
        log(f"  {nm:20s} FS3d: {sm.mean():.3f}±{sm.std():.3f}")
        results_fs.append({"Model":nm,"Params":"~10K","ZS":"N/A","FS3d":sm.mean(),"std":sm.std()})

log(f"\n  TABLE 2: Few-shot 3d (CORRECTED)")
log(f"  {'Model':<25} {'Params':>8} {'ZeroShot':>10} {'FS3d':>10}")
log(f"  {'-'*58}")
for r in sorted(results_fs, key=lambda x: x['FS3d']):
    log(f"  {r['Model']:<25} {str(r['Params']):>8} {str(r['ZS']):>10} {r['FS3d']:>7.3f}±{r['std']:.3f}")

# =====================================================================
# EXP B: Cross-domain (FIXED)
# =====================================================================
log(f"\n{'='*70}")
log("  EXP B: CROSS-DOMAIN (FIXED — val-only stopping)")
log(f"{'='*70}")

sites={"CO":(co.iloc[:int(len(co)*0.8)],co.iloc[int(len(co)*0.8):]),
       "FL":(fl.iloc[:int(len(fl)*0.8)],fl.iloc[int(len(fl)*0.8):]),
       "OR":(or_.iloc[:int(len(or_)*0.8)],or_.iloc[int(len(or_)*0.8):])}

pretrained={}
for src in ["CO","FL","OR"]:
    set_seed(42)
    X=torch.from_numpy(sc_co.transform(sites[src][0][FEAT].values).astype(np.float32)).to(DEVICE)
    r=torch.from_numpy((sites[src][0]["T_module"].values-sites[src][0]["T_Faiman_RH"].values).astype(np.float32)).unsqueeze(1).to(DEVICE)
    mdl=ResidualMLP().to(DEVICE)
    # Use internal val for stopping (train_fixed does 85/15 split internally)
    train_fixed(mdl,X,r,X[:100],torch.zeros(100).to(DEVICE),torch.zeros(100).to(DEVICE),epochs=300)
    pretrained[src]={k:v.clone() for k,v in mdl.state_dict().items()}

results_B=[]
log(f"\n  {'Src→Tgt':<10} {'Physics':>8} {'ZS':>8} {'FS3d':>10} {'Imp':>8}")
log(f"  {'-'*48}")

for src in ["CO","FL","OR"]:
    for tgt in ["CO","FL","OR"]:
        if src==tgt: continue
        te=sites[tgt][1]; pool=sites[tgt][0]
        phys=mean_absolute_error(te["T_module"],te["T_Faiman_RH"])
        
        X_te_t=torch.from_numpy(sc_co.transform(te[FEAT].values).astype(np.float32)).to(DEVICE)
        Tf_t=torch.from_numpy(te["T_Faiman_RH"].values.astype(np.float32)).to(DEVICE)
        y_t=torch.from_numpy(te["T_module"].values.astype(np.float32)).to(DEVICE)
        
        mdl=ResidualMLP().to(DEVICE); mdl.load_state_dict(pretrained[src]); mdl.eval()
        with torch.no_grad():
            zs=nn.functional.l1_loss(Tf_t+mdl(X_te_t).squeeze(),y_t).item()
        
        fs,fs_s=fewshot_v2(pretrained[src],ResidualMLP,pool,X_te_t,Tf_t,y_t,sc_co)
        imp=phys-fs
        results_B.append({"Src":src,"Tgt":tgt,"Phys":phys,"ZS":zs,"FS3d":fs,"Imp":imp})
        log(f"  {src}→{tgt}     {phys:>8.3f} {zs:>8.3f} {fs:>6.3f}±{fs_s:.2f} {imp:>+8.3f}")

beat=sum(1 for r in results_B if r['Imp']>0)
log(f"\n  Beat Faiman: {beat}/{len(results_B)}")

# =====================================================================
# EXP C: ABLATION (FIXED)
# =====================================================================
log(f"\n{'='*70}")
log("  EXP C: ABLATION (FIXED)")
log(f"{'='*70}")

results_C=[]
for tgt_n in ["FL","OR"]:
    pool,te=sites[tgt_n]
    phys=mean_absolute_error(te["T_module"],te["T_Faiman_RH"])
    X_te_t=torch.from_numpy(sc_co.transform(te[FEAT].values).astype(np.float32)).to(DEVICE)
    Tf_t=torch.from_numpy(te["T_Faiman_RH"].values.astype(np.float32)).to(DEVICE)
    y_t=torch.from_numpy(te["T_module"].values.astype(np.float32)).to(DEVICE)
    X_pool_t=torch.from_numpy(sc_co.transform(pool[FEAT].values).astype(np.float32)).to(DEVICE)
    r_pool_t=torch.from_numpy((pool["T_module"].values-pool["T_Faiman_RH"].values).astype(np.float32)).unsqueeze(1).to(DEVICE)
    
    log(f"\n  {tgt_n}: Faiman+RH={phys:.3f}")
    
    # A2: scratch full
    a2m=[]
    for s in range(N_SEEDS):
        set_seed(s); m=ResidualMLP().to(DEVICE)
        mae,_=train_fixed(m,X_pool_t,r_pool_t,X_te_t,Tf_t,y_t); a2m.append(mae)
    a2=np.mean(a2m)
    log(f"    A2 Scratch full:     {a2:.3f}")
    
    # A3: CO pretrain ZS
    mdl=ResidualMLP().to(DEVICE); mdl.load_state_dict(pretrained["CO"]); mdl.eval()
    with torch.no_grad():
        a3=nn.functional.l1_loss(Tf_t+mdl(X_te_t).squeeze(),y_t).item()
    log(f"    A3 CO ZS:            {a3:.3f}")
    
    # A4: Full method
    a4,a4s=fewshot_v2(pretrained["CO"],ResidualMLP,pool,X_te_t,Tf_t,y_t,sc_co)
    log(f"    A4 Full (PT+FS3d):   {a4:.3f}±{a4s:.3f}")
    
    # A5: Scratch 3d
    pd_=pool["timestamp"].dt.date.unique()
    a5m=[]
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        sel=np.random.choice(pd_,size=3,replace=False)
        ft=pool[pool["timestamp"].dt.date.isin(sel)]
        X_ft=torch.from_numpy(sc_co.transform(ft[FEAT].values).astype(np.float32)).to(DEVICE)
        r_ft=torch.from_numpy((ft["T_module"].values-ft["T_Faiman_RH"].values).astype(np.float32)).unsqueeze(1).to(DEVICE)
        set_seed(seed); mdl=ResidualMLP().to(DEVICE)
        mae,_=train_fixed(mdl,X_ft,r_ft,X_te_t,Tf_t,y_t,epochs=150,lr=5e-4,patience=25)
        a5m.append(mae)
    a5=np.mean(a5m)
    log(f"    A5 Scratch 3d:       {a5:.3f}")
    
    results_C.append({"Tgt":tgt_n,"A0_Phys":phys,"A2_Scratch":a2,"A3_ZS":a3,"A4_Full":a4,"A5_Scratch3d":a5})
    log(f"\n    Breakdown:")
    log(f"      Faiman→Residual(full):  {phys:.3f}→{a2:.3f} ({phys-a2:+.3f})")
    log(f"      Pretrain value:         scratch3d={a5:.3f} vs full={a4:.3f} → {a5-a4:+.3f}")

# SAVE
pd.DataFrame(results_A).to_csv(os.path.join(RESULT_DIR,"expA_fixed_v2.csv"),index=False)
pd.DataFrame(results_fs).to_csv(os.path.join(RESULT_DIR,"expA_fs_fixed_v2.csv"),index=False)
pd.DataFrame(results_B).to_csv(os.path.join(RESULT_DIR,"expB_fixed_v2.csv"),index=False)
pd.DataFrame(results_C).to_csv(os.path.join(RESULT_DIR,"expC_fixed_v2.csv"),index=False)

log(f"\n{'='*70}")
log("  ALL DONE — RESULTS SAVED")
log(f"{'='*70}")
