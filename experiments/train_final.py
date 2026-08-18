"""
Phase 1 FINAL — Keras(64,32) for all experiments
Single architecture, single pipeline, definitive results
"""
import pandas as pd
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

DATA = r'c:\Users\admin\NCKH\processed\pvdaq_7333_v2_2022_2023.csv'
MODEL_DIR = r'c:\Users\admin\NCKH\models'
RESULT_DIR = r'c:\Users\admin\NCKH\results'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

FEATURES = [
    'T_Faiman', 'T_ambient_mean', 'Wind_speed_mean', 'GHI_lag_30min',
    'POA_diff', 'GHI_diff', 'Hour_sin', 'Hour_cos', 'DoY_sin', 'DoY_cos',
]

def met(yt, yp):
    return (mean_absolute_error(yt, yp),
            np.sqrt(mean_squared_error(yt, yp)),
            r2_score(yt, yp))

def build_mlp():
    m = keras.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return m

def tflite_convert_dynamic(model, path):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    buf = c.convert()
    with open(path, 'wb') as f: f.write(buf)
    return len(buf)

def tflite_convert_int8(model, X_cal, path):
    def rep():
        idx = np.random.RandomState(42).choice(len(X_cal), min(1000, len(X_cal)), replace=False)
        for i in idx:
            yield [X_cal[i:i+1].astype(np.float32)]
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    buf = c.convert()
    with open(path, 'wb') as f: f.write(buf)
    return len(buf)

def tflite_eval(path, X_test, y_test):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = []
    for i in range(len(X_test)):
        interp.set_tensor(inp['index'], X_test[i:i+1].astype(np.float32))
        interp.invoke()
        preds.append(interp.get_tensor(out['index'])[0, 0])
    preds = np.array(preds)
    mae, rmse, r2 = met(y_test, preds)
    # Benchmark
    x = X_test[0:1].astype(np.float32)
    for _ in range(20):
        interp.set_tensor(inp['index'], x)
        interp.invoke()
    t0 = time.perf_counter()
    for _ in range(500):
        interp.set_tensor(inp['index'], x)
        interp.invoke()
    ms = (time.perf_counter() - t0) / 500 * 1000
    return mae, rmse, r2, ms

es = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# ─── Load ───
print('Loading...')
df = pd.read_csv(DATA)
df['timestamp'] = pd.to_datetime(df['timestamp'])
train_df = df[df['year'] == 2022].dropna(subset=FEATURES + ['T_module_mean', 'Residual']).copy()
test_df = df[df['year'] == 2023].dropna(subset=FEATURES + ['T_module_mean', 'Residual']).copy()
y_train = train_df['T_module_mean'].values
y_test = test_df['T_module_mean'].values
y_train_res = train_df['Residual'].values
t_faiman_test = test_df['T_Faiman'].values
print(f'Train: {len(train_df)}, Test: {len(test_df)}')

sc = StandardScaler()
X_tr = sc.fit_transform(train_df[FEATURES].values)
X_te = sc.transform(test_df[FEATURES].values)

rows = []

# ═══ 1. Faiman ═══
print('\n[1] Faiman...')
m, r, r2 = met(y_test, t_faiman_test)
rows.append(('Faiman', 'Physics', m, r, r2, '0', '<0.001'))
print(f'  MAE={m:.3f}')

# ═══ 2. Linear Regression ═══
print('[2] LinearReg...')
lr = LinearRegression().fit(X_tr, y_train)
m, r, r2 = met(y_test, lr.predict(X_te))
rows.append(('LinearReg', 'Direct', m, r, r2, '0.1', '<0.001'))
print(f'  MAE={m:.3f}')

# ═══ 3. Random Forest ═══
print('[3] RF...')
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_train)
m, r, r2 = met(y_test, rf.predict(X_te))
rows.append(('RF', 'Direct', m, r, r2, '~200', 'server'))
print(f'  MAE={m:.3f}')

# ═══ 4. MLP Direct (Keras 64→32→1) ═══
print('[4] MLP Direct...')
mlp_d = build_mlp()
mlp_d.fit(X_tr, y_train, epochs=200, batch_size=256, validation_split=0.1, callbacks=[es, rl], verbose=0)
pred_d = mlp_d.predict(X_te, verbose=0).flatten()
m_d, r_d, r2_d = met(y_test, pred_d)
rows.append(('MLP-F32', 'Direct', m_d, r_d, r2_d, '-', '-'))
print(f'  Float32: MAE={m_d:.3f}, R2={r2_d:.4f}')
mlp_d.save(os.path.join(MODEL_DIR, 'mlp_64_32_direct.keras'))

# TFLite
dyn_path = os.path.join(MODEL_DIR, 'mlp_64_32_dynamic.tflite')
dyn_sz = tflite_convert_dynamic(mlp_d, dyn_path)
m_dyn, r_dyn, r2_dyn, ms_dyn = tflite_eval(dyn_path, X_te, y_test)
rows.append(('MLP-Dynamic', 'Direct', m_dyn, r_dyn, r2_dyn, f'{dyn_sz/1024:.1f}', f'{ms_dyn:.3f}'))
print(f'  Dynamic: MAE={m_dyn:.3f}, Size={dyn_sz/1024:.1f}KB')

int8_path = os.path.join(MODEL_DIR, 'mlp_64_32_int8.tflite')
int8_sz = tflite_convert_int8(mlp_d, X_tr, int8_path)
m_i8, r_i8, r2_i8, ms_i8 = tflite_eval(int8_path, X_te, y_test)
rows.append(('MLP-INT8', 'Direct', m_i8, r_i8, r2_i8, f'{int8_sz/1024:.1f}', f'{ms_i8:.3f}'))
print(f'  INT8: MAE={m_i8:.3f}, Size={int8_sz/1024:.1f}KB, degrad={m_i8-m_d:.3f}')

# ═══ 5. MLP Residual ═══
print('[5] MLP Residual...')
mlp_r = build_mlp()
mlp_r.fit(X_tr, y_train_res, epochs=200, batch_size=256, validation_split=0.1, callbacks=[es, rl], verbose=0)
pred_r = t_faiman_test + mlp_r.predict(X_te, verbose=0).flatten()
m_r, r_r, r2_r = met(y_test, pred_r)
rows.append(('MLP-Residual', 'Residual', m_r, r_r, r2_r, '-', '-'))
print(f'  Residual: MAE={m_r:.3f}')

# ═══ 6. Cross-station ═══
print('[6] Cross-station (WS1+2+3 → WS4)...')
cs_X, cs_y = [], []
for ws in ['T_mod_WS1', 'T_mod_WS2', 'T_mod_WS3']:
    d = df.dropna(subset=FEATURES + [ws])
    cs_X.append(d[FEATURES].values)
    cs_y.append(d[ws].values)
X_cs = np.vstack(cs_X)
y_cs = np.concatenate(cs_y)

ws4 = df.dropna(subset=FEATURES + ['T_mod_WS4'])
X_ws4 = ws4[FEATURES].values
y_ws4 = ws4['T_mod_WS4'].values

sc_cs = StandardScaler()
X_cs_s = sc_cs.fit_transform(X_cs)
X_ws4_s = sc_cs.transform(X_ws4)

mlp_cs = build_mlp()
mlp_cs.fit(X_cs_s, y_cs, epochs=200, batch_size=256, validation_split=0.1, callbacks=[es, rl], verbose=0)
pred_cs = mlp_cs.predict(X_ws4_s, verbose=0).flatten()
m_cs, r_cs, r2_cs = met(y_ws4, pred_cs)
rows.append(('MLP-CrossStation', 'Direct', m_cs, r_cs, r2_cs, '-', '-'))
print(f'  Cross-station: MAE={m_cs:.3f}')

# Same-station WS4
ws4_22 = df[(df['year']==2022)].dropna(subset=FEATURES + ['T_mod_WS4'])
ws4_23 = df[(df['year']==2023)].dropna(subset=FEATURES + ['T_mod_WS4'])
sc_ss = StandardScaler()
X_ss_tr = sc_ss.fit_transform(ws4_22[FEATURES].values)
X_ss_te = sc_ss.transform(ws4_23[FEATURES].values)
mlp_ss = build_mlp()
mlp_ss.fit(X_ss_tr, ws4_22['T_mod_WS4'].values, epochs=200, batch_size=256,
           validation_split=0.1, callbacks=[es, rl], verbose=0)
pred_ss = mlp_ss.predict(X_ss_te, verbose=0).flatten()
m_ss, r_ss, r2_ss = met(ws4_23['T_mod_WS4'].values, pred_ss)
rows.append(('MLP-SameStation', 'Direct', m_ss, r_ss, r2_ss, '-', '-'))
print(f'  Same-station WS4: MAE={m_ss:.3f}')
print(f'  Gap (cross-same): {m_cs - m_ss:.3f}°C')

# ═══ CNN (for paper reference) ═══
print('[7] CNN reference (raw, window=6)...')
FEAT_CNN = ['T_Faiman','T_ambient_mean','GHI_mean','POA_mean','Wind_speed_mean',
            'Hour_sin','Hour_cos','DoY_sin','DoY_cos']
sc_c = StandardScaler()
cnn_tr = sc_c.fit_transform(train_df[FEAT_CNN].values)
cnn_te = sc_c.transform(test_df[FEAT_CNN].values)
def make_win(X, y, w):
    return np.array([X[i-w:i] for i in range(w, len(X))], dtype=np.float32), y[w:]
Xw_tr, yw_tr = make_win(cnn_tr, y_train, 6)
Xw_te, yw_te = make_win(cnn_te, y_test, 6)
cnn = keras.Sequential([
    layers.Input(shape=(6, 9)),
    layers.Conv1D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    layers.Conv1D(16, 3, activation='relu', padding='same'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])
cnn.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
cnn.fit(Xw_tr, yw_tr, epochs=300, batch_size=256, validation_split=0.1, callbacks=[es, rl], verbose=0)
pred_cnn = cnn.predict(Xw_te, verbose=0).flatten()
m_cnn, r_cnn, r2_cnn = met(yw_te, pred_cnn)
rows.append(('CNN-F32', 'Direct', m_cnn, r_cnn, r2_cnn, '66', '-'))
print(f'  CNN: MAE={m_cnn:.3f}')

# ═══ RESULTS ═══
lines = [
    'PHASE 1 DEFINITIVE RESULTS — Keras(64,32)',
    '=' * 74, '',
    f'Architecture: Dense(64) → Dense(32) → Dense(1), ~2,753 params',
    f'Train: 2022 ({len(train_df)}), Test: 2023 ({len(test_df)})',
    f'Features: {FEATURES}', '',
    f'{"Model":<20} {"Mode":<10} {"MAE":>7} {"RMSE":>7} {"R2":>7} {"KB":>7} {"ms":>7}',
    '-' * 74,
]
for name, mode, mae, rmse, r2, sz, inf in rows:
    lines.append(f'{name:<20} {mode:<10} {mae:>7.3f} {rmse:>7.3f} {r2:>7.4f} {str(sz):>7} {str(inf):>7}')

lines.append('')
lines.append('KEY METRICS:')
lines.append(f'  Faiman → MLP improvement: {(1-m_d/met(y_test,t_faiman_test)[0])*100:.1f}%')
lines.append(f'  INT8 degradation: {m_i8-m_d:.3f}°C')
lines.append(f'  Direct vs Residual: {m_d:.3f} vs {m_r:.3f}')
lines.append(f'  MLP+lag vs CNN+raw: {m_d:.3f} vs {m_cnn:.3f}')
lines.append(f'  Cross-station: {m_cs:.3f}, Same-station: {m_ss:.3f}, Gap: {m_cs-m_ss:.3f}°C')

with open(os.path.join(RESULT_DIR, 'phase1_definitive.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'\n✅ {RESULT_DIR}/phase1_definitive.txt')
print('🎉 Phase 1 DEFINITIVE — all models, one architecture!')
