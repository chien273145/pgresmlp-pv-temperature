"""Extract weights from Keras MLP(32,16) → C header for ESP32 pure inference."""
import numpy as np, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

FEATURES = ['T_Faiman','T_ambient_mean','Wind_speed_mean','GHI_lag_30min']
DATA = r'c:\Users\admin\NCKH\processed\pvdaq_7333_v2_2022_2023.csv'
ESP32_DIR = r'c:\Users\admin\NCKH\esp32\esp32_inference'

df = pd.read_csv(DATA)
train = df[df['year']==2022].dropna(subset=FEATURES+['T_module_mean'])
test = df[df['year']==2023].dropna(subset=FEATURES+['T_module_mean'])
sc = StandardScaler()
Xtr = sc.fit_transform(train[FEATURES].values)
Xte = sc.transform(test[FEATURES].values)
y_tr, y_te = train['T_module_mean'].values, test['T_module_mean'].values

tf.random.set_seed(42); np.random.seed(42)
m = keras.Sequential([layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'), layers.Dense(16, activation='relu'), layers.Dense(1)])
m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
m.fit(Xtr, y_tr, epochs=200, batch_size=256, validation_split=0.1,
      callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                 callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)], verbose=0)

pred = m.predict(Xte, verbose=0).flatten()
print(f"Keras MAE: {mean_absolute_error(y_te, pred):.4f}")

w1, b1 = m.layers[0].get_weights()
w2, b2 = m.layers[1].get_weights()
w3, b3 = m.layers[2].get_weights()
total = w1.size+b1.size+w2.size+b2.size+w3.size+b3.size
print(f"Params: {total} ({total*4} bytes)")

# Verify
h1 = np.maximum(0, Xte @ w1 + b1)
h2 = np.maximum(0, h1 @ w2 + b2)
pred_c = (h2 @ w3 + b3).flatten()
print(f"C-sim MAE: {mean_absolute_error(y_te, pred_c):.4f}, max_diff: {np.max(np.abs(pred-pred_c)):.10f}")

def arr_c(name, arr):
    f = arr.flatten()
    s = [f'const float {name}[{len(f)}] = {{']
    for i in range(0, len(f), 6):
        s.append('  ' + ', '.join(f'{v:.8f}f' for v in f[i:i+6]) + ',')
    s.append('};'); return '\n'.join(s)

h = ['// MLP [4->32->16->1] weights (float32)', f'// Params: {total} ({total*4} bytes)',
     '#ifndef MODEL_WEIGHTS_H', '#define MODEL_WEIGHTS_H', '',
     '#define N_FEATURES 4', '#define H1 32', '#define H2 16', '',
     '// Scaler',
     'const float sc_mean[4] = {' + ', '.join(f'{v:.8f}f' for v in sc.mean_) + '};',
     'const float sc_scale[4] = {' + ', '.join(f'{v:.8f}f' for v in sc.scale_) + '};',
     '', arr_c('W1', w1), '', arr_c('B1', b1), '', arr_c('W2', w2), '',
     arr_c('B2', b2), '', arr_c('W3', w3), '', arr_c('B3', b3), '',
     '#endif']

p = os.path.join(ESP32_DIR, 'model_weights.h')
with open(p, 'w') as f: f.write('\n'.join(h))
print(f"-> {p} ({os.path.getsize(p)} bytes)")
