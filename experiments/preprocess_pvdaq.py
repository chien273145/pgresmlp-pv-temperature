"""
PVDAQ System 7333 — Data Preprocessing Pipeline v2
Incorporates advisor feedback:
  1. Lag features (thermal mass/hysteresis)
  2. Cyclical time encoding (sin/cos)
  3. Faiman physics baseline + residual target
  4. Drop DNI/DHI (multicollinearity)
  5. Cross-station analysis ready
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'c:\Users\admin\NCKH'
OUTPUT_DIR = r'c:\Users\admin\NCKH\processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Column Mappings
# =============================================================================
ENV_RENAME = {
    'utc_measured_on': 'timestamp',
    # Module Temperature — TARGET
    'sos-02-012-ws2-mod-t-c__175559': 'T_mod_WS2',
    'sos-02-041-ws1-mod-t-c__175565': 'T_mod_WS1',
    'sos-03-106-ws4-mod-t-c__175571': 'T_mod_WS4',
    'sos-06-083-ws3-mod-t-c__175577': 'T_mod_WS3',
    # Ambient Temperature
    'sos-02-012-ws2-amb-t-c__175557': 'T_amb_WS2',
    'sos-02-041-ws1-amb-t-c__175563': 'T_amb_WS1',
    'sos-03-106-ws4-amb-t-c__175569': 'T_amb_WS4',
    'sos-06-083-ws3-amb-t-c__175575': 'T_amb_WS3',
    # Wind Speed
    'sos-02-012-ws2-wind-spd__175562': 'Wind_spd_WS2',
    'sos-02-041-ws1-wind-spd__175568': 'Wind_spd_WS1',
    'sos-03-106-ws4-wind-spd__175574': 'Wind_spd_WS4',
    'sos-06-083-ws3-wind-spd__175580': 'Wind_spd_WS3',
    # Wind Direction
    'sos-02-012-ws2-wind-dir__175561': 'Wind_dir_WS2',
    'sos-02-041-ws1-wind-dir__175567': 'Wind_dir_WS1',
    'sos-03-106-ws4-wind-dir__175573': 'Wind_dir_WS4',
    'sos-06-083-ws3-wind-dir__175579': 'Wind_dir_WS3',
}

IRR_RENAME = {
    'utc_measured_on': 'timestamp',
    # GHI
    'sos-02-012-ws2-ghi-irrad__175558': 'GHI_WS2',
    'sos-02-041-ws1-ghi-irrad__175564': 'GHI_WS1',
    'sos-03-106-ws4-ghi-irrad__175570': 'GHI_WS4',
    'sos-06-083-ws3-ghi-irrad__175576': 'GHI_WS3',
    # POA
    'sos-02-012-ws2-poa-irrad__175560': 'POA_WS2',
    'sos-02-041-ws1-poa-irrad__175566': 'POA_WS1',
    'sos-03-106-ws4-poa-irrad__175572': 'POA_WS4',
    'sos-06-083-ws3-poa-irrad__175578': 'POA_WS3',
}

def load_and_rename(env_path, irr_path, year_label):
    """Load environment + irradiance, rename, merge on timestamp."""
    print(f'\n[{year_label}] Loading environment...')
    env = pd.read_csv(env_path, usecols=list(ENV_RENAME.keys()))
    env.rename(columns=ENV_RENAME, inplace=True)
    print(f'  {len(env)} rows, {len(env.columns)} columns')

    print(f'[{year_label}] Loading irradiance...')
    irr = pd.read_csv(irr_path, usecols=list(IRR_RENAME.keys()))
    irr.rename(columns=IRR_RENAME, inplace=True)
    print(f'  {len(irr)} rows, {len(irr.columns)} columns')

    print(f'[{year_label}] Merging...')
    df = pd.merge(env, irr, on='timestamp', how='inner')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    print(f'  Merged: {len(df)} rows')
    return df


# =============================================================================
# STEP 1: Load & Merge
# =============================================================================
print('=' * 70)
print('STEP 1: LOAD & MERGE')
print('=' * 70)

df_2022 = load_and_rename(
    os.path.join(DATA_DIR, '7333_5_min_environment_20220101_20221231.csv'),
    os.path.join(DATA_DIR, '7333_5_min_irradiance_20220101_20221231.csv'), '2022')
df_2023 = load_and_rename(
    os.path.join(DATA_DIR, '7333_5_min_environment_20230101_20231231.csv'),
    os.path.join(DATA_DIR, '7333_5_min_irradiance_20230101_20231231.csv'), '2023')

df = pd.concat([df_2022, df_2023], ignore_index=True)
df.sort_values('timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)
print(f'\nCombined: {len(df)} rows')

# =============================================================================
# STEP 2: Averaged Features
# =============================================================================
print('\n' + '=' * 70)
print('STEP 2: AVERAGED FEATURES')
print('=' * 70)

df['T_module_mean'] = df[['T_mod_WS1', 'T_mod_WS2', 'T_mod_WS3', 'T_mod_WS4']].mean(axis=1)
df['T_ambient_mean'] = df[['T_amb_WS1', 'T_amb_WS2', 'T_amb_WS3', 'T_amb_WS4']].mean(axis=1)
df['GHI_mean'] = df[['GHI_WS1', 'GHI_WS2', 'GHI_WS3', 'GHI_WS4']].mean(axis=1)
df['POA_mean'] = df[['POA_WS1', 'POA_WS2', 'POA_WS3', 'POA_WS4']].mean(axis=1)
df['Wind_speed_mean'] = df[['Wind_spd_WS1', 'Wind_spd_WS2', 'Wind_spd_WS3', 'Wind_spd_WS4']].mean(axis=1)

# Time features (raw — for cyclical encoding later)
df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
df['month'] = df['timestamp'].dt.month
df['day_of_year'] = df['timestamp'].dt.dayofyear

print('Created averaged features')

# =============================================================================
# STEP 3: Filter Nighttime + Handle Missing
# =============================================================================
print('\n' + '=' * 70)
print('STEP 3: FILTER NIGHTTIME + HANDLE MISSING')
print('=' * 70)

print(f'Before: {len(df)} rows, T_module NaN: {df["T_module_mean"].isna().sum()}')

# Daytime filter (GHI > 10 W/m²)
df_day = df[df['GHI_mean'] > 10].copy()
print(f'After GHI>10: {len(df_day)} rows')

# Drop rows missing target or key inputs
df_clean = df_day.dropna(subset=['T_module_mean', 'T_ambient_mean'])
print(f'After dropping NaN target/T_amb: {len(df_clean)} rows')

# Forward/backward fill for remaining features
for col in ['GHI_mean', 'POA_mean', 'Wind_speed_mean']:
    before = df_clean[col].isna().sum()
    df_clean.loc[:, col] = df_clean[col].ffill().bfill()
    after = df_clean[col].isna().sum()
    if before > 0:
        print(f'  {col}: {before} NaN -> {after}')

df_clean = df_clean.dropna(subset=['GHI_mean', 'POA_mean', 'Wind_speed_mean'])
print(f'After all fills: {len(df_clean)} rows')

# =============================================================================
# STEP 4: Outlier Clipping
# =============================================================================
print('\n' + '=' * 70)
print('STEP 4: OUTLIER CLIPPING')
print('=' * 70)

clips = {
    'T_module_mean': (-20, 85),
    'T_ambient_mean': (-30, 55),
    'GHI_mean': (0, 1400),
    'POA_mean': (0, 1500),
    'Wind_speed_mean': (0, 40),
}
for col, (lo, hi) in clips.items():
    before = len(df_clean)
    df_clean = df_clean[(df_clean[col] >= lo) & (df_clean[col] <= hi)]
    removed = before - len(df_clean)
    if removed > 0:
        print(f'  {col}: removed {removed} outliers')
print(f'After clipping: {len(df_clean)} rows')

# =============================================================================
# STEP 5: NEW FEATURES (Advisor Feedback)
# =============================================================================
print('\n' + '=' * 70)
print('STEP 5: ADVISOR-RECOMMENDED FEATURES')
print('=' * 70)

# --- 5a. Lag Features (Thermal Mass / Hysteresis) ---
# PV module ~25kg → 15-30 min thermal inertia
print('Creating lag features (rolling means)...')
df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)

# 15-min rolling mean (3 × 5-min steps)
df_clean['POA_lag_15min'] = df_clean['POA_mean'].rolling(window=3, min_periods=1).mean()
df_clean['GHI_lag_15min'] = df_clean['GHI_mean'].rolling(window=3, min_periods=1).mean()
df_clean['Wind_lag_15min'] = df_clean['Wind_speed_mean'].rolling(window=3, min_periods=1).mean()
df_clean['T_amb_lag_15min'] = df_clean['T_ambient_mean'].rolling(window=3, min_periods=1).mean()

# 30-min rolling mean (6 × 5-min steps)
df_clean['POA_lag_30min'] = df_clean['POA_mean'].rolling(window=6, min_periods=1).mean()
df_clean['GHI_lag_30min'] = df_clean['GHI_mean'].rolling(window=6, min_periods=1).mean()

# Rate of change (derivative proxy)
df_clean['POA_diff'] = df_clean['POA_mean'].diff().fillna(0)
df_clean['GHI_diff'] = df_clean['GHI_mean'].diff().fillna(0)

print('  Created: POA_lag_15min, POA_lag_30min, GHI_lag_15min, GHI_lag_30min')
print('  Created: Wind_lag_15min, T_amb_lag_15min')
print('  Created: POA_diff, GHI_diff (rate of change)')

# --- 5b. Cyclical Time Encoding ---
print('Creating cyclical time features...')
df_clean['Hour_sin'] = np.sin(2 * np.pi * df_clean['hour'] / 24)
df_clean['Hour_cos'] = np.cos(2 * np.pi * df_clean['hour'] / 24)
df_clean['Month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
df_clean['Month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)
df_clean['DoY_sin'] = np.sin(2 * np.pi * df_clean['day_of_year'] / 365)
df_clean['DoY_cos'] = np.cos(2 * np.pi * df_clean['day_of_year'] / 365)
print('  Created: Hour_sin, Hour_cos, Month_sin, Month_cos, DoY_sin, DoY_cos')

# --- 5c. Faiman Physics Baseline ---
print('Computing Faiman model baseline...')
# Faiman equation: T_mod = T_amb + POA / (U0 + U1 × Wind)
U0 = 25.0   # W/(m²·K), free convection heat transfer coefficient
U1 = 6.84   # W·s/(m³·K), forced convection coefficient
# Avoid division by zero: clip wind to minimum 0.1 m/s
wind_safe = df_clean['Wind_speed_mean'].clip(lower=0.1)
df_clean['T_Faiman'] = df_clean['T_ambient_mean'] + df_clean['POA_mean'] / (U0 + U1 * wind_safe)

# Residual: what physics model gets wrong → AI's job
df_clean['Residual'] = df_clean['T_module_mean'] - df_clean['T_Faiman']

faiman_mae = np.abs(df_clean['Residual']).mean()
faiman_rmse = np.sqrt((df_clean['Residual'] ** 2).mean())
print(f'  Faiman baseline: MAE = {faiman_mae:.2f}°C, RMSE = {faiman_rmse:.2f}°C')
print(f'  Residual stats: mean={df_clean["Residual"].mean():.2f}, std={df_clean["Residual"].std():.2f}')

# --- 5d. Delta T ---
df_clean['Delta_T'] = df_clean['T_module_mean'] - df_clean['T_ambient_mean']

# =============================================================================
# STEP 6: Final Column Selection & Export
# =============================================================================
print('\n' + '=' * 70)
print('STEP 6: EXPORT')
print('=' * 70)

# Core model columns (no DNI/DHI — multicollinearity)
model_cols = [
    'timestamp', 'year',
    # Target
    'T_module_mean',
    # Physics target
    'T_Faiman', 'Residual',
    # Current features
    'T_ambient_mean', 'GHI_mean', 'POA_mean', 'Wind_speed_mean',
    # Lag features (thermal mass)
    'POA_lag_15min', 'POA_lag_30min', 'GHI_lag_15min', 'GHI_lag_30min',
    'Wind_lag_15min', 'T_amb_lag_15min',
    # Rate of change
    'POA_diff', 'GHI_diff',
    # Cyclical time
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DoY_sin', 'DoY_cos',
    # Metadata
    'Delta_T', 'hour', 'month', 'day_of_year',
    # Per-station targets (cross-station experiments)
    'T_mod_WS1', 'T_mod_WS2', 'T_mod_WS3', 'T_mod_WS4',
]

df_final = df_clean[model_cols].copy()
df_final.reset_index(drop=True, inplace=True)

print(f'Final shape: {df_final.shape}')
print(f'\nPer-year:')
for y, g in df_final.groupby('year'):
    print(f'  {y}: {len(g)} rows')

# Key stats
print(f'\nKey statistics:')
for col in ['T_module_mean', 'T_ambient_mean', 'GHI_mean', 'POA_mean', 
            'Wind_speed_mean', 'T_Faiman', 'Residual', 'Delta_T']:
    s = df_final[col]
    print(f'  {col:20s}: mean={s.mean():8.2f}, std={s.std():7.2f}, min={s.min():8.2f}, max={s.max():7.2f}')

# Missing check
core_features = [c for c in model_cols if c not in ['timestamp', 'year', 'hour', 'month', 'day_of_year',
                 'T_mod_WS1', 'T_mod_WS2', 'T_mod_WS3', 'T_mod_WS4']]
missing = df_final[core_features].isnull().sum()
missing_any = missing[missing > 0]
if len(missing_any) == 0:
    print('\n✅ No missing values in core features!')
else:
    print(f'\n⚠️ Missing values:')
    for col, n in missing_any.items():
        print(f'  {col}: {n} ({n/len(df_final)*100:.2f}%)')

# Save
output_path = os.path.join(OUTPUT_DIR, 'pvdaq_7333_v2_2022_2023.csv')
df_final.to_csv(output_path, index=False)
print(f'\n✅ Saved: {output_path}')
print(f'   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB')

# Feature list for model training reference
feature_list = [
    'T_ambient_mean', 'GHI_mean', 'POA_mean', 'Wind_speed_mean',
    'POA_lag_15min', 'POA_lag_30min', 'GHI_lag_15min', 'GHI_lag_30min',
    'Wind_lag_15min', 'T_amb_lag_15min',
    'POA_diff', 'GHI_diff',
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DoY_sin', 'DoY_cos',
    'T_Faiman',
]
print(f'\n📋 Feature list for training ({len(feature_list)} features):')
for i, f in enumerate(feature_list):
    print(f'  {i+1:2d}. {f}')

print('\n🎉 Preprocessing v2 complete!')
