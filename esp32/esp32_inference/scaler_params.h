// StandardScaler parameters (fit on 2022 train set)
// Features: T_Faiman, T_ambient_mean, Wind_speed_mean, GHI_lag_30min
#ifndef SCALER_PARAMS_H
#define SCALER_PARAMS_H

const int N_FEATURES = 4;

// scaler.mean_
const float scaler_mean[] = {
  37.97368321f, 22.82722709f, 2.79634884f, 489.59075894f
};

// scaler.scale_ (std)
const float scaler_scale[] = {
  14.22118219f, 9.48768756f, 1.60827901f, 296.81553273f
};

// Apply: x_scaled[i] = (x_raw[i] - scaler_mean[i]) / scaler_scale[i]

#endif // SCALER_PARAMS_H