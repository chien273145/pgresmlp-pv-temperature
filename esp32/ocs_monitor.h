// NExT-PINN v4 OCS Monitor — Faiman Residual MLP
// OCS_CO = 32.19C | Phase1: 4.816C | Phase2a: 5.106C
#pragma once
#include <math.h>
#define OCS_CO 32.19f
#define OCS_WARN_DIST 2.5f
#define OCS_CRIT_DIST 1.2f
typedef enum { OCS_OK=0, OCS_WARN=1, OCS_CRIT=2 } OCSStatus;
static float _obuf[12]={0}; static uint8_t _oi=0;
OCSStatus ocs_check(float T) {
    _obuf[_oi++%12]=T;
    float m=0; for(int i=0;i<12;i++) m+=_obuf[i]; m/=12.0f;
    float d=fabsf(m-OCS_CO);
    if(d<OCS_CRIT_DIST) return OCS_CRIT;
    if(d<OCS_WARN_DIST) return OCS_WARN;
    return OCS_OK;
}
