"""Physical constants and defaults for the PHARE load estimator."""

from dataclasses import dataclass

Re_km = 6371.2  # Earth radius in km

# A PHARE-like particle: 3 doubles position + 3 doubles velocity + 1 charge.
BYTES_PER_PARTICLE = 7 * 8  # 56 bytes
PPC = 100  # particles per cell

# Cost model: aggregate single-thread time per particle per timestep.
SEC_PER_PARTICLE_PER_STEP = 10e-9  # 10 ns

# Time-step: dt = 1e-3 / Omega_ci, with Omega_ci ~ 1 rad/s in the magnetosheath.
DT_SECONDS = 1e-3

# Physical run target.
TARGET_RUN_HOURS = 3.0

# Proton mass in kg.
M_PROTON_KG = 1.67262192e-27


@dataclass(frozen=True)
class SolarWind:
    n_cm3: float       # number density, cm^-3
    V_kms: float       # bulk speed, km/s
    Bz_nT: float       # IMF Bz, nT

    @property
    def Pdyn_nPa(self) -> float:
        # Pd [nPa] = m_p * n * V^2 ; with n in cm^-3 and V in km/s,
        # the convenient formula is 1.6726e-6 * n * V^2  (gives nPa).
        return 1.6726e-6 * self.n_cm3 * self.V_kms ** 2


NOMINAL_SW = SolarWind(n_cm3=5.0, V_kms=400.0, Bz_nT=0.0)
