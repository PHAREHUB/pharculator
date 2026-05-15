"""Physical constants and defaults for the PHARE load estimator."""

from dataclasses import dataclass

Re_km = 6371.2  # Earth radius in km

# Ion inertial length used as the resolution unit.
DELTA_I_KM = 100.0

# Mesh resolutions per level, in units of delta_i.
L0_DX_KM = 0.8 * DELTA_I_KM   # 80 km — MHD, full domain, no particles
L1_DX_KM = 0.8 * DELTA_I_KM   # 80 km — PIC, sheath shell with 3 Re buffers
L2_DX_KM = 0.4 * DELTA_I_KM   # 40 km — PIC, 1.5 Re bands around MP and BS
L3_DX_KM = 0.2 * DELTA_I_KM   # 20 km — PIC, 0.5 Re bands around MP and BS
REFERENCE_UNIFORM_DX_KM = 0.2 * DELTA_I_KM   # 20 km PIC over the whole box

# A PHARE-like particle: 3 doubles position + 3 doubles velocity + 1 charge.
BYTES_PER_PARTICLE = 7 * 8  # 56 bytes
PPC = 100  # particles per cell

# Cost model: aggregate single-thread time per particle per timestep.
SEC_PER_PARTICLE_PER_STEP = 10e-9  # 10 ns

# L3 (the finest level) and the reference uniform run use the base time step.
# Each coarser AMR level fires every 4 steps of the next finer one
# (dt scales as dx**2 with the user's convention).
DT_SECONDS = 1e-3            # L3 / uniform time step
STEPS_PER_L3 = {
    "L0": 1.0 / 64.0,
    "L1": 1.0 / 16.0,
    "L2": 1.0 / 4.0,
    "L3": 1.0,
    "uniform": 1.0,
}

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
