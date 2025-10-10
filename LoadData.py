import numpy as np
import scipy.io as sio
from scipy.integrate import simpson


# ===== Config =====
WORK_DIR = '/Users/pengzehao/Desktop/Multimodal/Iyer_etal_2020_SFH_data/'
EXTN = '_sfhs_psds.mat'

# Keep your 139-bin grid:
TARGET_X = np.linspace(0.1, 13.6, 139)

# If you prefer exact dx=0.1 spacing, uncomment the next line (→ 136 bins):
# TARGET_X = np.arange(0.1, 13.6 + 1e-9, 0.1)

DX_FOR_INTEGRATED_NORM = 0.1
TARGET_AREA = 500.0

def _mass_threshold(sim_name: str) -> float:
    return 10.0 if sim_name in ('Simba', 'Mufasa') else 9.0

def _mass_cut_indices(logmass: np.ndarray, sim_name: str) -> np.ndarray:
    thr = _mass_threshold(sim_name)
    return np.where(logmass > thr)[0]

def _nonzero_mask_trapz(sfh_array: np.ndarray, time: np.ndarray) -> np.ndarray:
    areas = np.array([np.trapz(sfh, time) for sfh in sfh_array])
    return areas != 0.0

def _interp_batch(sfh_array: np.ndarray, from_time: np.ndarray, to_time: np.ndarray) -> np.ndarray:
    out = np.empty((sfh_array.shape[0], to_time.size), dtype=float)
    for i, row in enumerate(sfh_array):
        out[i] = np.interp(to_time, from_time, row)
    return out

def _normalize_trapz_dx_to_area(sfh_array_on_uniform: np.ndarray,
                                dx: float = DX_FOR_INTEGRATED_NORM,
                                target_area: float = TARGET_AREA) -> np.ndarray:
    """
    Match normalize_sfh(..., method='integrated'): divide by np.trapz(..., dx=dx, axis=1),
    then scale so the total integral equals target_area.
    """
    denom = np.trapz(sfh_array_on_uniform, dx=dx, axis=1)  # shape (N,)
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    return (sfh_array_on_uniform / denom_safe[:, None]) * target_area

# ===== Main API =====
def loaddata(simulation_names: list):
    """
    Order: (1) SM cuts, (2) delete zeros, (3) interpolate, (4) normalize to area=500
    with np.trapz(..., dx=0.1, axis=1)—matching your normalize_sfh() semantics.
    """
    combined = []

    for sim_name in simulation_names:
        mat = sio.loadmat(WORK_DIR + sim_name + EXTN)
        smalltime = mat['smalltime'][0]
        raw_sfhs = mat['smallsfhs'].T
        logmass = np.array(mat['logmass'].ravel())

        # (1) SM cuts
        keep_idx = _mass_cut_indices(logmass, sim_name)
        raw_sfhs = raw_sfhs[keep_idx]
        logmass = logmass[keep_idx]

        # present-day SFR from the *raw* series (pre-norm)
        presentsfr_raw = raw_sfhs[:, -1]

        # (2) delete 0 SFHs on native grid
        nz_mask = _nonzero_mask_trapz(raw_sfhs, smalltime)
        raw_sfhs, logmass, presentsfr_raw = raw_sfhs[nz_mask], logmass[nz_mask], presentsfr_raw[nz_mask]

        # (3) interpolate to uniform grid
        ah = _interp_batch(raw_sfhs, smalltime, TARGET_X)

        # (4) normalize by integrated area (dx=0.1) → 500 total
        ah = _normalize_trapz_dx_to_area(ah, dx=DX_FOR_INTEGRATED_NORM, target_area=TARGET_AREA)

        for arr, m, s in zip(ah, logmass, presentsfr_raw):
            combined.append([arr, [m, np.arcsinh(s)], sim_name])

    return combined

def filter_zeroes(inputHistories: np.ndarray, mass_presentsfr: np.ndarray, labels: np.ndarray):
    """
    Filter out galaxies with zero SFH *on the current grid*.
    """
    zero_indices = np.array([i for i in range(len(inputHistories)) if np.trapz(inputHistories[i]) == 0])
    if zero_indices.size == 0:
        return inputHistories, mass_presentsfr, labels

    mask = np.ones(inputHistories.shape[0], dtype=bool)
    mask[zero_indices] = False
    return inputHistories[mask], mass_presentsfr[mask], labels[mask]

def count_zeroes(inputHistories: np.ndarray, mass_presentsfr: np.ndarray, labels: np.ndarray):
    """
    Count number of galaxies with zero SFH by simulation label (string labels expected).
    """
    zero_indices = np.array([i for i in range(len(inputHistories)) if np.trapz(inputHistories[i]) == 0])
    mask = np.ones(inputHistories.shape[0], dtype=bool)
    mask[zero_indices] = False

    print('Total number of galaxies:', len(inputHistories))
    print('Number of galaxies with zero SFH in each simulation:')
    for sim in sorted(set(labels.tolist())):
        print(sim, np.sum(labels == sim) - np.sum(labels[mask] == sim))
    print('Total number of galaxies with zero SFH:', len(zero_indices))

def filter_data(data: list):
    """
    Convert loaddata() output into arrays and numeric labels, and remove any residual zeros.
    NOTE: SFHs are already Simpson-normalized (area≈500) and interpolated by loaddata().
    """
    inputHistories = np.array([element[0] for element in data])          # (N, 139)
    mass_presentsfr = np.array([element[1] for element in data])         # (N, 2) -> [logmass, arcsinh(present_sfr_raw)]
    label_names = np.array([element[2] for element in data])             # (N,) strings

    # Safety pass (idempotent): remove zeros again if any slipped through
    filtered_SFH, filtered_mass_presentsfr, filtered_label_names = filter_zeroes(
        inputHistories, mass_presentsfr, label_names
    )

    # map labels to numeric ids (original mapping)
    name2id = {
        'Eagle': 0,
        'Mufasa': 1,
        'SC-Sam': 2,
        'Simba': 3,
        'Illustris': 4,
        'IllustrisTNG': 5,
        'UniverseMachine': 6,
    }
    filtered_labels = np.array([name2id[name] for name in filtered_label_names], dtype='float32')

    # Already normalized to AHs (area~500) on the uniform grid
    filtered_AH = filtered_SFH

    return filtered_AH, filtered_mass_presentsfr, filtered_labels

def un_one_hot(one_hot_array: np.ndarray):
    """
    Un-one hot encodes an array.
    """
    return np.argmax(one_hot_array, axis=-1)
