import numpy as np
import pickle
from scipy.interpolate import UnivariateSpline
from deepreservoir.data.metadata import project_metadata

m = project_metadata()
niip_pickle = m.path("niip_demand_spline_pkl")


with open(niip_pickle, "rb") as f:
    spline_data = pickle.load(f)

spline = UnivariateSpline._from_tck((spline_data["t"], spline_data["c"], spline_data["k"]))

def niip_daily_demand(doy):
    # Computes daily demand in cfs
    doy = np.asarray(doy)
    raw = np.where((doy >= 50) & (doy <= 300), spline(doy), 0.0)
    return np.maximum(raw, 0.0)
