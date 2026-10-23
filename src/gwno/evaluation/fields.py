import numpy as np

def field_metrics(prediction, truth, active_mask, scale, late_steps=10):
    p, y = np.asarray(prediction)[1:,active_mask], np.asarray(truth)[1:,active_mask]
    error = p-y
    variance = np.sum((y-y.mean())**2)
    rmse = float(np.sqrt(np.mean(error**2)))
    return dict(rmse=rmse, mae=float(np.mean(abs(error))), nrmse=rmse/scale,
                r2=None if variance < 1e-12 else float(1-np.sum(error**2)/variance),
                late_nrmse=float(np.sqrt(np.mean(error[-late_steps:]**2))/scale),
                negative_fraction=float(np.mean(p<0)),
                time_rmse=np.sqrt(np.mean(error**2, axis=1)).tolist())
