import numpy as np

def rmspe(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    error = 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
    return error