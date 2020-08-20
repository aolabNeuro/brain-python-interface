import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def frNormalization(spk):
    spkz = np.sqrt(spk)
    spk_norm = np.zeros_like(spk)
    for cn in range(spkz.shape[0]):
        psth = spkz[cn, :, :]
        for trn in range(psth.shape[0]):
            # trlSpk = psth[trn, :] - np.nanmean(psth[trn, :])
            trlSpk = stats.mstats.zscore(psth[trn, :], nan_policy='omit')
            # scale to Hz
            spk_norm[cn, trn, :] = trlSpk

            del trlSpk
        del psth
    return spk_norm

    # calculate mean FR in Hz across stepSize time bins (temporal, within trial)
def moveMean_step(stepSize, data):
    starts = np.arange(0, data.shape[2], stepSize)
    FR = np.zeros((data.shape[0], data.shape[1], len(starts)))
    for cn in range(data.shape[0]):
        psth = data[cn, :, :]
        for trn in range(psth.shape[0]):
            fr = np.zeros((data.shape[1], len(starts)))
            for tp in range(len(starts) - 1):
                fr[trn, tp] = np.nanmean(psth[trn, starts[tp]:starts[tp] + stepSize]) / 0.01
                # scale to Hz
            FR[cn, :, :] = fr

        del psth
    return FR