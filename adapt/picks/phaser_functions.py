import logging
import numpy as np
import matplotlib.pyplot as plt
# For PHASE DETECTOR
from obspy.signal.trigger import trigger_onset, classic_sta_lta
#
from scipy.fftpack import fft, fftfreq
from scipy import signal
#
from adapt.processing import normalizeTrace
from adapt.utils import nearest_power_of_two
import adapt.errors as QE

logger = logging.getLogger(__name__)


"""
This module contains all the user-defined functions for testing the phase
(outlier or not).

It the single function returns TRUE, it means that he discovered OUTLIER
or a MISPHASE
"""


# ====================================================================
# ====================================================================
#                           OUTLIERS
# NB: the meta_stat_dict is already extracted. Go for the keys!
# ====================================================================
# ====================================================================

def quake_prediction_delta(proc_trace,
                           raw_trace,
                           picktime,
                           meta_stat_dict,
                           prediction_time_idx=0,
                           threshold=1.0,
                           check_delays_of=None):
    """ Simply returns the value of the ABSOLUTE difference among
        the given phase and the prediction.

        This function has in input BY DEFAULT the pick dict, the event
        object and the metastatdict. The rest is usermade.
        If abs(delta) is major than threshold, the test if failed.

        if `check_delays_of` is specified, the test will fail
        only if the station delay of the given phase (P/S)
    """
    res_out = {}
    try:
        _qt = picktime
        _pt = meta_stat_dict['predicted_picks'][prediction_time_idx][1]
        _delta = np.abs(_qt - _pt)
        res_out['output'] = _delta
    except (KeyError, TypeError):
        # MB: missing predicted pick in pickdict:
        res_out['output'] = None
        res_out['result'] = None
        return res_out

    if check_delays_of:
        # Real Check (only if statDelays)
        if check_delays_of.lower() == "p":
            if _delta >= threshold and meta_stat_dict['p_delay'] > 0.0:
                res_out['result'] = True
            else:
                res_out['result'] = False

        elif check_delays_of.lower() == "s":
            if _delta >= threshold and meta_stat_dict['s_delay'] > 0.0:
                res_out['result'] = True
            else:
                res_out['result'] = False

        else:
            raise QE.InvalidParameter("`check_delays_of` must be either 'P'/'S'"
                                      " and not %r" % check_delays_of)
    else:
        # User want to test it anyway
        if _delta >= threshold:
            res_out['result'] = True
        else:
            res_out['result'] = False
    #
    return res_out


def quake_bait_difference(proc_trace,
                          raw_trace,
                          picktime,
                          meta_stat_dict,
                          bait_idx=0,
                          threshold=1):
    """ Compare the difference among 1 phase pick and one of the
        multipicking associated timepicks(bait). Calculate the absolute
        value of the time-delta: if abs(delta) is major than threshold,
        the test if failed.

        v0.5.1 The BAIT picks are obtained from the station metadata,
               ewhile the pick one from the PickContainer
    """
    res_out = {}
    #
    try:
        _qt = picktime
        _bt = meta_stat_dict['bait_picks'][bait_idx][0]
        _delta = np.abs(_qt - _bt)
        res_out['output'] = _delta
    except (KeyError, TypeError):
        # MB: missing pick in pickdict:
        res_out['output'] = None
        res_out['result'] = None
        return res_out

    # Real Check
    if _delta >= threshold:
        res_out['result'] = True
    else:
        res_out['result'] = False
    #
    return res_out


# # ============= TO BE UPDATED TO THE NEW CALL FROM ShootOUTLIERS
# def bait_prediction_difference(pick_dict, meta_stat_dict, event_obj,
#                                phase_idx=0, threshold=1):
#     """ Compare the difference among 1 phase pick and one of the
#         multipicking associated timepicks(bait). Calculate the absolute
#         value of the time-delta: if abs(delta) is major than threshold,
#         the test if failed.

#         v0.5.1 The BAIT picks are obtained from the station metadata,
#                ewhile the pick one from the PickContainer
#     """
#     res_out = {}
#     #
#     try:
#         _bt = meta_stat_dict['bait_picks'][phase_idx][0]
#         _pt = meta_stat_dict['predicted_picks'][phase_idx][1]
#         _delta = np.abs(_bt - _pt)
#         res_out['output'] = _delta
#     except (KeyError, TypeError):
#         # MB: missing pick in pickdict:
#         res_out['output'] = None
#         res_out['result'] = None
#         return res_out

#     # Real Check
#     if _delta <= threshold:
#         res_out['result'] = True
#     else:
#         res_out['result'] = False
#     #
#     return res_out


# ==== Still to implement
# def multipicking_difference(pick_dict, meta_stat_dict, event_obj,
#                             phase_idx=0, picktag_list=('P1', 'P2'),
#                             threshold=1):
#     """ Compare the final ADAPT picks at different rounds
#         Please note that we need the pick_dict for a single station
#     """
#     if len(picktag_list) != 2:
#         raise QE.InvalidParameter("I need at least 2 picktag to create DELTA")
#     #
#     res_out = {}
#     try:
#         pick_one = pick_dict
#         pick_two = pick_dict

#         res_out['output'] = None
#         res_out['result'] = None

#     except KeyError:
#         # MB: missing pick in pickdict:
#         logger.warning("Pair %s not found in PickDict" % picktag_list)
#         res_out['output'] = None
#         res_out['result'] = None
#     return res_out


# ====================================================================
# ====================================================================
#                           PHASER
# ====================================================================
# ====================================================================

def _dospectrum(wt, startt, endt, nfft_exp=1):
    """ Calculate PSD and return it together with frequency axis
        and most energetic band.
    """
    tr = wt.copy()
    tr = tr.trim(startt, endt)
    # tr.detrend(type='constant')
    tr.taper(max_percentage=0.05)
    #
    if nfft_exp:
        nfft, cutidx = nearest_power_of_two(tr.data.size,
                                            increase_exp=nfft_exp)
    else:
        nfft, cutidx = nearest_power_of_two(tr.data.size)

    # FFT SIGNAL
    tr_fft = fft(tr.data, nfft)
    tr_fft_phase = np.angle(tr_fft)
    tr_psd = (np.abs(tr_fft)**2)/nfft
    fax = fftfreq(int(nfft), d=tr.stats.delta)
    #
    fax = fax[0:cutidx]
    tr_psd = tr_psd[0:cutidx]
    tr_fft = tr_fft[0:cutidx]
    tr_fft_phase = tr_fft_phase[0:cutidx]

    # FIND MAXIMUM
    max_freq_idx = np.where(tr_psd == np.max(tr_psd))[0]
    max_freq_idx = max_freq_idx[0]
    max_freq = fax[max_freq_idx]
    max_psd = tr_psd[max_freq_idx]

    return fax, tr_psd, max_freq, max_psd


def _low_freq_trend(wt, pick_time, tw_noise=0.5, conf=0.95):
    """
    This method evaluate the relation among prom,inent discendent or
    ascendent behavior of the time-series. This will eventually help
    to detect the Gibbs effect, and  therefore avoid mispicks.
    It determines either positive and negative trend (above threshold)

    return True if trend found --> bad pick
    """
    tr = wt.copy()
    tr.trim(pick_time-tw_noise, pick_time)
    asign = np.sign(np.diff(tr.data))
    unique, counts = np.unique(asign, return_counts=True)
    dsign = dict(zip(unique, counts))
    #
    for key in (-1.0, 1.0):
        if key in dsign and dsign[key]:
            pass
        else:
            dsign[key] = 0

    # ------ Out + Log
    pos_trend = dsign[1.0] / len(asign)
    neg_trend = dsign[-1.0] / len(asign)
    if pos_trend >= conf or neg_trend >= conf:
        logger.debug(("TRUE: Pos. %5.2f  -  Neg. %5.2f  [%5.2f]") %
                     (dsign[1.0] / len(asign), dsign[-1.0] / len(asign), conf))
        return True
    else:
        logger.debug(("FALSE: Pos. %5.2f  -  Neg. %5.2f  [%5.2f]") %
                     (dsign[1.0] / len(asign), dsign[-1.0] / len(asign), conf))
        return False


def _triple_amplitude_calc(wt,
                           pick,
                           tw_signal=0.6,
                           tw_noise=0.6,
                           tw_start=0.6,
                           start_buff=0.5,
                           debug_plot=False):
    """ Helper function to calculate 3 different amplitudes:
            - Signal one
            - Noise one
            - Far Noise one
    """
    sig = wt.slice(pick, pick + tw_signal)
    noi = wt.slice(pick-tw_noise, pick)
    sta = wt.slice(wt.stats.starttime + start_buff,
                   wt.stats.starttime+start_buff+tw_start)

    sigmax = np.max(sig.data)
    sigstd = np.std(sig.data)

    noimax = np.max(noi.data)
    noistd = np.std(noi.data)

    stamax = np.max(sta.data)
    stastd = np.std(sta.data)

    logger.debug("MAX SIGNAL: %5.2f - STD SIGNAL: %5.2f" % (sigmax, sigstd))
    logger.debug("MAX NOISE:  %5.2f - STD NOISE: %5.2f" % (noimax, noistd))
    logger.debug("MAX BACKGROUND:  %5.2f - STD BACKGROUND: %5.2f" %
                 (stamax, stastd))
    #
    return sigmax, sigstd, noimax, noistd, stamax, stastd


def _triple_amplitude_calc_longnoise(
                           wt,
                           pick,
                           tw_signal=0.6,
                           tw_noise=0.6,
                           tw_long=0.6,
                           debug_plot=False):
    """ Helper function to calculate 3 different amplitudes:
            - Signal one
            - Noise one
            - Long Noise one
        It's basically 3 different window close to each other
    """
    sig = wt.slice(pick, pick + tw_signal)
    noi = wt.slice(pick-tw_noise, pick)
    sta = wt.slice(pick-tw_noise-tw_long, pick-tw_noise)

    sigmax = np.max(sig.data)
    sigstd = np.std(sig.data)

    noimax = np.max(noi.data)
    noistd = np.std(noi.data)

    stamax = np.max(sta.data)
    stastd = np.std(sta.data)

    logger.debug("MAX SIGNAL: %5.2f - STD SIGNAL: %5.2f" % (sigmax, sigstd))
    logger.debug("MAX NOISE:  %5.2f - STD NOISE: %5.2f" % (noimax, noistd))
    logger.debug("MAX LONG NOISE:  %5.2f - STD LONG NOISE: %5.2f" %
                 (stamax, stastd))
    #
    return sigmax, sigstd, noimax, noistd, stamax, stastd


def _earle_shearer(wt,
                   startt,
                   endt,
                   wsta=0.5,
                   wlta=1.0,
                   trig_on=2.0,
                   trig_off=1.5,
                   debug_plot=False):
    """ Based on the PHASE DETECTOR algorithm from Earle&Shearer_1994.
        This method, will simply detect if one or more seismic phases
        are present in the signal. A simple ENVELOPE and STA/LTA are
        used for this action.

        Return list of IDX_trigger(on-off) and TIME_trigger(on/off)
    """
    tr = wt.copy()
    df = tr.stats.sampling_rate
    tr = tr.trim(startt, endt)
    # Calculate ENVELOPE EarleShearer
    # E(t) = np.sqrt(s(t)**2 + s_hilb(t)**2)
    env = np.sqrt((tr.data**2) + (signal.hilbert(tr.data) ** 2))
    cft = classic_sta_lta(env, int(wsta * df), int(wlta * df))
    on_off_idx = np.array(trigger_onset(cft,
                                        trig_on,
                                        trig_off))
    on_off_time = on_off_idx*(1/df)

    if debug_plot:
        ax1 = plt.subplot(311)
        ax1.plot(tr.times(), tr.data, 'k')

        ax2 = plt.subplot(312)
        ax2.plot(env, 'b')

        ax3 = plt.subplot(313)
        ax3.axhline(trig_on, color='black', linestyle=":", label="trig on")
        ax3.axhline(trig_off, color='grey', linestyle=":", label="trig off")
        for zz in on_off_idx:
            ax3.axvline(zz[0], color='green', label="trig on")
            ax3.axvline(zz[1], label="trig off")
        #
        ax3.plot(cft, 'r')
        plt.show()
    #
    return on_off_time, on_off_idx


def secondary_phase_detection(proc_trace,
                              raw_trace,
                              pick,
                              meta_stat_dict,
                              minimal_distance=None,
                              use_raw=False,
                              tw_signal=0.6,
                              tw_noise=0.6,
                              tw_start=2.0,
                              start_buff=1.0,
                              tw_noise_earleshearer=2.0,           # 5.75
                              tw_noise_buffer_earleshearer=0.0,    # 0.25
                              earleshearer_dict={},
                              thr_earleshearer=0.9,
                              thr_psd=10.0,
                              low_freq_trend_dict={},
                              debug_plot=False):
    """ Comprehensive test for secondary detection
        NB: minimal_distance represent the minimum epicentral distance
            FROM WHICH apply this function.
    """
    res_out = {}

    # Check Minimal Distance:
    if minimal_distance:
        if not meta_stat_dict["epidist"] >= minimal_distance:
            logger.debug(
              ("Skipped! EpiDist less than minimal_distance parameter: %f") %
              minimal_distance)
            res_out['result'] = False
            return res_out  # exit the function with False

    # Select trace
    if use_raw:
        tr = raw_trace.copy()
    else:
        tr = proc_trace.copy()

    # ----- Spectrum (SIGNAL, NOISE, BACKGROUND)
    fax_sig, tr_psd_sig, max_freq_sig, max_psd_sig = _dospectrum(
                                                            tr,
                                                            pick,
                                                            pick+tw_signal,
                                                            nfft_exp=2)
    fax_noi, tr_psd_noi, max_freq_noi, max_psd_noi = _dospectrum(tr,
                                                                 pick-tw_noise,
                                                                 pick,
                                                                 nfft_exp=2)

    RES_FREQ = max_psd_sig/max_psd_noi >= thr_psd

    # ----- Triple Amplitude
    sigmax, sigstd, noimax, noistd, stamax, stastd = _triple_amplitude_calc(
                                                       tr, pick,
                                                       tw_signal=tw_signal,
                                                       tw_noise=tw_noise,
                                                       tw_start=tw_start,
                                                       start_buff=start_buff)
    RES_3AMP = (np.abs(sigmax) >= np.abs(noimax) >= np.abs(stamax) and
                noistd/stastd >= 2.0)

    # ----- EarleShearer
    startt = pick - tw_noise_buffer_earleshearer - tw_noise_earleshearer
    endt = pick - tw_noise_buffer_earleshearer
    on_off_time, on_off_idx = _earle_shearer(tr, startt, endt,
                                             **earleshearer_dict)
    RES_SHEAR = (on_off_time.size != 0 and
                 (on_off_time[-1][1] - on_off_time[-1][0]) >= thr_earleshearer)

    # ----- Filter Effect / GIBBS ==> TRUE if trend found
    RES_TREND = _low_freq_trend(tr, pick, **low_freq_trend_dict)

    if debug_plot:
        SAMELEN = (tr_psd_sig.size == tr_psd_noi.size)
        if SAMELEN:
            ax1 = plt.subplot(311)
            ax2 = plt.subplot(312)
            ax3 = plt.subplot(313)
        else:
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(211)
        #
        ax1.plot(tr.times("matplotlib"), tr.data,
                 color='black', label='original')
        ax1.axvline(pick.datetime, color='gold', label='pick')
        ax1.axvline((pick-tw_noise).datetime, color='blue', linestyle="--")
        ax1.axvline((pick+tw_signal).datetime, color='blue', linestyle="--")
        ax1.axvline((tr.stats.starttime + start_buff).datetime,
                    color='blue', linestyle="--")
        ax1.axvline((tr.stats.starttime + start_buff + tw_start).datetime,
                    color='blue', linestyle="--")
        plt.legend()

        ax2 = plt.subplot(312)
        ax2.loglog(fax_sig, tr_psd_sig, color='black', label='signal')
        ax2.loglog(fax_noi, tr_psd_noi, color='red', label='noise')
        # ax2.loglog(fax_sta, tr_psd_sta, color='blue', label='background')
        plt.grid()
        plt.legend()

        if SAMELEN:
            ax3 = plt.subplot(313, sharex=ax2)
            ax3.loglog(fax_sig, tr_psd_sig/tr_psd_noi,
                       color='black', label='signal PSD / noise PSD')
            plt.grid()
            plt.legend()

        ax1.set_title(" - ".join([tr.stats.station,
                                  "Freq: "+str(RES_FREQ),
                                  "3Ampl: "+str(RES_3AMP),
                                  "EarleShearer: "+str(RES_SHEAR),
                                  "Trend: "+str(RES_TREND)]))

        # logger.debug(
        #    'BACKGROUND FREQ: %4.2f - NOISE FREQ: %4.2f - SIGNAL FREQ: %4.2f' %
        #    (max_freq_sta, max_freq_noi, max_freq_sig))
        logger.debug(
           'PSD SIGNAL/NOISE RATION: %4.2f' % (max_psd_sig/max_psd_noi))
        # logger.debug(
        #    'MAXFREQ NOISE/BACKGROUND: %4.2f' % (max_freq_noi/max_freq_sta))
        plt.show()

    # ==================================================  FINAL DECISION
    # ==================================================
    res_out['output'] = (max_freq_noi >= max_freq_sig,
                         RES_FREQ, RES_3AMP, RES_SHEAR, RES_TREND)
    if ((max_freq_noi >= max_freq_sig and               # Freq DECREASE
         RES_FREQ and RES_3AMP and not RES_TREND) or
        (max_freq_noi < max_freq_sig and                # Freq INCREASE
         RES_FREQ and RES_3AMP and RES_SHEAR and not RES_TREND)):
        res_out['result'] = True
    else:
        res_out['result'] = False
    #
    return res_out
