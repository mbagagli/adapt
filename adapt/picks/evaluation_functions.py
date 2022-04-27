import numpy as np
import matplotlib.pyplot as plt
from adapt.processing import normalizeTrace
from adapt import errors as QE

"""ADAPT evaluation functions module.

In this module are defined all the possible functions used by
adapt.picks.evaluation.Gandalf class to evaluate a phase pick over
a given trace.
The user could append additional test functions in this module and
call them with the adapt.picks.evaluation.Gandalf class. Each functions
needs to be called with the correct string key in Gandalf object.

Note:
    All the functions need 3 positional values:
        - a _processed_ obspy.core.Trace instance
        - a _raw_ obspy.core.Trace instance
        - a _picktime_ obspy.core.UTCDateTime instance

    The others parameters are keyword arguments with setted defaults,
    they mostly vary from function to function.

    All the functions returns a dictionary containing the 2 mandatory keys
        - _"result"_ containing the boolean result of the test
        - _"output"_ containing all the necessary instances of the test.
            Check the single function helper for a full description.

"""


def max_signal2noise_ratio(proc_trace,
                           raw_trace,
                           utc,
                           noise_window=0.1,
                           signal_window=0.1,
                           threshold=1.01,
                           use_raw=False,
                           normalize=[],
                           debug_plot=False):
    """Calculate the MAX ratio of a custom signal2noise windows

    If the ratio of the MAX signal window and the MAX noise window
    is higher or equal the given threshold, the test is passed.

    Args:
        proc_trace (obspy.core.Trace): the processed work trace
        raw_trace (obspy.core.Trace): the raw input work trace
        utc (obspy.core.UTCDateTime): the test's picktime associated

        noise_window (:obj:`float`, optional): the amount of seconds
            to be used for trim the noise window around utc.
            Default to 0.1.
        signal_window (:obj:`float`, optional): the amount of seconds
            to be used for trim the signal window around utc.
            Default to 0.1.
        threshold (:obj:`float`, optional): the threshold value for the
            mean signal2noise ratio. Default to 0.2.
        use_raw (:obj:`bool`, optional): if set to True, the raw trace
            is used for the test instead. Default to False.
        normalize (:obj:`list`, optional): if a list is given, the trace
            is normalized based on the given values [min, max] and will
            affect the entire trace given in input. Default to None.

    Returns:
        outdict (dict): this dictionary contains the 'result' boolean
            key and 'output' key containing the calculated ratio value

    Note:
        This method works with the absolute values of the trace

    """
    outdict = {}
    # Selection
    if use_raw:
        wt = raw_trace.copy()
    else:
        wt = proc_trace.copy()
    # Processing and Selection
    wt.data = np.abs(wt.data)
    if normalize:
        if isinstance(normalize, list):
            wt.data = normalizeTrace(wt.data, rangeVal=normalize)
        else:
            raise QE.InvalidParameter()
    noise = wt.slice(utc - noise_window, utc)
    signal = wt.slice(utc, utc + signal_window)
    #
    _check_val = np.max(signal.data)/np.max(noise.data)
    if _check_val >= threshold:
        outdict['result'] = True
        outdict['output'] = _check_val
    else:
        outdict['result'] = False
        outdict['output'] = _check_val
    #
    if debug_plot:
        xt = wt.times(reftime=utc)
        plt.plot(xt, wt.data)
        plt.axvline(utc - utc,
                    color="gold",
                    linewidth=2,
                    linestyle='solid',
                    label="pick")
        plt.axvline(noise.stats.starttime - utc,
                    color="grey",
                    linewidth=1,
                    linestyle=':')
        plt.axvline(signal.stats.endtime - utc,
                    color="grey",
                    linewidth=1,
                    linestyle=':')
        #
        _idxn = np.where(wt.data == np.max(noise.data))[0][0]
        _idxs = np.where(signal.data == np.max(signal.data))[0][0]
        plt.plot(
                (wt.stats.starttime - utc) + (_idxn*noise.stats.delta),
                 np.max(noise.data), marker="o", color="darkred")
        plt.plot(0.0 + (_idxs*signal.stats.delta),
                 np.max(signal.data), marker="o", color="darkred")
        #
        plt.title('[%s] Max(Signal) / Max(Noise) = %5.2f' % (outdict['result'],
                                                             _check_val))
        plt.show()
    return outdict


def mean_signal2noise_ratio(proc_trace,
                            raw_trace,
                            utc,
                            noise_window=0.1,
                            signal_window=0.1,
                            threshold=0.2,
                            use_raw=False,
                            normalize=[],
                            debug_plot=False):
    """Calculate the MEAN ratio of a custom signal2noise windows

    If the ratio of the MEAN signal window and the MEAN noise window
    is higher or equal the given threshold, the test is passed.

    Args:
        proc_trace (obspy.core.Trace): the processed work trace
        raw_trace (obspy.core.Trace): the raw input work trace
        utc (obspy.core.UTCDateTime): the test's picktime associated

        noise_window (:obj:`float`, optional): the amount of seconds
            to be used for trim the noise window around utc.
            Default to 0.1.
        signal_window (:obj:`float`, optional): the amount of seconds
            to be used for trim the signal window around utc.
            Default to 0.1.
        threshold (:obj:`float`, optional): the threshold value for the
            mean signal2noise ratio. Default to 0.2.
        use_raw (:obj:`bool`, optional): if set to True, the raw trace
            is used for the test instead. Default to False.
        normalize (:obj:`list`, optional): if a list is given, the trace
            is normalized based on the given values [min, max] and will
            affect the entire trace given in input. Default to None.

    Returns:
        outdict (dict): this dictionary contains the 'result' boolean
            key and 'output' key containing the calculated ratio value

    Note:
        This method works with the absolute values of the trace

    """
    outdict = {}
    # Selection
    if use_raw:
        wt = raw_trace.copy()
    else:
        wt = proc_trace.copy()
    # Processing and Selection
    wt.data = np.abs(wt.data)
    if normalize:
        if isinstance(normalize, list):
            wt.data = normalizeTrace(wt.data, rangeVal=normalize)
        else:
            raise QE.InvalidParameter()
    noise = wt.slice(utc - noise_window, utc)
    signal = wt.slice(utc, utc + signal_window)
    #
    _check_val = np.mean(signal.data)/np.mean(noise.data)
    if _check_val >= threshold:
        outdict['result'] = True
        outdict['output'] = _check_val
    else:
        outdict['result'] = False
        outdict['output'] = _check_val
    #
    if debug_plot:
        plt.plot(wt.times(reftime=utc), wt.data)
        plt.axvline(utc - utc,
                    color="gold",
                    linewidth=2,
                    linestyle='solid',
                    label="pick")
        plt.axvline(noise.stats.starttime - utc,
                    color="grey",
                    linewidth=1,
                    linestyle=':')
        plt.axvline(signal.stats.endtime - utc,
                    color="grey",
                    linewidth=1,
                    linestyle=':')
        plt.plot((noise.stats.starttime - utc, noise.stats.endtime - utc),
                 (np.mean(noise.data), np.mean(noise.data)),
                 color="darkred",
                 linewidth=1,
                 linestyle='--',
                 label="noise_mean")
        plt.plot((signal.stats.starttime - utc, signal.stats.endtime - utc),
                 (np.mean(signal.data), np.mean(signal.data)),
                 color="darkred",
                 linewidth=1,
                 linestyle='--',
                 label="signal_mean")
        #
        plt.title('[%s] Mean(Signal) / Mean(Noise) = %5.2f' % (
                                                    (_check_val >= threshold),
                                                    _check_val))
        plt.show()
    #
    return outdict


# def LowFreqTrend(wt, bpd, timewin, conf=0.95):
def low_freq_trend(proc_trace,
                   raw_trace,
                   utc,
                   sign_window=0.1,
                   angle_range=(0, 10),
                   angle_window=0.1,
                   confidence=0.2,
                   use_raw=False,
                   normalize=[],
                   debug_plot=False):
    """
    This method should help avoiding mispicks due
    to the so-called filter effect by recognizing trends (pos or negative)
    return False if trend found --> bad pick

    v0.4.1: adding a general trend with a angle range interval

    """
    outdict = {}
    # ------------------- Selection
    if use_raw:
        wt = raw_trace.copy()
    else:
        wt = proc_trace.copy()
    # Processing and Selection
    wt.data = np.abs(wt.data)
    if normalize:
        if isinstance(normalize, list):
            wt.data = normalizeTrace(wt.data, rangeVal=normalize)
        else:
            raise QE.InvalidParameter()

    workt_sign = wt.slice(utc, utc + sign_window)
    workt_angle = wt.slice(utc, utc + angle_window)

    # ------------------- Sign cofidence
    asign = np.sign(np.diff(workt_sign.data))
    unique, counts = np.unique(asign, return_counts=True)
    dsign = dict(zip(unique, counts))
    #
    for key in (-1.0, 1.0):
        if key in dsign and dsign[key]:
            pass
        else:
            dsign[key] = 0

    # Checks
    pos_trend = dsign[1.0]/len(asign)
    neg_trend = dsign[-1.0]/len(asign)
    if pos_trend >= confidence or neg_trend >= confidence:
        # Test Failed
        sign_test_pass = False
    else:
        # Test Passed
        sign_test_pass = True

    # ------------------- Derivative
    hh = workt_angle.data[-1] - workt_angle.data[0]
    ll = angle_window
    angle_res = np.degrees(np.arctan(hh/ll))
    if angle_res >= angle_range[0] and angle_res <= angle_range[1]:
        # Test Failed
        angle_test_pass = False
    else:
        # Test Passed
        angle_test_pass = True

    # ------------------- Populate output
    outdict['output'] = {'pos': pos_trend,
                         'neg': neg_trend,
                         'angle': angle_res}

    # ------------------- Result
    if not sign_test_pass or not angle_test_pass:
        outdict['result'] = False
    else:
        outdict['result'] = True
    #
    if debug_plot:
        plt.plot(wt.times(reftime=utc), wt.data)
        plt.axvline(utc - utc,
                    color="gold",
                    linewidth=2,
                    linestyle='solid',
                    label="pick")

        plt.plot([utc - utc, workt_angle.stats.endtime - utc],
                 [workt_angle.data[0], workt_angle.data[-1]],
                 color="darkred",
                 linewidth=2,
                 linestyle='solid',
                 label="angle")
        #
        plt.title(('[%s] Angle Test = %5.2f / ' +
                   '[%s] Sign Test (POS: %5.2f - NEG: %5.2f)') % (
                                                    angle_test_pass,
                                                    angle_res,
                                                    sign_test_pass,
                                                    pos_trend, neg_trend))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    #
    return outdict
