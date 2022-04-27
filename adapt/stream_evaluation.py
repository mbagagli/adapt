import sys
import logging
#
import adapt.errors as QE
#
from obspy.signal import freqattributes as fa
from scipy import signal  # For windowing
#
import math as M
import numpy as np
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

logger = logging.getLogger(__name__)


# --------------------------------------------- Orchestrator
def doStreamEvaluation(st, **kwargs):
    """
    This method is used to  check potentially noisy, unuseful
    stream prior the Multipicker phase.

    :type st: ObsPy.Stream
    :param st: input stream object used by the class
    """
    testsResultsBool, testsResultsDict = [], {}
    # sorting in alphabetical order the testfunctions
    sortedkeys = sorted(kwargs, key=str.lower)
    for xx in sortedkeys:
        logger.info("%s - %r" % (xx, kwargs[xx]))
        testFunction = getattr(sys.modules[__name__], xx)
        ts, td = testFunction(st, **kwargs[xx])
        testsResultsBool.append(ts)
        testsResultsDict[xx] = td
    # Print results
    for xx in sortedkeys:
        logger.info("%s --> %s" % (testsResultsDict[xx]["message"],
                    str(testsResultsDict[xx]["value"])))
    # Return results
    if False in testsResultsBool:
        logger.info("Stream not accepted")
        return False, testsResultsDict
    else:
        logger.info("Stream ACCEPTED")
        return True, testsResultsDict


# --------------------------------------------- Function
def Noise_StaLta(st, chan="Z", **kwargs):
    """
    This fast STA-LTA detector must be run on the entire trace
    prior the pickers run, in order to avoid mispicks.

    This function must be run on the vertical component channel.
    If no possible 'seismic' signal is present in vertical component
    no need to proceed seeking further on the remaining channels

    Original:
    # ---- vars: This variable is hardcoded
    # wsta = 3        # sec
    # wlta = 8        # sec
    # thr_on = 2.0
    # thr_off = 0.3

    """
    outDict = {}
    tr = st.select(channel=("*"+chan))[0]
    df = tr.stats.sampling_rate
    cft = recursive_sta_lta(tr.data,
                            int(kwargs["wsta"] * df),
                            int(kwargs["wlta"] * df))
    on_off = np.array(trigger_onset(cft, kwargs["thr_on"], kwargs["thr_off"]))
    logger.debug("%r" % on_off)
    if on_off.size == 0:
        outDict['result'] = False
        outDict['message'] = ("%s channel is too noisy, no evident pick!" %
                              chan)
        outDict['value'] = on_off
        return False, outDict
    else:
        outDict['result'] = True
        outDict['message'] = ("%s channel contains valid transient!" %
                              chan)
        outDict['value'] = on_off
        return True, outDict


def calcSpectrogram(st, channel="*Z",
                    window="hann", nperseg=0.05,   # sec
                    noverlap=0.025,  # sec
                    nfft=0.07,   # sec
                    detrend="constant",
                    scaling="spectrum"):
    """
    Simply create the spectrogram of the input array.
    """
    tr = st.select(channel=channel)[0]
    fs = tr.stats.sampling_rate
    npts = len(tr.data)
    # Convert
    nperseg = _sec2sample(fs, nperseg)
    noverlap = _sec2sample(fs, noverlap)
    nfft = _nearest_pow_2(_sec2sample(fs, noverlap))
    if nfft >= npts:
        nfft = _nearest_pow_2(npts / 8.0)

    # Select window
    if window.lower() in ('hann', 'hanning'):
        win = signal.hann(nperseg)
    elif window.lower() in ('black', 'blackman'):
        win = signal.blackman(nperseg)
    else:
        logger.error("Erroneous windowing type")
        raise QE.InvalidVariable()

    freqax, timeax, Sxx = signal.spectrogram(tr.data,
                                             tr.stats.sampling_rate,
                                             window=win,
                                             nperseg=nperseg,    # samplenow
                                             noverlap=noverlap,  # samplenow
                                             nfft=nfft,          # samplenow
                                             detrend=detrend,
                                             scaling=scaling
                                             )
    return (freqax, timeax, Sxx)


def calcFFT(st, chan="*Z", window="hann", nfft=1024):
    """
    Simply create the spectrogram of the input array.
    If doplot, also a matplotlib axes is returned.

    Return the frequency axes and the positive spectrum part

    Links:
     - https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    """
    tr = st.select(channel=chan)[0]
    fs = tr.stats.sampling_rate

    if window.lower() in ('hann', 'hanning'):
        win = signal.hann(len(tr.data))
    elif window.lower() in ('black', 'blackman'):
        win = signal.blackman(len(tr.data))
    else:
        logger.error("Erroneous windowing type")
        raise QE.InvalidVariable()

    ssp = fa.spectrum(tr.data, win, nfft)
    # taking only the positive freq part --> calc freq
    ssp1 = ssp[0:len(ssp)//2]
    fax1 = np.arange(0, len(ssp1))*fs/nfft
    return fax1, ssp1


def _sec2sample(fs, value):
    """
    Utility method to define convert USER input parameter (seconds)
    into obspy pickers 'n_sample' units.

    Python3 round(float)==int // Python2 round(float)==float
    BETTER USE: int(round(... to have compatibility

    Formula: int(round( INSEC * df))
    *** NB: in sec could be float
    """
    return int(round(value * fs))


def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0

    :type x: float
    :param x: Number
    :rtype: Int
    :return: Nearest power of 2 to x
    """
    a = M.pow(2, M.ceil(np.log2(x)))
    b = M.pow(2, M.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b
