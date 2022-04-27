import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal
#
from obspy import Trace
from obspy import UTCDateTime
#
from adapt.stream_evaluation import _sec2sample, _nearest_pow_2
import adapt.errors as QE
import logging

"""
In this module are stored all the features extraction
funtions and routines ordered for time and frequency domains.

Some example of features are extracted from:
- http://tsfresh.readthedocs.io/en/latest/text/list_of_features.html

"""

logger = logging.getLogger(__name__)


# ========================================================
# ======================================================== PRIVATE
# ========================================================

def _dospectrum(wt, startt, endt, nfft_exp=None, nfft_value=None):
    """ Calculate PSD and return it together with frequency axis
        plus the dominant frequency (high energy) and its relative psd
        value.

        nfft_value has more importance over nfft_exp.
        - nfft_value will specify the exact value!
        - nfft_exp will create the NFFT will increase the exponential of
                   the 2-based formalae
    """
    tr = wt.copy()
    tr = tr.trim(startt, endt)
    tr.detrend(type='demean')
    tr.taper(max_percentage=0.05)

    # ===== Switch for NFFT: the nfft_value has the priority
    if nfft_value and nfft_exp:
        logger.warning("WARNING: both nfft_exp and nfft_value are given! " +
                       "Only nfft_value will be used ...")

    if nfft_value:
        nfft = nfft_value
    elif nfft_exp:
        nfft = int(2**(np.ceil(np.log2(tr.data.size)) + nfft_exp))
    else:
        nfft = int(2**(np.ceil(np.log2(tr.data.size))))
    cutidx = int(np.floor(nfft/2.0))
    # ======================================================

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


def _calcSpectrogramWave(tr,
                         window="hann", nperseg=0.05,   # sec
                         noverlap=0.025,  # sec
                         nfft=0.07,   # sec
                         detrend="constant",
                         scaling="spectrum"):
    """
    Simply create the spectrogram of the input array.
    """
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
        raise TypeError

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


def _normalizeTrace(workList, rangeVal=[-1, 1]):
    """
    This simple method will normalize the trace between rangeVal.
    Simply by scaling everything...

    """
    minVal, maxVal = min(workList), max(workList)
    workList[:] = [((x - minVal) / (maxVal - minVal)) *
                   (rangeVal[1] - rangeVal[0]) for x in workList]
    workList = workList + rangeVal[0]
    return workList


def _createCF(inarray, funct_list=None):
    """
    Simple method to create the carachteristic function of BaIt
    picking algorithm

    *** NB: The outarray of 13.02.2019 (the squared one), better enanche
            impulsive features of the signal, but it's really weak on
            emergent arrivals, especially with The SignalAmp feature.
    """
    if not funct_list:
        outarray = _instantenous_attributes_AMP(inarray)  # MB 20.03.2019 - test -
        # outarray = abs(inarray)                         # ORIGINAL
        # outarray = abs(inarray**2)                      # MB 13.02.2019 - test -
        # outarray = np.sqrt(abs(inarray))                # MB 13.02.2019 - test -
        outarray = _normalizeTrace(outarray, rangeVal=[0, 1])  # ORIGINAL
    return outarray


# Reference for INSTANTANEOUS ATTRIBUTES:
# https://www.gaussianwaves.com/2017/04/extracting-instantaneous-amplitude-phase-frequency-hilbert-transform/
def _instantenous_attributes_AMP(x):
    """
    From instantenous attributes of a seismic signal
    x is a numpy array
    """
    z = signal.hilbert(x)        # form the analytical signal
    inst_amplitude = np.abs(z)   # envelope extraction
    return inst_amplitude


def _instantenous_attributes_PHASE(x):
    """
    From instantenous attributes of a seismic signal
    x is a numpy array
    """
    z = signal.hilbert(x)        # form the analytical signal
    inst_phase = np.unwrap(np.angle(z))
    return inst_phase


def _instantenous_attributes_FREQ(x, fs):
    """
    From instantenous attributes of a seismic signal
    x is a numpy array

    *** NB the output array will have 1 sample less (from the start)
    """
    z = signal.hilbert(x)        # form the analytical signal
    inst_phase = np.unwrap(np.angle(z))
    inst_freq = np.diff(inst_phase)/(2*np.pi)*fs
    return inst_freq


# ========================================================
# ======================================================== TIME
# ========================================================


def max_signal_to_noise_ratio(raw,
                              proc,
                              pick_time,
                              use_raw=False,
                              usecf=False,
                              signal_window=1.0,
                              noise_window=1.0,
                              buffer_window=0.0,
                              compare_to_mean_noise=True):
    """
    Return the ratio between the MAX value of signal window
    against the MAX value of noise window

    Possibility of select a buffer period for Noise and Signal from
    picktime (buffer_window)

    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if not isinstance(pick_time, UTCDateTime):
        logger.error("PICK TIME is not valid obspy.UTCDateTime instance!")
        raise QE.InvalidType()
    if signal_window < 0.0 or noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    sigWave = wt.slice(pick_time + buffer_window,
                       pick_time + buffer_window + signal_window)
    noiWave = wt.slice(pick_time - buffer_window - noise_window,
                       pick_time - buffer_window)
    #

    sigmax = np.max(sigWave.data)
    noimax = np.max(noiWave.data)
    noimean = np.mean(noiWave.data)
    #
    if compare_to_mean_noise:
        outval = sigmax/noimean
    else:
        outval = sigmax/noimax
    return float(outval)


def mean_signal_to_noise_ratio(raw,
                               proc,
                               pick_time,
                               use_raw=False,
                               usecf=False,
                               signal_window=1.0,
                               noise_window=1.0,
                               buffer_window=0.0,
                               compare_to_mean_noise=True):
    """
    Return the ratio between the MAX value of signal window
    against the MAX value of noise window

    Possibility of select a buffer period for Noise and Signal from
    picktime (buffer_window)

    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if not isinstance(pick_time, UTCDateTime):
        logger.error("PICK TIME is not valid obspy.UTCDateTime instance!")
        raise QE.InvalidType()
    if signal_window < 0.0 or noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    sigWave = wt.slice(pick_time + buffer_window,
                       pick_time + buffer_window + signal_window)
    noiWave = wt.slice(pick_time - buffer_window - noise_window,
                       pick_time - buffer_window)
    #

    sigmean = np.mean(sigWave.data)
    noimax = np.max(noiWave.data)
    noimean = np.mean(noiWave.data)
    #
    if compare_to_mean_noise:
        outval = sigmean/noimean
    else:
        outval = sigmean/noimax
    return float(outval)


def max_signal_to_longnoise_ratio(raw,
                                  proc,
                                  pick_time,
                                  use_raw=False,
                                  usecf=False,
                                  signal_window=1.0,
                                  buffer_signal_window=0.0,
                                  longnoise_window=1.0,
                                  buffer_longnoise_window=1.0,
                                  compare_to_mean_noise=True):
    """ Helper function to calculate 3 different amplitudes:
            - Signal one
            - Noise one
            - Long noise one
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)

    # ------- WORK
    sig = wt.slice(pick_time, pick_time + buffer_signal_window + signal_window)
    noi = wt.slice(pick_time - buffer_longnoise_window - longnoise_window,
                   pick_time - buffer_longnoise_window)

    sigmax = np.max(sig.data)
    noimax = np.max(noi.data)
    noimean = np.mean(noi.data)

    #
    if compare_to_mean_noise:
        outval = sigmax/noimean
    else:
        outval = sigmax/noimax
    return float(outval)


def mean_signal_to_longnoise_ratio(raw,
                                   proc,
                                   pick_time,
                                   use_raw=False,
                                   usecf=False,
                                   signal_window=1.0,
                                   buffer_signal_window=0.0,
                                   longnoise_window=1.0,
                                   buffer_longnoise_window=1.0,
                                   compare_to_mean_noise=True):
    """ Helper function to calculate 3 different amplitudes:
            - Signal one
            - Noise one
            - Long noise one
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)

    # ------- WORK
    sig = wt.slice(pick_time, pick_time + buffer_signal_window + signal_window)
    noi = wt.slice(pick_time - buffer_longnoise_window - longnoise_window,
                   pick_time - buffer_longnoise_window)

    sigmean = np.mean(sig.data)
    noimax = np.max(noi.data)
    noimean = np.mean(noi.data)

    #
    if compare_to_mean_noise:
        outval = sigmean/noimean
    else:
        outval = sigmean/noimax
    return float(outval)


def max_signal_to_startnoise_ratio(raw,
                                   proc,
                                   pick_time,
                                   use_raw=False,
                                   usecf=False,
                                   signal_window=1.0,
                                   buffer_signal_window=0.0,
                                   startnoise_window=1.0,
                                   buffer_startnoise_window=1.0,
                                   compare_to_mean_noise=True):
    """ Helper function to calculate 3 different amplitudes:
            - Signal one
            - Noise one
            - Long noise one
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)

    # ------- WORK
    sig = wt.slice(pick_time, pick_time + buffer_signal_window + signal_window)
    noi = wt.slice(wt.stats.starttime + buffer_startnoise_window,
                   wt.stats.starttime + buffer_startnoise_window + startnoise_window)

    sigmax = np.max(sig.data)
    noimax = np.max(noi.data)
    noimean = np.mean(noi.data)

    #
    if compare_to_mean_noise:
        outval = sigmax/noimean
    else:
        outval = sigmax/noimax
    return float(outval)


def mean_signal_to_startnoise_ratio(raw,
                                    proc,
                                    pick_time,
                                    use_raw=False,
                                    usecf=False,
                                    signal_window=1.0,
                                    buffer_signal_window=0.0,
                                    startnoise_window=1.0,
                                    buffer_startnoise_window=1.0,
                                    compare_to_mean_noise=True):
    """ Helper function to calculate 3 different amplitudes:
            - Signal one
            - Noise one
            - Long noise one
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)

    # ------- WORK
    sig = wt.slice(pick_time, pick_time + buffer_signal_window + signal_window)
    noi = wt.slice(wt.stats.starttime + buffer_startnoise_window,
                   wt.stats.starttime + buffer_startnoise_window + startnoise_window)

    sigmean = np.mean(sig.data)
    noimax = np.max(noi.data)
    noimean = np.mean(noi.data)

    #
    if compare_to_mean_noise:
        outval = sigmean/noimean
    else:
        outval = sigmean/noimax
    return float(outval)


def max_longnoise_to_startnoise_ratio(raw,
                                      proc,
                                      pick_time,
                                      use_raw=False,
                                      usecf=False,
                                      longnoise_window=1.0,
                                      buffer_longnoise_window=1.0,
                                      startnoise_window=1.0,
                                      buffer_startnoise_window=1.0,
                                      compare_to_mean_noise=True):
    """ Helper function to calculate 3 different amplitudes:
            - Signal one
            - Noise one
            - Long noise one
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)

    # ------- WORK
    sig = wt.slice(pick_time - buffer_longnoise_window - longnoise_window,
                   pick_time - buffer_longnoise_window)
    noi = wt.slice(wt.stats.starttime + buffer_startnoise_window,
                   wt.stats.starttime + buffer_startnoise_window + startnoise_window)

    longmax = np.max(sig.data)
    startmax = np.max(noi.data)
    startmean = np.mean(noi.data)

    #
    if compare_to_mean_noise:
        outval = longmax/startmean
    else:
        outval = longmax/startmax
    return float(outval)


def mean_longnoise_to_startnoise_ratio(raw,
                                       proc,
                                       pick_time,
                                       use_raw=False,
                                       usecf=False,
                                       longnoise_window=1.0,
                                       buffer_longnoise_window=1.0,
                                       startnoise_window=1.0,
                                       buffer_startnoise_window=1.0,
                                       compare_to_mean_noise=True):
    """ Helper function to calculate 3 different amplitudes:
            - Signal one
            - Noise one
            - Long noise one
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)

    # ------- WORK
    sig = wt.slice(pick_time - buffer_longnoise_window - longnoise_window,
                   pick_time - buffer_longnoise_window)
    noi = wt.slice(wt.stats.starttime + buffer_startnoise_window,
                   wt.stats.starttime + buffer_startnoise_window + startnoise_window)

    longmean = np.max(sig.data)
    startmax = np.max(noi.data)
    startmean = np.mean(noi.data)

    #
    if compare_to_mean_noise:
        outval = longmean/startmean
    else:
        outval = longmean/startmax
    return float(outval)


def std_noise(raw,
              proc,
              pick_time,
              use_raw=False,
              usecf=False,
              noise_window=1.0,
              buffer_window=0.0):
    """
    Just calculating the variance nearby the pick

    Possibility of select a buffer period for Noise and Signal from
    picktime (buffer_window)
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if not isinstance(pick_time, UTCDateTime):
        logger.error("PICK TIME is not valid obspy.UTCDateTime instance!")
        raise QE.InvalidType()
    if noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    wave = wt.slice(pick_time - buffer_window - noise_window,
                    pick_time - buffer_window)
    return float(np.std(wave.data))


def std_signal(raw,
               proc,
               pick_time,
               use_raw=False,
               usecf=False,
               signal_window=1.0,
               buffer_window=0.0):
    """
    Just calculating the variance nearby the pick

    Possibility of select a buffer period for Noise and Signal from
    picktime (buffer_window)
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if not isinstance(pick_time, UTCDateTime):
        logger.error("PICK TIME is not valid obspy.UTCDateTime instance!")
        raise QE.InvalidType()
    if signal_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------
    #
    wave = wt.slice(pick_time + buffer_window,
                    pick_time + buffer_window + signal_window)
    return float(np.std(wave.data))


def var_noise(raw,
              proc,
              pick_time,
              use_raw=False,
              usecf=False,
              noise_window=1.0,
              buffer_window=0.0):
    """
    Just calculating the variance nearby the pick

    Possibility of select a buffer period for Noise and Signal from
    picktime (buffer_window)
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if not isinstance(pick_time, UTCDateTime):
        logger.error("PICK TIME is not valid obspy.UTCDateTime instance!")
        raise QE.InvalidType()
    if noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    wave = wt.slice(pick_time - buffer_window - noise_window,
                    pick_time - buffer_window)
    return float(np.var(wave.data))


def var_signal(raw,
               proc,
               pick_time,
               use_raw=False,
               usecf=False,
               signal_window=1.0,
               buffer_window=0.0):
    """
    Just calculating the variance nearby the pick

    Possibility of select a buffer period for Noise and Signal from
    picktime (buffer_window)
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if not isinstance(pick_time, UTCDateTime):
        logger.error("PICK TIME is not valid obspy.UTCDateTime instance!")
        raise QE.InvalidType()
    if signal_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------
    #
    wave = wt.slice(pick_time + buffer_window,
                    pick_time + buffer_window + signal_window)
    return float(np.var(wave.data))


def complexity_factor(raw,
                      proc,
                      use_raw=False,
                      usecf=False,
                      z_transform=True):
    """
    This function calculator is an estimate for a time series complexity
    [1] (A more complex time series has more peaks, valleys etc.).
    It calculates the value of

    .. math::

        \\sqrt{ \\sum_{i=0}^{n-2lag} ( x_{i} - x_{i+1})^2 }

    .. rubric:: References

    |  [1] Batista, Gustavo EAPA, et al (2014).
    |  CID: an efficient complexity-invariant distance for time series.
    |  Data Mining and Knowledge Difscovery 28.3 (2014): 634-669.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param normalize: should the time series be z-transformed?
    :type normalize: bool

    :return: the value of this feature
    :return type: float
    """

    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    x = np.asarray(wt.data)
    if z_transform:
        s = np.std(x)
        if s != 0:
            x = (x - np.mean(x)) / s
        else:
            return 0.0

    x = np.diff(x)
    return np.sqrt(np.sum((x * x)))


def absolute_energy(raw,
                    proc,
                    pick_time,
                    use_raw=False,
                    usecf=False,
                    signal_window=1.0,
                    noise_window=1.0,
                    buffer_window=0.0):
    """
    Return the squared sum of the given array. Selection can be made

    .. math::
        \\sum_{i=0}^{N} x_{i}^2

    :param wave:
    :type wave:
    :param pt:
    :type pt:
    :param winSize: trimming around the given time
    :type winSize: float

    :return: array's squared sum value
    :rtype: float

    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if signal_window < 0.0 or noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    wt = wt.slice(pick_time - buffer_window - noise_window,
                  pick_time + buffer_window + signal_window)
    return float(np.sum(np.square(wt.data)))


def absolute_energy_signal(raw,
                           proc,
                           pick_time,
                           use_raw=False,
                           usecf=False,
                           signal_window=1.0,
                           buffer_window=0.0):
    """
    Return the squared sum of the given array. Selection can be made

    .. math::
        \\sum_{i=0}^{N} x_{i}^2

    :param wave:
    :type wave:
    :param pt:
    :type pt:
    :param winSize: trimming around the given time
    :type winSize: float

    :return: array's squared sum value
    :rtype: float

    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if signal_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    wt = wt.slice(pick_time,
                  pick_time + buffer_window + signal_window)
    return float(np.sum(np.square(wt.data)))


def absolute_energy_noise(raw,
                          proc,
                          pick_time,
                          use_raw=False,
                          usecf=False,
                          noise_window=1.0,
                          buffer_window=0.0):
    """
    Return the squared sum of the given array. Selection can be made

    .. math::
        \\sum_{i=0}^{N} x_{i}^2

    :param wave:
    :type wave:
    :param pt:
    :type pt:
    :param winSize: trimming around the given time
    :type winSize: float

    :return: array's squared sum value
    :rtype: float

    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    wt = wt.slice(pick_time - buffer_window - noise_window, pick_time)
    return float(np.sum(np.square(wt.data)))


def sum_values(raw,
               proc,
               pick_time,
               use_raw=False,
               usecf=False,
               signal_window=1.0,
               noise_window=1.0,
               buffer_window=0.0):
    """
    Calculates the sum over the time series values

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if signal_window < 0.0 or noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    wt = wt.slice(pick_time - buffer_window - noise_window,
                  pick_time + buffer_window + signal_window)
    return np.sum(wt.data)


def mean_autocorrelation(raw,
                         proc,
                         pick_time,
                         use_raw=False,
                         usecf=False,
                         signal_window=1.0,
                         noise_window=1.0,
                         buffer_window=0.0):
    """ Portion of signal pre/post pick """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if signal_window < 0.0 or noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    # Work
    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        return result[len(result) // 2:]
    #
    tmp = wt.slice(pick_time - buffer_window - noise_window,
                   pick_time + buffer_window + signal_window)
    return autocorr(tmp.data).mean()


def max_autocorrelation(raw,
                        proc,
                        pick_time,
                        use_raw=False,
                        usecf=False,
                        signal_window=1.0,
                        noise_window=1.0,
                        buffer_window=0.0):
    """ Portion of signal pre/post pick """
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if signal_window < 0.0 or noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    # Work
    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        return result[len(result) // 2:]
    #
    tmp = wt.slice(pick_time - buffer_window - noise_window,
                   pick_time + buffer_window + signal_window)
    return autocorr(tmp.data).max()


def noise_over_threshold(raw,
                         proc,
                         pick_time,
                         threshold=0.5,  # can be also 'sd', 'var', '2_sd'
                         use_raw=False,
                         usecf=False,
                         # signal_window=1.0,
                         noise_window=1.0,
                         buffer_window=0.0,
                         debug_plot=False):
    """ Portion of signal pre/post pick, percentage of signal over threshold"""
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()

    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    # Work
    tmp = wt.slice(pick_time - buffer_window - noise_window, pick_time)

    # define thresh
    if isinstance(threshold, str):
        if threshold.lower() == "sd":
            mythr = np.std(tmp.data)
        elif threshold.lower() == "var":
            mythr = np.var(tmp.data)
        else:
            tmplst = threshold.split('_')
            if np.size(tmplst) == 2 and tmplst[-1].lower() == "sd":
                mythr = float(tmplst[0]) * np.std(tmp.data)
            elif np.size(tmplst) == 2 and tmplst[-1].lower() == "var":
                mythr = float(tmplst[0]) * np.var(tmp.data)
            else:
                logger.error("something wrong with input input parameters! " +
                             "Check the guidelines (i.e 2_sd ecc..)")
                raise QE.InvalidParameter()

    else:
        mythr = threshold

    # search and return percent
    match_arr = np.where(tmp.data >= mythr)[0]    # above threshold
    if match_arr.size >= 1 and tmp.data.size >= 1:
        myperc = match_arr.size / tmp.data.size
    else:
        myperc = 0.0

    if debug_plot:
        import matplotlib.pyplot as plt
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(proc.times("matplotlib"), proc.data)
        ax1.axvline(pick_time.datetime)
        plt.title(("%6.2f %%") % myperc)
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(tmp.data)
        ax2.axhline(mythr, color="black")
        plt.show()

    return myperc


def signal_below_threshold(raw,
                           proc,
                           pick_time,
                           threshold=0.5,  # can be also 'sd', 'var', '2_sd'
                           use_raw=False,
                           usecf=False,
                           signal_window=1.0,
                           # noise_window=1.0,
                           buffer_window=0.0,
                           debug_plot=False):
    """ Portion of signal pre/post pick, percentage of signal over threshold"""
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if signal_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()

    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)
    # -------

    # Work
    tmp = wt.slice(pick_time, pick_time + buffer_window + signal_window)

    # define thresh
    if isinstance(threshold, str):
        if threshold.lower() == "sd":
            mythr = np.std(tmp.data)
        elif threshold.lower() == "var":
            mythr = np.var(tmp.data)
        else:
            tmplst = threshold.split('_')
            if np.size(tmplst) == 2 and tmplst[-1].lower() == "sd":
                mythr = float(tmplst[0]) * np.std(tmp.data)
            elif np.size(tmplst) == 2 and tmplst[-1].lower() == "var":
                mythr = float(tmplst[0]) * np.var(tmp.data)
            else:
                logger.error("something wrong with input input parameters! " +
                             "Check the guidelines (i.e 2_sd ecc..)")
                raise QE.InvalidParameter()

    else:
        mythr = threshold

    # search and return percent
    match_arr = np.where(tmp.data <= mythr)[0]   # below threshold
    if match_arr.size >= 1 and tmp.data.size >= 1:
        myperc = match_arr.size / tmp.data.size
    else:
        myperc = 0.0

    if debug_plot:
        import matplotlib.pyplot as plt
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(proc.times("matplotlib"), proc.data)
        ax1.axvline(pick_time.datetime)
        plt.title(("%6.2f %%") % myperc)
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(tmp.data)
        ax2.axhline(mythr, color="black")
        plt.show()

    return myperc


def dominant_frequencies_shift(raw,
                               proc,
                               pick_time,
                               use_raw=False,
                               usecf=False,
                               signal_window=1.0,
                               noise_window=1.0,
                               buffer_window=0.0,
                               debug_plot=False):
    """ Portion of signal pre/post pick, percentage of signal over threshold"""
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if signal_window < 0.0 or noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)

    # ==== Common Highest NFFT  (v0.5.8) after Memory Error bugfix
    noisamp = wt.slice(pick_time - buffer_window - noise_window, pick_time)
    noinfft = int(2**(np.ceil(np.log2(noisamp.data.size)) + 2))

    sigsamp = wt.slice(pick_time, pick_time + buffer_window + signal_window)
    signfft = int(2**(np.ceil(np.log2(sigsamp.data.size)) + 2))

    mynfft = np.max([signfft, noinfft])
    # ==============================================================

    # ------- WORK
    # Calculate spectrogram
    fax_noi, psd_noi, noifreq, _ = _dospectrum(
                                   wt,
                                   pick_time - buffer_window - noise_window,
                                   pick_time, nfft_value=mynfft)
    fax_sig, psd_sig, sigfreq, _ = _dospectrum(
                                   wt, pick_time,
                                   pick_time + buffer_window + signal_window,
                                   nfft_value=mynfft)
    if debug_plot:
        import matplotlib.pyplot as plt
        ax1 = plt.subplot(1, 1, 1)
        ax1.semilogy(fax_noi, psd_noi, color="gold")
        ax1.semilogy(fax_sig, psd_sig, color="teal")
        ax1.axvline(noifreq, color="gold")
        ax1.axvline(sigfreq, color="teal")
        plt.title("Fs-Fn = %6.2f" % (sigfreq - noifreq))
        plt.xlabel("Frequency")
        plt.ylabel("PSD")
        plt.show()

    # Fs-Fn
    return (sigfreq - noifreq)


def dominant_frequencies_ratio(raw,
                               proc,
                               pick_time,
                               use_raw=False,
                               usecf=False,
                               signal_window=1.0,
                               noise_window=1.0,
                               buffer_window=0.0,
                               debug_plot=False):
    """ Portion of signal pre/post pick, percentage of signal over threshold"""
    # ------- MB: next lines are for setup only
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if signal_window < 0.0 or noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)

    # ==== Common Highest NFFT  (v0.5.8) after Memory Error bugfix
    noisamp = wt.slice(pick_time - buffer_window - noise_window, pick_time)
    noinfft = int(2**(np.ceil(np.log2(noisamp.data.size)) + 2))

    sigsamp = wt.slice(pick_time, pick_time + buffer_window + signal_window)
    signfft = int(2**(np.ceil(np.log2(sigsamp.data.size)) + 2))

    mynfft = np.max([signfft, noinfft])
    # ==============================================================

    # ------- WORK
    # Calculate spectrogram
    fax_noi, psd_noi, noifreq, noienergy = _dospectrum(
                                   wt,
                                   pick_time - buffer_window - noise_window,
                                   pick_time, nfft_value=mynfft)
    fax_sig, psd_sig, sigfreq, sigenergy = _dospectrum(
                                   wt, pick_time,
                                   pick_time + buffer_window + signal_window,
                                   nfft_value=mynfft)

    if debug_plot:
        import matplotlib.pyplot as plt
        ax1 = plt.subplot(1, 1, 1)
        ax1.semilogy(fax_noi, psd_noi, color="gold")
        ax1.semilogy(fax_sig, psd_sig, color="teal")
        ax1.axvline(noifreq, color="gold")
        ax1.axvline(sigfreq, color="teal")
        plt.title("Fs/Fn = %6.2f" % (sigenergy / noienergy))
        plt.xlabel("Frequency")
        plt.ylabel("PSD")
        plt.show()

    # Fs-Fn
    return (sigenergy / noienergy)


def dominant_frequency_signal(raw,
                              proc,
                              pick_time,
                              use_raw=False,
                              usecf=False,
                              signal_window=1.0,
                              buffer_window=0.0):
    """ Portion of signal pre/post pick, percentage of signal over threshold

        NB: if used in combination with
            - dominant_frequencises_shift
            - dominant_frequencises_ratio
            PLEASE BE CONSISTENT WITH THE WINDOW SELECTION !!!
            The selected time window will affect the NFFT and therefore
            the final domninant frequency
    """
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if signal_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #
    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)

    # Calculate spectrogram
    _, _, sigfreq, _ = _dospectrum(wt, pick_time,
                                   pick_time + buffer_window + signal_window,
                                   nfft_exp=2)
    # Fs-Fn
    return sigfreq


def dominant_frequency_noise(raw,
                             proc,
                             pick_time,
                             use_raw=False,
                             usecf=False,
                             noise_window=1.0,
                             buffer_window=0.0):
    """ Portion of signal pre/post pick, percentage of signal over threshold

        NB: if used in combination with
            - dominant_frequencises_shift
            - dominant_frequencises_ratio
            PLEASE BE CONSISTENT WITH THE WINDOW SELECTION !!!
            The selected time window will affect the NFFT and therefore
            the final domninant frequency
    """
    if not isinstance(proc, Trace) or not isinstance(raw, Trace):
        logger.error("INPUT TRACES are not valid obspy.Trace instance!")
        raise QE.InvalidType()
    if noise_window < 0.0:
        logger.error("NOISE or SIGNAL windows are negative!")
        raise QE.InvalidVariable()
    #

    if use_raw:
        wt = raw.copy()
    else:
        wt = proc.copy()
    #
    if usecf:
        wt.data = _createCF(wt.data)

    # Calculate spectrogram
    _, _, noifreq, _ = _dospectrum(wt,
                                   pick_time - buffer_window - noise_window,
                                   pick_time,
                                   nfft_exp=2)
    # Fs-Fn
    return noifreq

# ========================================================
# ======================================================== BASKET to take from
# ========================================================

# --> *** NB: Format to the module standard if you want to use it

# def local_minima_noise(wave, pt, wint=1.0, usecf=False):
#     """
#     Implementation to find the number of local minima in the noise part
#     """
#     if not isinstance(wave, Trace):
#         print(type(pt))
#         raise TypeError
#     if not isinstance(pt, UTCDateTime):
#         print(type(pt))
#         raise TypeError
#     if usecf:
#         wave.data = _createCF(wave.data)
#     dd = wave.slice(pt-wint, pt)
#     #
#     _idx = []
#     for i in range(1, len(dd.data)):
#         try:
#             if (
#                 ((dd.data[i-1] - dd.data[i]) > 0 and
#                     (dd.data[i+1] - dd.data[i]) >= 0) or
#                 ((dd.data[i+1] - dd.data[i]) > 0 and
#                     (dd.data[i-1] - dd.data[i]) >= 0)):
#                 _idx.append(i)
#         except IndexError:
#             # If enter here is the boundaries of the array
#             continue
#     return len(_idx), _idx


# def local_minima_signal(wave, pt, wint=1.0, usecf=False):
#     """
#     Implementation to find the number of local minima in the signal part
#     """
#     if not isinstance(wave, Trace):
#         print(type(pt))
#         raise TypeError
#     if not isinstance(pt, UTCDateTime):
#         print(type(pt))
#         raise TypeError
#     dd = wave.slice(pt, pt+wint)
#     #
#     _idx = []
#     for i in range(1, len(dd.data)):
#         try:
#             if (
#                 ((dd.data[i-1] - dd.data[i]) > 0 and
#                     (dd.data[i+1] - dd.data[i]) >= 0) or
#                 ((dd.data[i+1] - dd.data[i]) > 0 and
#                     (dd.data[i-1] - dd.data[i]) >= 0)):
#                 _idx.append(i)
#         except IndexError:
#             # If enter here is the boundaries of the array
#             continue
#     return len(_idx), _idx


# def linear_regression_noise(wave, pt, wint=1.0, usecf=False):
#     """ Simply using the linear regression on top of it """
#     if not isinstance(wave, Trace):
#         print(type(pt))
#         raise TypeError
#     if not isinstance(pt, UTCDateTime):
#         print(type(pt))
#         raise TypeError
#     dd = wave.slice(pt-wint, pt)
#     #
#     x = np.arange(0, len(dd.data), 1)
#     y = dd.data
#     slope, intercept = np.polyfit(x, y, 1)
#     y_reg = [ii*slope+intercept for ii in range(len(dd.data))]
#     # plt.scatter(x,dd.data) ; plt.plot(x,y_reg,'r'); plt.show()
#     return slope, intercept


# def linear_regression_signal(wave, pt, wint=1.0, usecf=False):
#     """ Simply using the linear regression on top of it """
#     if not isinstance(wave, Trace):
#         print(type(pt))
#         raise TypeError
#     if not isinstance(pt, UTCDateTime):
#         print(type(pt))
#         raise TypeError
#     dd = wave.slice(pt, pt+wint)
#     #
#     x = np.arange(0, len(dd.data), 1)
#     y = dd.data
#     slope, intercept = np.polyfit(x, y, 1)
#     y_reg = [ii*slope+intercept for ii in range(len(dd.data))]
#     # plt.scatter(x,dd.data) ; plt.plot(x,y_reg,'r'); plt.show()
#     return slope, intercept


# def std_noise(wave, pt, wint=1.0):
#     """ Noise standard deviation from E[x] """
#     if not isinstance(wave, Trace):
#         print(type(pt))
#         raise TypeError
#     if not isinstance(pt, UTCDateTime):
#         print(type(pt))
#         raise TypeError
#     dd = wave.slice(pt-wint, pt)
#     #
#     return np.std(dd.data)


# def std_signal(wave, pt, wint=1.0):
#     """ Signal standard deviation from E[x] """
#     if not isinstance(wave, Trace):
#         print(type(pt))
#         raise TypeError
#     if not isinstance(pt, UTCDateTime):
#         print(type(pt))
#         raise TypeError
#     dd = wave.slice(pt, pt+wint)
#     #
#     return np.std(dd.data)


# def std_signal2noise_ratio(wave, pt, wint_noise=1.0, wint_signal=1.0):
#     """ Ratio between signal and noise window stds. """
#     if not isinstance(wave, Trace):
#         print(type(pt))
#         raise TypeError
#     if not isinstance(pt, UTCDateTime):
#         print(type(pt))
#         raise TypeError
#     ddn = wave.slice(pt-wint_noise, pt)
#     dds = wave.slice(pt, pt+wint_signal)
#     #
#     return np.std(dds.data)/np.std(ddn.data)

# ========================================================
# ======================================================== FREQUENCY
# ========================================================

# MB: To be formatted to the module standard if you want to use it
# def spectralRMS(wave, pt, wint=1.0):
#     ''' WinSize represent the time window pre and post pick
#         over which the feature is calculated.

#         Calculate the RMS of the fft Amplitude
#         of the window around the pick
#     '''
#     tmp = wave.slice(pt - wint, pt + wint)
#     myfft = np.fft.fft(tmp.data)
#     myfftAMP = abs(myfft)
#     #
#     myRMS = np.sqrt(np.mean(myfftAMP**2))
#     return myRMS


# MB: To be formatted to the module standard if you want to use it
# def InstAttr(wave, pt, winSize=1.0, usecf=False):
#     """
#     Just calculating the variance nearby the pick
#     """
#     if usecf:
#         wave.data = _instantenous_attributes_FREQ(
#                                                   wave.data,
#                                                   wave.stats.sampling_rate)
#         # wave.data = _instantenous_attributes_AMP(
#         #                                           wave.data)
#     #
#     wave = wave.slice(pt - winSize, pt + winSize)
#     wave.plot()
#     return float(np.var(wave.data))


# MB: To be formatted to the module standard if you want to use it
# def MySpectrogram(wave, pt, winSize=1.0, usecf=False):
#     """
#     Return some kind of variance of the spectrogram
#     """
#     if usecf:
#         wave.data = _createCF(wave.data)
#     #
#     wave = wave.slice(pt - winSize, pt + winSize)
#     specTuple = _calcSpectrogramWave(wave,
#                                      window="hann", nperseg=0.1,   # sec
#                                      noverlap=0.06,  # sec
#                                      nfft=0.05,   # sec
#                                      # detrend="constant",
#                                      detrend=None,
#                                      scaling="spectrum")
#     wave.plot()
#     ax, _ = QPL.plot_Spectrogram(specTuple[0],
#                                  specTuple[1],
#                                  specTuple[2],
#                                  show=True,
#                                  dbscale=False,
#                                  log=False)
#     return 10
