import os
import logging
import numpy as np
from scipy.fft import fft, ifft, fftfreq
#
from obspy import read
#
import matplotlib.pyplot as plt
import adapt.utils as QU
import adapt.errors as QE

logger = logging.getLogger(__name__)


# -------------------------------------------- Functions


def importWaveformsStream(waveformdir, format="SAC", complist=None):
    """
    This function will return an obspy Stream object with all
    the data contained in the waveformdir.

    ***NB: quick and dirty, maybe a cross check for missing
    component should be added here.
    """
    logger.info("Loading waveforms dir")
    if complist and isinstance(complist, list):
        wildcardstr = "*" + str(complist) + "." + format
        instr = read(waveformdir+os.sep+wildcardstr)
    else:
        # load everything
        instr = read(waveformdir+os.sep+"*."+format)
    return instr


def importAllWaveforms(waveformdir):
    """
    This function will return an obspy Stream object with all
    the data contained in the waveformdir.

    NB: the data-format must be supported by obspy_read
    """
    logger.info("Importing ...")
    instr = read(waveformdir+os.sep+"*."+format)
    logger.info("a total of %d traces!" % len(instr))
    return instr


def processStream(st, copystream=True, clear_empty_traces=True,
                  integrate_accelerometer=True, **kwargs):
    """
    This function will process the input ObsPy stream.
    *args[0] must be an inventory file previously loaded by
    ADAPT or otherwise raise error.

    """
    logger.info("pre-processing the stream: %s waves" % len(st))

    # copy it
    if copystream:
        prst = st.copy()
    else:
        prst = st

    # clear Stream from empty traces (v0.5.11)
    if clear_empty_traces:
        empty_stream = prst.select(npts=0)
        if len(empty_stream) > 0:
            logger.info("Removing %d EMPTY TRACES ..." % len(empty_stream))
            for _tr in empty_stream:
                logger.debug(print(_tr))
                prst.remove(_tr)

    # Sensitivity // instrument response
    if (kwargs["PREPROCESS"]["removesensitivity"] and
       kwargs["ARRAY"]["loadinventory"]):
        prst.remove_sensitivity(kwargs["GENERAL"]["workingrootdir"] + os.sep +
                                kwargs["GENERAL"]["ARRAY"]["inventoryxmlfile"])
    # Process
    if kwargs["PREPROCESS"]["removemean"]:
        prst.detrend('demean')                    # Remove Mean
    if kwargs["PREPROCESS"]["removelineartrend"]:
        prst.detrend('simple')                    # Remove Linear Trend

    if kwargs["PREPROCESS"]["taperpar"]:
        prst.taper(kwargs["PREPROCESS"]["taperpar"][0],
                   kwargs["PREPROCESS"]["taperpar"][1])
    if kwargs["PREPROCESS"]["filterpar"]:
        filterStreamTrace(prst, **kwargs["PREPROCESS"]["filterpar"])

    # Integrate ACCELEROMETERS (v0.6.3)
    if integrate_accelerometer:
        for _ch in ("HG*", "HN*"):  # "CH*"): #CH are OBS --> veloc.
            accel_stream = prst.select(channel=_ch)
            if len(accel_stream) > 0:
                logger.info("Integrating %d ACCELEROMETER TRACES [%s] ..." %
                            (len(accel_stream), _ch))
                for _tr in accel_stream:
                    _tr.integrate(method='cumtrapz')
    #
    return prst


def normalizeTrace(workarr, rangeVal=[-1, 1]):
    ''' This simple method will normalize the trace between rangeVal.
        Simply by scaling everything...
        *** INPUT MUST BE an iterable

    '''
    in_is_nparr = True
    if not isinstance(workarr, np.ndarray):
        in_is_nparr = False
        workarr = np.asarray(workarr)
    #
    minVal = np.min(workarr)
    maxVal = np.max(workarr)
    workarr = np.asarray([((x - minVal) / (maxVal - minVal)) *
                         (rangeVal[1] - rangeVal[0]) for x in workarr])
    workarr = workarr + rangeVal[0]
    #
    if in_is_nparr:
        return workarr
    else:
        return list(workarr)


def filterStreamTrace(st, filtType, f1, f2, corners, zerophase):
    ''' Wrapper for filter waveform in a stream/trace object

        NB: from v0.7.34 the stream is filtered trace-by-trace
        in order to avoid possible error (quick fix)
    '''
    if filtType in ['bandstop', 'bandpass']:
        if not f1 or not f2:
            raise QE.InvalidVariable()
        #
        for _tr in st:
            try:
                _tr.filter(
                    filtType,
                    freqmin=f1,
                    freqmax=f2,
                    corners=corners,
                    zerophase=zerophase)
            except ValueError as err:
                # Probably a channel metadata error (i.e: sampling freq)
                logger.error(err)
                continue

    elif filtType in ['highpass', 'lowpass']:
        if not f1 and f2:
            raise QE.InvalidVariable()
        #
        for _tr in st:
            try:
                _tr.filter(filtType,
                           freq=f1,
                           corners=corners,
                           zerophase=zerophase)
            except ValueError as err:
                # Probably a channel metadata error (i.e: sampling freq)
                logger.error(err)
                continue
    else:
        raise QE.InvalidVariable()
    #
    return st


def smooth(x, window_len=11, window='hanning'):
    """ Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def wiener_filter(st,
                  pickt,
                  channel="*Z",
                  noise_window=None,
                  signal_window=None,
                  buffer_time=0.0,
                  debug_plot=False,
                  copy=False):
    """ Simple implementation of the Wiener Filtering method, it tryes
        to remove the noise spectrum from the real signal spectrum.
    """
    if copy:
        st = st.copy()
    #
    tr = st.select(channel=channel)[0]

    _in_size = tr.data.size
    #
    if not noise_window and not signal_window:
        noise = tr.slice(tr.stats.starttime, pickt-buffer_time)
        signal = tr.slice(pickt+buffer_time, tr.stats.endtime)
    elif noise_window and signal_window:
        noise = tr.slice(pickt-buffer_time-noise_window, pickt-buffer_time)
        signal = tr.slice(pickt+buffer_time, pickt+buffer_time+signal_window)
    elif not noise_window and signal_window:
        noise = tr.slice(tr.stats.starttime, pickt-buffer_time)
        signal = tr.slice(pickt+buffer_time, pickt+buffer_time+signal_window)
    elif noise_window and not signal_window:
        noise = tr.slice(pickt-buffer_time-noise_window, pickt-buffer_time)
        signal = tr.slice(pickt+buffer_time, tr.stats.endtime)
    else:
        raise QE.InvalidParameter("Something wrong in slicing selection!")
    # Closest nfft
    # if not nfft_value:
    nfft_value, cut_fft, maxin = QU.common_power_of_two((noise.data.size,
                                                         signal.data.size,
                                                         tr.data.size),
                                                        increase_exp=0)

    # print(nfft_value, cut_fft, maxin )
    # nfft_value, cut_fft = QU.nearest_power_of_two(
    #         np.max([signal.data.size, noise.data.size]), increase_exp=1)

    # Noise FFT
    noise.detrend('demean')
    noise.taper(max_percentage=0.05, type='cosine')
    noise_fft = fft(noise.data, nfft_value)
    noise_psd = (np.abs(noise_fft)**2)/nfft_value

    # Signal FFT
    signal.detrend('demean')
    signal.taper(max_percentage=0.05, type='cosine')
    signal_fft = fft(signal.data, nfft_value)
    signal_psd = (np.abs(signal_fft)**2)/nfft_value

    # Trace FFT
    tr.detrend('demean')
    tr.taper(max_percentage=0.05, type='cosine')
    tr_fft = fft(tr.data, nfft_value)   # NFFT VALUE MUST BE HIGHER THAN tr.data
    fax = fftfreq(int(nfft_value), d=tr.stats.delta)

    # ==== Freq
    # WienerFilter = (PSD_signalnoise - PSD_noise) / PSD_signalnoise
    # ifft(WienerFilter x FFT(trace))
    W = (signal_psd - noise_psd) / signal_psd  # PSD
    out_spectr = W*tr_fft
    out_spectr[0] = 0.0   # forced tapering removing the dc
    out_spectr[-1] = 0.0  # forced tapering the dc

    # Prepare output/normalize
    out_signal = ifft(out_spectr)[0:maxin]

    if debug_plot:
        print(nfft_value, cut_fft, maxin)
        plt.figure(figsize=(12, 8))
        norm_fac = 1  # nfft_value

        # ----
        plt.subplot(2, 1, 1)
        _tmp = ifft(out_spectr)/norm_fac  # full ifft
        _size_diff = _tmp.size - tr.times("matplotlib").size
        _init = tr.stats.starttime
        _end = tr.stats.endtime + (_size_diff * tr.stats.delta)
        _tmp_times = np.linspace(_init.matplotlib_date,
                                 _end.matplotlib_date,
                                 _size_diff+tr.stats.npts)
        plt.plot(_tmp_times, _tmp/norm_fac, 'r')

        #
        plt.plot(tr.times("matplotlib"), tr.data/norm_fac, '--g',
                 label="data")
        plt.plot(tr.times("matplotlib"), out_signal/norm_fac, 'k',
                 label="filtered")
        plt.axvline(pickt.datetime, label="pick")
        plt.legend()

        # ----
        plt.subplot(2, 1, 2)
        plt.semilogy(fax[0:cut_fft], signal_psd[0:cut_fft], 'g',
                     label="signal")
        plt.semilogy(fax[0:cut_fft], noise_psd[0:cut_fft], 'r',
                     label="noise")
        plt.semilogy(fax[0:cut_fft], W[0:cut_fft], 'gold',
                     label="wiener_filter")
        plt.semilogy(fax[0:cut_fft], out_spectr[0:cut_fft], 'k',
                     label="out_spectr")
        plt.grid()
        plt.legend()

        #
        plt.show()
    #
    if _in_size == out_signal.size:
        tr.data = out_signal.real
    else:
        raise QE.InvalidVariable("Input and Filtered Trace differs in size!")
    #
    return st


def wrap_wiener_filter(st,
                       pickt,
                       channel="*Z",
                       trimtrace_signal=None,
                       trimtrace_noise=None,
                       noise_window=None,
                       signal_window=None,
                       buffer_time=0.0,
                       debug_plot=False,
                       copy=False):
    """ Wrapper for calling the wiener_filter function """
    # Cut trace
    if copy:
        wst = st.copy
    else:
        wst = st
    #
    if trimtrace_noise or trimtrace_signal:
        if trimtrace_noise and trimtrace_signal:
            wst.trim(pickt-trimtrace_noise, pickt+trimtrace_signal)
        else:
            raise QE.MissingVariable(
                "Need a trimtrace_signal AND trimtrace_noise vals!")
    #
    outst = wiener_filter(st, pickt, channel=channel,
                          noise_window=noise_window,
                          signal_window=signal_window,
                          buffer_time=buffer_time,
                          debug_plot=debug_plot,
                          copy=False)
    return outst
