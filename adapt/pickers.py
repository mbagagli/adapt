import logging
import numpy as np
from obspy.signal.trigger import pk_baer
# Utils
from obspy import UTCDateTime
from adapt.processing import normalizeTrace
# Error and Checks
from obspy.core.stream import Stream
from obspy.core.trace import Trace
import adapt.errors as QE
import adapt.plot as QP

logger = logging.getLogger(__name__)


# =========================================
# ================  Utils  ================
# =========================================


def _AICcf(td, win=None):
    """
    This method will return the index of the minimum AIC carachteristic
    function.

    td must be a  `numpy.ndarray`
    """
    # --------------------  Creation of the carachteristic function
    # AIC(k)=k*log(variance(x[1,k]))+(n-k-1)*log(variance(x[k+1,n]))
    AIC = np.array([])
    for ii in range(1, len(td)):
        with np.errstate(divide='raise'):
            try:
                var1 = np.log(np.var(td[0:ii]))
            except FloatingPointError:  # if var==0 --> log is -inf
                var1 = 0.00
            #
            try:
                var2 = np.log(np.var(td[ii:]))
            except FloatingPointError:  # if var==0 --> log is -inf
                var2 = 0.00
        #
        val1 = ii*var1
        val2 = (len(td)-ii-1)*var2
        AIC = np.append(AIC, (val1+val2))
    # -------------------- New idx search (avoid window's boarders)
    # (ascending order min->max) OK!
    idx = sorted(range(len(AIC)), key=lambda k: AIC[k])[0]

    # --- OLD (here for reference)
    # idxLst = sorted(range(len(AIC)), key=lambda k: AIC[k])
    # if idxLst[0]+1 not in (1, len(AIC)):  # need index. start from 1
    #     idx = idxLst[0]+1
    # else:
    #     idx = idxLst[1]+1

    # --- REALLY OLD idx search
    # idx_old=int(np.where(AIC==np.min(AIC))[0])+1
    # ****   +1 order to make multiplications
    # **** didn't take into account to minimum at the border of
    # **** the searching window
    return idx, AIC


def _sec2sample(fs, **kwargs):
    """
    Utility method to define convert USER input parameter (seconds)
    into obspy pickers 'n_sample' units.

    Python3 round(float)==int // Python2 round(float)==float
    BETTER USE: int(round(... to have compatibility

    Formula: int(round( INSEC * df))
    *** NB: in sec could be float
    """
    outargs = {}
    for ii in kwargs.keys():
        outargs[ii] = int(round(kwargs[ii] * fs))
    return outargs

# =========================================
# ===============  Pickers  ===============
# =========================================


def fp_wrap(st, fppar=None, channel="*Z"):
    """
    Wrapper around the FilterPicker of A.Lomax

    Default parameter for the FilterPicker are:

    - filter_window=3
    - longterm_window=5
    - t_up=0.2
    - threshold_1=10
    - threshold_2=10
    - base=2

    At the moment this function returns only the FIRST ARRIVAL
    of the picker

    """

    # Check picker installation
    try:
        from filterpicker import filterpicker as FP
    except ImportError:
        logger.error("The FILTERPICKER picker package is not installed !!!")
        return None, None, None

    if not isinstance(st, Stream):
        logger.error("Input trace is not a valid ObsPy stream: %s" %
                     type(st))
        raise TypeError

    # Select channel/data
    tr = st.select(channel=channel)[0]
    ts = tr.data
    dt = tr.stats.delta
    #
    if fppar and isinstance(fppar, dict):
        myfp = FP.FilterPicker(dt, ts, **{_kk: _vv
                                          for _kk, _vv in fppar.items()
                                          if _kk not in ('debugplot')})
        pickTime_relative, pickUnc, pickBand = myfp.run()
        # NB: the previous vars are list!
        try:
            if fppar['debugplot']:
                myfp.plot()
        except KeyError:
            pass
    else:
        logger.error("Picker's parameter is not a valid dict")
        raise QE.InvalidVariable()
    # Closing
    if pickTime_relative.size:
        pickTime_UTC = tr.stats.starttime + pickTime_relative[0]
        return pickTime_UTC, pickUnc[0], pickBand[0]
    else:
        return None, None, None


def BaerKradolfer(st, bkparam=None):
    """
    Method to run obspy BaerKradolfer picker on pre-defined slices.
    tdownmax,tupevent,thr1,thr2,preset_len,p_dur.

    Bit more detail:

    tupevent: should be the inverse of high-pass freq or
              low freq in bandpass
    tdownmax: Half of tupevent
    p_dur:    time-interval in which MAX AMP is evaluated (SN qual)

    :type bkpar: dict
    :type df: float
    """
    if not isinstance(st, Stream):
        logger.error("Input trace is not a valid ObsPy stream: %s" %
                     type(st))
        raise TypeError
    # Extract  needed trace (Z component)
    tr = st.select(channel="*Z")[0]
    tr.taper(type='hann', max_percentage=0.05)  # MB: v0.4.1
    #
    if bkparam and isinstance(bkparam, dict):
        df = float(tr.stats.sampling_rate)
        threshold1 = bkparam["thr1"]   # need to save this before
        threshold2 = bkparam["thr2"]   # need to save this before
        par_sample = _sec2sample(df, **bkparam)
        try:
            PickSample, PhaseInfo, CF = pk_baer(tr.data, df,
                                                par_sample["tdownmax"],
                                                par_sample["tupevent"],
                                                threshold1,
                                                threshold2,
                                                par_sample["preset_len"],
                                                par_sample["p_dur"],
                                                return_cf=True)
        except KeyError:
            logger.error("Wrong BaerKradolfer parameter (keys)")
            logger.debug("%r" % bkparam)
            return None, None

        # If NO PICK found, BK returns PickSample==1 and PickInfo=''
        if PickSample != 1 and PhaseInfo != '':
            logger.debug("BK idx: %r - info: %r" % (PickSample, PhaseInfo))
            # convert pick from samples back to seconds and adj info
            # (Absolute from first sample)
            PickTime = PickSample / df
            PickTimeUTC = tr.stats.starttime + PickTime
            PhaseInfo = str(PhaseInfo).strip()
            return PickTimeUTC, PhaseInfo, CF, PickSample
        else:
            return None, None, CF, PickSample
    else:
        return None, None, CF, PickSample


def baitwrap(st, stream_raw=None, channel="*Z", **kwargs):
    """
    Simple wrapper build to keep things neat on the code.
    This method will call bait picking algorithm
    """

    # Check picker installation
    try:
        from bait.bait import BaIt
    except ImportError:
        logger.error("The BAIT picker package is not installed !!!")
        return None, None, None

    # Initializeclass
    itbk = BaIt(st, stream_raw=stream_raw, channel="*Z",
                **{_kk: _vv
                   for _kk, _vv in kwargs.items()
                    if _kk not in ("outpickersel",
                                   "pidxsel")})
    # MB: to see that a new instance is created everytime and class is resetted
    if itbk.baitdict:
        logger.error("BaIt istance is not correctly resetted, double check!")
        logger.error(itbk.baitdict)
        raise QE.BadInstance()
    #
    itbk.CatchEmAll()
    # Extract the first valid pick encountered --> return None,None already
    pick, pick_info = itbk.getTruePick(
                 idx=kwargs["pidxsel"],
                 picker=kwargs["outpickersel"],
                 compact_format=True)
    return pick, pick_info, itbk


def MyAIC(st, channel="*Z"):
    """
    This method is defining an AIC picker. It will use th one contained
    into the AUREM package due also to the faster C-implementation
    to detect the right on-phase timing of a phase
    on a time-window defined by wintrim (seconds) before
    and after the pick. The AIC picker is much more precise than
    Baer to detect the right sample.

    OUT:
        pickTime_UTC, AIC, idx

    REFERENCES:
    - Kalkan, E. (2016). "An automatic P-phase arrival time picker",
      Bull. of Seismol. Soc. of Am., 106, No. 3,
      doi: 10.1785/0120150111
    - Akaike, H. (1974). A new look at the statistical model
      identification, Trans. Automat. Contr. 19, no. 6, 716–723,
      doi: 10.1109/TAC.1974.1100705

    """

    # Check picker installation
    try:
        from aurem.pickers import AIC
    except ImportError:
        logger.error("The AUREM-AIC picker package is not installed !!!")
        return None, None, None

    if not isinstance(st, Stream):
        logger.error("Input trace is not a valid ObsPy stream: %s" %
                     type(st))
        raise TypeError
    # Run AUREM-AIC
    aicobj = AIC(st, channel=channel)
    aicobj.work()
    pickTime_UTC = aicobj.get_pick()
    aicfun = aicobj.get_aic_function()
    try:
        idx = aicobj.get_pick_index()
    except TypeError:
        logger.debug("AIC returned wrong type index ... setting to 0!")
        idx = 0
    #
    logger.debug("AIC pick/idx: %s / %d" % (pickTime_UTC, idx))
    return pickTime_UTC, aicfun, idx


def HOS(st,
        time_win=1.0,
        mode="kurtosis",
        channel="*Z",
        thresh=2.0,
        picksel="aic",  # "diff"/"AIC" from v031
        transform_dict={},
        debugplot=False):
    """
    High Order Statistic pickers --> Skewness, Kurtosis

    Trying to implement it as described in Chapter16_rev1 pp.22-24

    tw = moving timewindow length in sec () / or list, tuple

    mode can be "kurt","k","kurtosis" / "skew","s","skewness"

    picksel can be "gauss"/"diff" or "aic" to have the method for picking
    selection after CF transformation

    The thresh parameter is needed only if picksel=="diff".
    The thresh parameter is the num of times the value of diff need to
    exceed the standard deviation.

    return also the eval function that should represent the method adopted

    NB: Since 08/2019 this function use the package HOST for picking.
    NB: A the moment the picker is returning the median!

    ADAPT v0.7.0: the HOST package is updated to v2.1.1 due to fast CF
                  calucation implemented in C under the hood.

    """

    # Check picker installation
    try:
        from host.picker import Host
    except ImportError:
        logger.error("The HOST picker package is non installed !!!")
        return None, None, None, None

    _tr = st.select(channel=channel)[0]

    HOSTobj = Host(_tr,
                   time_win,
                   hos_method=mode,
                   transform_cf=transform_dict,
                   detection_method=picksel)
    #
    HOSTobj.work(debug_plot=debugplot)
    # The next instances are dicts
    pickTime_UTC = HOSTobj.get_picks_UTC()
    hos_arr = HOSTobj.get_HOS()
    eval_fun = HOSTobj.get_eval_functions()
    hos_idx = HOSTobj.get_picks_index()
    # HOS return
    return pickTime_UTC['median'], hos_arr, eval_fun, hos_idx


# ==== for HOST version 1.1.1 <=== Just for reference
# def HOSold(
#         st, time_win=1.0, mode="kurtosis",
#         channel="*Z",
#         thresh=2.0,
#         picksel="aic",  # "diff"/"AIC" from v031
#         transform_dict={},
#         debugplot=False):

#     High Order Statistic pickers --> Skewness, Kurtosis

#     Trying to implement it as described in Chapter16_rev1 pp.22-24

#     tw = moving timewindow length in sec () / or list, tuple

#     mode can be "kurt","k","kurtosis" / "skew","s","skewness"

#     picksel can be "gauss"/"diff" or "aic" to have the method for picking
#     selection after CF transformation

#     The thresh parameter is needed only if picksel=="diff".
#     The thresh parameter is the num of times the value of diff need to
#     exceed the standard deviation.

#     return also the eval function that should represent the method adopted

#     NB: Since 08/2019 this function use the package HOST for picking.
#     NB: A the moment the picker is returning the median!


#     HOSTobj = Host(st,
#                    time_win,
#                    channel=channel,
#                    hos_method=mode,
#                    transform_cf=transform_dict,
#                    detection_method=picksel,
#                    diff_gauss_thresh=thresh)
#     #
#     HOSTobj.work(debug_plot=debugplot)
#     # The next instances are dicts
#     pickTime_UTC = HOSTobj.get_picks_UTC()
#     hos_arr = HOSTobj.get_HOS()
#     eval_fun = HOSTobj.get_eval_functions()
#     hos_idx = HOSTobj.get_picks_index()
#     # HOS return
#     return pickTime_UTC['median'], hos_arr, eval_fun, hos_idx
