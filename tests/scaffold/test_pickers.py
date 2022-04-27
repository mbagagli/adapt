import pprint
from obspy import read, UTCDateTime
#
import adapt.pickers as QPK



def test_BaerKradolfer_yestaper():
    """ Test the picker """
    errors = []
    st = read()
    #
    PickTimeUTC, PhaseInfo, CF, _idx = QPK.BaerKradolfer(
                                                  st,
                                                  {'tdownmax': 0.5167,  # sec
                                                   'tupevent': 0.1,     # sec
                                                   'thr1': 10,           # 10
                                                   'thr2': 20,           # 20
                                                   'preset_len': 0.15,   # sec
                                                   'p_dur': 1.0}         # sec
                                                  )
    if not PickTimeUTC == UTCDateTime("2009-08-24T00:20:03.390000"):
        errors.append("Returned pick TIMES doesn't match")
    if not PhaseInfo == "IPD0":
        errors.append("Returned pick INFOS doesn't match")
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_HOS_kurtosis_aic_single_tw():
    """ Test the picker """
    errors = []
    st = read()
    st.detrend('demean')
    st.detrend("simple")
    st.filter("bandpass", freqmin=1.0, freqmax=30)
    #
    PickTimeUTC, hos_arr, hos_AIC, _idx = QPK.HOS(
                  st, time_win=0.7,
                  mode="kurtosis",
                  transform_dict={'transform_f2': {},
                                  'transform_smooth': {'window_type': 'hanning'}},
                  picksel="aic",
                  channel="*Z",
                  thresh=0.075,
                  debugplot=False)
    #
    pprint.pprint(PickTimeUTC)
    # old HOST
    # if not PickTimeUTC == UTCDateTime(2009, 8, 24, 0, 20, 8, 80000):
    if not PickTimeUTC == UTCDateTime(2009, 8, 24, 0, 20, 7, 610000):
        errors.append("Returned pick TIMES doesn't match")
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_HOS_kurtosis_aic_multi_tw():
    """ Test the picker """
    errors = []
    st = read()
    st.detrend('demean')
    st.detrend("simple")
    st.filter("bandpass", freqmin=1.0, freqmax=30)
    st.trim(UTCDateTime("2009-08-24T00:20:06.500000"),
            UTCDateTime("2009-08-24T00:20:08.500000"))
    #
    PickTimeUTC, hos_arr, hos_AIC, _idx = QPK.HOS(
                  st, time_win=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                  mode="kurtosis",
                  transform_dict={'transform_f2': {},
                                  'transform_smooth': {'window_type': 'hanning'}},
                  picksel="aic",
                  channel="*Z",
                  thresh=0.075,
                  debugplot=False)
    # MB:
    # PickTimeUTC, hos_arr, hos_AIC, _idx --> are all DICT.
    #   -PickTimeUTC contains two keys for final pick: 'mean'/'median'
    # v0.8
    if not PickTimeUTC == UTCDateTime(2009, 8, 24, 0, 20, 7, 695000):
        errors.append("Returned pick TIMES doesn't match")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_HOS_kurtosis_diff():
    """ Test the picker """
    errors = []
    st = read()
    st.detrend('demean')
    st.detrend("simple")
    st.filter("bandpass", freqmin=1.0, freqmax=30)
    #
    PickTimeUTC, hos_arr, hos_AIC, _idx = QPK.HOS(
                  st, time_win=0.74975,
                  mode="kurtosis",
                  transform_dict={'transform_f2': {},
                                  'transform_smooth': {'window_type': 'hanning'}},
                  picksel=("diff", 0.75),
                  channel="*Z",
                  debugplot=False)
    # v0.8
    if not PickTimeUTC == UTCDateTime(2009, 8, 24, 0, 20, 7, 420000):
        errors.append("Returned pick TIMES doesn't match")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
