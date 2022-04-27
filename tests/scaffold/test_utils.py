import adapt.utils as QU
import adapt.database as QD
from adapt.processing import normalizeTrace
import numpy as np
#
from obspy import read
from obspy import UTCDateTime


# --------------------------------------------------------

def test_getFirstPickPhase():
    """
    """
    errors = []
    #
    fake_pick = QD.PickContainer("test", "test1", "test2")
    fake_pick.addPick("RJOB", "test_P1",
                      timeUTC_pick=UTCDateTime("2009-08-24T00:20:03"))
    fake_pick.addPick("RJOB", "test_P2",
                      timeUTC_pick=UTCDateTime("2009-08-24T01:20:03"))
    #
    a = QU.getFirstPickPhase(fake_pick, "RJOB", searchkey="tes")
    if a != "test_P1":
        errors.append("wrong phase tag returned")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_getFirstPickSlice():
    """
    """
    errors = []
    #
    fake_pick = QD.PickContainer("test", "test1", "test2")
    fake_pick.addPick("RJOB", "test_P2",
                      timeUTC_pick=UTCDateTime("2009-08-24T01:20:03"))
    fake_pick.addPick("RJOB", "test_P2",
                      timeUTC_pick=UTCDateTime("2009-08-24T01:45:03"))
    fake_pick.addPick("RJOB", "test_P1",
                      timeUTC_pick=UTCDateTime("2009-08-24T00:20:03"))
    fake_pick.addPick("RJOB", "test_P1",
                      timeUTC_pick=UTCDateTime("2009-08-24T00:50:03"))
    fake_pick.addPick("RJOB", "Predicted_Pg",
                      timeUTC_pick=UTCDateTime(2016, 4, 25, 10, 28, 47, 8278))
    fake_pick.addPick("RJOB", "Predicted_PmP",
                      timeUTC_pick=UTCDateTime(2016, 4, 25, 10, 28, 48, 77597))
    fake_pick.addPick(
                    "RJOB",
                    "Predicted_Pn",
                    timeUTC_pick=UTCDateTime(2016, 4, 25, 10, 28, 47, 240390))
    # #
    ttaglist_one = QU.get_pick_slice(fake_pick, "RJOB",
                                     searchkey="^tes",
                                     phase_pick_indexnum=1,
                                     arrival_order="aLl")
    ttaglist_two = QU.get_pick_slice(fake_pick, "RJOB",
                                     searchkey="^Predicted_",
                                     phase_pick_indexnum=0,
                                     arrival_order=1)
    #
    if len(ttaglist_one) != 2:
        errors.append("returned length of 'all' option is wrong")
    if ttaglist_one[0][0] != "test_P1":
        errors.append("wrong phase tag returned ttaglist_one")
    if ttaglist_one[1][1] != UTCDateTime(2009, 8, 24, 1, 45, 3):
        errors.append("wrong phase time returned ttaglist_one")

    if len(ttaglist_two) != 1:
        errors.append("returned list length should be 1!")

    if ttaglist_two[0][0] != "Predicted_Pn":
        errors.append("wrong phase tag returned ttaglist_two")
    if ttaglist_two[0][1] != UTCDateTime(2016, 4, 25, 10, 28, 47, 240390):
        errors.append("wrong phase time returned ttaglist_two")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_normalize_trace():
    errors = []
    #
    tr = read("./tests_data/obspy_read.mseed")[0]
    prior = type(tr.data)
    stnorm = normalizeTrace(tr.data, rangeVal=[0, 1])
    after = type(stnorm)
    #
    if prior != after:
        errors.append("wrong type return")
    if np.min(stnorm) != 0.0 or np.max(stnorm) != 1.0:
        errors.append("wrong normalization return")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
