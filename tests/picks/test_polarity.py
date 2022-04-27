import os
import sys
import pprint
from obspy import read
from obspy import UTCDateTime
#
import adapt.utils as QU
from adapt.picks.polarity import Polarizer

# ==========================================


def load_and_process():
    """Utility function to load and filter the trace"""
    rst = read('./tests_data/obspy_read.mseed')
    pst = rst.copy()
    pst.detrend('demean')
    pst.detrend('simple')
    pst.taper(max_percentage=0.05, type='cosine')
    pst.filter("bandpass",
               freqmin=1,
               freqmax=30,
               corners=2,
               zerophase=True)
    return pst, rst


def test_polarizer_init():
    errors = []
    #
    pst, rst = load_and_process()
    pickUTC = UTCDateTime()  # current date, just as test
    try:
        Polarizer(pst,
                  rst,
                  pickUTC,
                  channel="*Z",
                  definition_method="conservative",
                  sec_after_pick=0.1,
                  use_raw=False)
    except TypeError:
        errors.append("Polarizer Class not correctly initialized")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_polarizer_work_simple_1():
    errors = []
    #
    pst, rst = load_and_process()
    pickUTC = UTCDateTime("2009-08-24T00:20:07.7")
    myp = Polarizer(pst,
                    rst,
                    pickUTC,
                    channel="*Z",
                    definition_method="simple",
                    sec_after_pick=0.1)
    myp.work()
    pol = myp.get_polarity()
    if pol != "-":
        errors.append("Polarity with SIMPLE method fails")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_polarizer_work_simple_2():
    errors = []
    #
    pst, rst = load_and_process()
    pickUTC = UTCDateTime("2009-08-24T00:20:07.7")
    myp = Polarizer(pst,
                    rst,
                    pickUTC,
                    channel="*Z",
                    definition_method="simple",
                    sec_after_pick=0.04)
    myp.work()
    pol = myp.get_polarity()
    if pol != "D":
        errors.append("Polarity with SIMPLE method fails")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_polarizer_work_conservative_1():
    errors = []
    #
    pst, rst = load_and_process()
    pickUTC = UTCDateTime("2009-08-24T00:20:07.7")
    myp = Polarizer(pst,
                    rst,
                    pickUTC,
                    channel="*Z",
                    definition_method="conservative",
                    sec_after_pick=0.1)
    myp.work()
    pol = myp.get_polarity()
    if pol != "-":
        errors.append("Polarity with CONSERVATIVE method fails")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_polarizer_work_conservative_2():
    errors = []
    #
    pst, rst = load_and_process()
    pickUTC = UTCDateTime("2009-08-24T00:20:07.7")
    myp = Polarizer(pst,
                    rst,
                    pickUTC,
                    channel="*Z",
                    definition_method="conservative",
                    sec_after_pick=0.04)
    myp.work()
    pol = myp.get_polarity()
    if pol != "-":
        errors.append("Polarity with CONSERVATIVE method fails")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_polarizer_work_conservative_3():
    errors = []
    #
    pst, rst = load_and_process()
    pickUTC = UTCDateTime("2009-08-24T00:20:07.7")
    myp = Polarizer(pst,
                    rst,
                    pickUTC,
                    channel="*Z",
                    definition_method="conservative",
                    sec_after_pick=0.03)
    myp.work()
    pol = myp.get_polarity()
    if pol != "D":
        errors.append("Polarity with CONSERVATIVE method fails")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
