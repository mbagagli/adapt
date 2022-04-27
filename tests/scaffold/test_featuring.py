from obspy import UTCDateTime
from adapt.database import PickContainer
import adapt.errors as QE
import adapt.utils as QU
import adapt.scaffold.featuring as QF
import pprint
import copy
import numpy.testing as npt
#
from obspy import read

_tmp_pick = QU.loadPickleObj('./tests_data/test_analysis_pd.pkl')

def load_and_process(rst):
    """Utility function to load and filter the trace"""
    pst = rst.copy()
    pst.detrend('demean')
    pst.detrend('simple')
    pst.taper(max_percentage=0.05, type='cosine')
    pst.filter("bandpass",
               freqmin=1,
               freqmax=30,
               corners=2,
               zerophase=True)
    return pst


def test_initialization():
    """ testinit method """
    errors = []
    rst = read("./tests_data/KP201606231437.A366A.A362A.stream")
    pst = load_and_process(rst)
    pd = PickContainer.from_dictionary(_tmp_pick)
    #
    try:
        miner = QF.Miner(rst,
                         pst,
                         pd,
                         phase_list=[('P1', 0),],
                         feature_dict={})
    except TypeError:
        errors.append("Miner class not correctly initialized")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_digging():
    """ testinit method """
    errors = []
    rst = read("./tests_data/KP201606231437.A366A.A362A.stream")
    pst = load_and_process(rst)
    pd = PickContainer.from_dictionary(_tmp_pick)
    #
    rst = rst.select(station="A366A")
    pst = pst.select(station="A366A")

    # Features
    fd = {'std_signal': {
               'use_raw': False,
               'usecf': False,
               'signal_window': 1.0,
               'buffer_window': 0.0}
          }

    #
    miner = QF.Miner(rst,
                     pst,
                     pd,
                     phase_list=[('BAIT_ROUND1', 0), ],
                     feature_dict=fd)

    miner.digging()
    od = miner.get_gold()

    if len(od.keys()) != 1:
        errors.append("Wrong dict length output")

    try:
        npt.assert_allclose(od['BAIT_ROUND1']['std_signal'],
                            2619.409145644714, rtol=1e-5, atol=0)  # atol=0.000001)
    except AssertionError:
        errors.append("Wrong std_signal output: %f - %f" % (
                                    od['BAIT_ROUND1']['std_signal'],
                                    2619.409145644714))
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_store_in_place():
    """ testinit method """
    errors = []
    rst = read("./tests_data/KP201606231437.A366A.A362A.stream")
    pst = load_and_process(rst)
    pickdict = PickContainer.from_dictionary(_tmp_pick)
    pd = copy.deepcopy(pickdict)
    #
    rst = rst.select(station="A366A")
    pst = pst.select(station="A366A")

    # Features
    fd = {'std_signal': {
               'use_raw': False,
               'usecf': False,
               'signal_window': 1.0,
               'buffer_window': 0.0}
          }

    #
    same_time = UTCDateTime('2016-06-23T14:38:56.140000')
    miner = QF.Miner(rst,
                     pst,
                     pd,
                     # phase_list=[('BAIT_ROUND1', 0), ],
                     feature_dict=fd,
                     pick_time=same_time)

    miner.digging()
    try:
        miner.store_in_place()
        errors.append("Store in place is DONE even pick_time param is given!")
    except QE.MissingAttribute:
        pass
    od = miner.get_gold()
    mykey = od.keys()
    #
    if len(mykey) != 1:
        errors.append("Wrong dict length output")

    if "SINGLE_TIME" not in mykey:
        errors.append("Keys are not set correctly! Keys stored: %s" % mykey)

    try:
        npt.assert_allclose(od['SINGLE_TIME']['std_signal'],
                            2619.409145644714, rtol=1e-5, atol=0)  # atol=0.000001)
    except AssertionError:
        errors.append("Wrong std_signal output: %f - %f" % (
                                    od['SINGLE_TIME']['std_signal'],
                                    2619.409145644714))
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
