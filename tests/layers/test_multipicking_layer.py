import pprint
import numpy.testing as npt
from obspy import read, UTCDateTime
import adapt.utils as QU
import adapt.processing as QPR
from adapt.layers import MultipickingLayer

# In this file are contained all the tests for the multipicking layers


def _compare_floats_dictionaries(testdict, refdict, decimal=10):
    """ Use numpy assert_almost_equal for float comparison inside dicts
    """
    asserr = []
    for _k, _v in testdict.items():
        if isinstance(_v, (float, int)):
            try:
                npt.assert_almost_equal(
                  _v, refdict[_k], decimal=decimal)
            except AssertionError:
                asserr.append((_k, _v))
    #
    return asserr


def test_init():
    conf = QU.get_adapt_config("./tests_data/test_config_layer_v041.yml")
    st_raw = read()
    st_proc = QPR.processStream(st_raw,
                                copystream=True,
                                **conf)
    #
    hope = MultipickingLayer.Hive(
            'EQ0001',
            'test_event',
            st_proc,
            st_raw,
            associated_time_UTC=None,
            associated_time_tag=None,
            associated_time_attribute_dict=None,
            multipicking_dict=None,
            multipicking_tag="MP")


def test_simpleWorkflow():
    errors = []

    conf = QU.get_adapt_config("./tests_data/test_config_layer_v041.yml")
    st_raw = read()
    st_proc = QPR.processStream(st_raw,
                                copystream=True,
                                **conf)

    # Init
    mp_dict_complete = QU.get_adapt_config(conf['MULTIPICKING_LAYER_P1'],
                                       check_version=False)
    collect_mp_dict = {_k: _v for _k, _v in mp_dict_complete.items()
                       if _k != 'JUDGER_PICK'}
    collect_judge_dict = mp_dict_complete['JUDGER_PICK']

    hope = MultipickingLayer.Hive(
            'EQ0001',
            'test_event',
            st_proc,
            st_raw,
            associated_time_UTC=UTCDateTime("2009-08-24T00:20:07.7"),
            associated_time_tag="BAIT_LAYER",
            multipicking_dict=collect_mp_dict,
            multipicking_tag="MP")

    # 0) run layerclass
    hope.pick()

    # ================================================ Median EVALUATION

    # 1) define pick --> with JudgerClass
    hope.define_pick(collect_judge_dict)

    # 2) store picks
    tmpd = hope.get_final_pick_dict()
    tpn = tuple(tmpd.keys())[0]
    tpd = tmpd[tpn]["finalPick"][0]           # Taking only the first pick

    # Start checking
    if tpd['timeUTC_pick'] != UTCDateTime(2009, 8, 24, 0, 20, 7, 705000):
        errors.append("Final pickTime is not correct MEDIAN")
    if tpd['pickpolar'] != 'D':
        errors.append("Polarity is not correct MEDIAN --> %s" %
                      tpd['pickpolar'])
    try:
        npt.assert_almost_equal(tpd['pickerror'], 0.048669863517337028,
                                decimal=8)
    except AssertionError:
        errors.append("PickError is not correct MEDIAN: %f" % tpd['pickerror'])

    # # # ================================================ Mean EVALUATION

    collect_judge_dict["extract_picktime"] = "mean"

    # 1) define pick
    hope.define_pick(collect_judge_dict)

    # 2) store picks
    tmpd = hope.get_final_pick_dict()
    tpn = tuple(tmpd.keys())[0]
    tpd = tmpd[tpn]["finalPick"][0]

    # --- Start checking
    # old host: (< 2.1.1) UTCDateTime(2009, 8, 24, 0, 20, 7, 688750):
    # ['pickerror'] != 0.051097360681798758
    if tpd['timeUTC_pick'] != UTCDateTime(2009, 8, 24, 0, 20, 7, 692500):
        errors.append("Final pickTime is not correct MEAN")
    if tpd['pickpolar'] != 'D':
        errors.append("Polarity is not correct MEAN --> %s" %
                      tpd['pickpolar'])
    try:
        npt.assert_almost_equal(tpd['pickerror'], 0.048669863517337028,
                                decimal=8)
    except AssertionError:
        errors.append("PickError is not correct MEAN: %f" % tpd['pickerror'])
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))

def test_adding_associated_pick():
    errors = []

    conf = QU.get_adapt_config("./tests_data/test_config_layer_v041.yml")
    st_raw = read()
    st_proc = QPR.processStream(st_raw,
                                copystream=True,
                                **conf)

    # Init
    hope = MultipickingLayer.Hive(
            'EQ0001',
            'test_event',
            st_proc,
            st_raw,
            associated_time_UTC=UTCDateTime("2009-08-24T00:20:07.7"),
            associated_time_tag="BAIT_LAYER",
            associated_time_attribute_dict={
                    'pickpolar': "U",
                    'timeUTC_pick': UTCDateTime("1987-03-02T00:00:00")},
            multipicking_dict=QU.get_adapt_config(
                                    conf['MULTIPICKING_LAYER_P1'],
                                    check_version=False),
            multipicking_tag="MP")

    # Work
    hope._add_associated_pick()
    money = hope.get_work_pick_dict()

    # ================================================ Test
    if money['RJOB']['BAIT_LAYER'][0]['pickpolar'] != "U":
        errors.append("Attribute dict not correctly expanded")
    if (money['RJOB']['BAIT_LAYER'][0]['timeUTC_pick'] !=
       UTCDateTime("2009-08-24T00:20:07.7")):
        errors.append("Layer_AssociatedTimePick not correctly stored")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))



















