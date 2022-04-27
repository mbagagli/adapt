import pprint
from obspy import UTCDateTime
from adapt.database import PickContainer
import adapt.utils as QU
from adapt.picks.weight import Weighter
import numpy.testing as npt

_tmp_pick = QU.loadPickleObj("./tests_data/test_weight_pd.pkl")


def _custom_compare_dictionaries(testdict, refdict, decimal=10):
    """ This method will check both float and order/uniqueness of
    list/tuple by sorting them
    """
    asserr = []
    for _k, _v in testdict.items():
        if isinstance(_v, (float, int)):
            try:
                npt.assert_almost_equal(
                  _v, refdict[_k], decimal=decimal)
            except AssertionError:
                asserr.append((_k, _v))
        elif isinstance(_v, (list, tuple)):
            if sorted(_v) != sorted(refdict[_k]):
                asserr.append((_k, _v))
    #
    return asserr


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


def test_weighter_init():
    """
    Create the class and check parameters.
    """
    pd = PickContainer.from_dictionary(_tmp_pick)
    # pprint.pprint(pd["CADS"])
    valuable = Weighter(pd["CADS"],
                        analysis_key=("MP1", "BAIT_EVAL_1"),
                        wintest_thr=0.3,
                        interphase_thr=0.6)


def test_weighter_triage_1():
    """
    Analyze the class dinamics
    """
    errors = []
    # --- RESULTS
    comp_tp = {'failed_pickers': ['BK_1', 'BK_2', 'FP_1', 'FP_2', 'AIC_1', 'AIC_2'],
               'outliers': [],
               'pickers_involved': [],  # NO ['HOS'] because less then 3 valid obs
               'spare_obs': [('BAIT_1', 1454280284.99)],
               'stable_pickers': [('HOS', 1454280284.2649999, 0.0)],
               'tot_obs': 3,
               'valid_obs': [('HOS_1', 1454280284.2649999),
                             ('HOS_2', 1454280284.2649999)]}

    pd = PickContainer.from_dictionary(_tmp_pick)
    valuable = Weighter(pd["CADS"],
                        ("MP1", "BAIT_EVAL_1"),
                        wintest_thr=0.3,
                        interphase_thr=0.6)
    valuable.triage()

    if not valuable.get_triage_dict() == comp_tp:
        errors.append("TRIAGE DICT don't match")

    if valuable.get_triage_results():
        errors.append("RESULT DICT returned, not possible")

    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_weighter_triage_2():
    """
    Analyze the class dinamics
    """
    errors = []
    # --- RESULTS

    comp_tp = {'BK_AIC_1': 0.01999974250793457,
               'BK_AIC_2': 0.07999992370605469,
               'BK_FP_1': 0.0,
               'BK_FP_2': 0.04999995231628418,
               'BK_HOS_1': 0.009999990463256836,
               'BK_HOS_2': 0.04999995231628418,
               'FP_AIC_1': 0.01999974250793457,
               'FP_AIC_2': 0.029999971389770508,
               'FP_HOS_1': 0.009999990463256836,
               'FP_HOS_2': 0.0,
               'HOS_AIC_1': 0.009999752044677734,
               'HOS_AIC_2': 0.029999971389770508,
               'failed_pickers': [],
               'outliers': [],
               'pickers_involved': ['AIC', 'BAIT', 'BK', 'FP', 'HOS'],
               'spare_obs': [('BAIT_1', 1454280261.2700002)],
               'stable_pickers': [('BK', 1454280261.315, 0.04999995231628418),
                               ('FP', 1454280261.29, 0.0),
                               ('HOS', 1454280261.2849998, 0.009999990463256836),
                               ('AIC', 1454280261.2750003, 0.010000228881835938)],
               'tot_obs': 9,
               'valid_obs': [('BK_1', 1454280261.29),
                          ('BK_2', 1454280261.34),
                          ('FP_1', 1454280261.29),
                          ('FP_2', 1454280261.29),
                          ('HOS_1', 1454280261.28),
                          ('HOS_2', 1454280261.29),
                          ('AIC_1', 1454280261.2700002),
                          ('AIC_2', 1454280261.26),
                          ('BAIT_1', 1454280261.2700002)]}


    comp_tr = {'mean': UTCDateTime(2016, 1, 31, 22, 44, 21, 286667),
               'mean-median': -0.003333,
               'median': UTCDateTime(2016, 1, 31, 22, 44, 21, 290000),
               'mode': UTCDateTime(2016, 1, 31, 22, 44, 21, 290000),
               'modealt': UTCDateTime(2016, 1, 31, 22, 44, 21, 290000),
               'std': 0.021602407516798438,
               'var': 0.00046666401052182965,
               'aad': 0.014814747704399956,
               'mad': 0.0099999904632568359,
               'mmd': 0.079999923706054688,
               'bootmean': UTCDateTime(2016, 1, 31, 22, 44, 21, 285500),
               'bootmode': UTCDateTime(2016, 1, 31, 22, 44, 21, 290000),
               'bootmadmode': 0.009999990463256836
               }
    #
    pd = PickContainer.from_dictionary(_tmp_pick)
    valuable = Weighter(pd["A365A"],
                        ("MP1", "BAIT_EVAL_1"),
                        wintest_thr=0.3,
                        interphase_thr=0.6)
    valuable.triage()

    if not valuable.get_triage_dict() == comp_tp:
        errors.append("TRIAGE DICT don't match")
    if not valuable.get_triage_results() == comp_tr:
        errors.append("TRIAGE RESULTS don't match")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_weighter_triage_jk():
    """
    Analyze the class dinamics with the JACK KNIFE TRIAGE
    """
    errors = []
    # --- RESULTS

    comp_tp = {'BK_AIC_1': 0.01999974250793457,
               'BK_AIC_2': 0.07999992370605469,
               'BK_FP_1': 0.0,
               'BK_FP_2': 0.04999995231628418,
               'BK_HOS_1': 0.009999990463256836,
               'BK_HOS_2': 0.04999995231628418,
               'FP_AIC_1': 0.01999974250793457,
               'FP_AIC_2': 0.029999971389770508,
               'FP_HOS_1': 0.009999990463256836,
               'FP_HOS_2': 0.0,
               'HOS_AIC_1': 0.009999752044677734,
               'HOS_AIC_2': 0.029999971389770508,
               'failed_pickers': [],
               'outliers': [],
               'pickers_involved': ['AIC', 'BAIT', 'BK', 'FP', 'HOS'],
               'spare_obs': [],
               'stable_pickers': [],
               'tot_obs': 9,
               'valid_obs': [('BAIT_1', 1454280261.2700002),
                             ('BK_1', 1454280261.29),
                             ('BK_2', 1454280261.34),
                             ('FP_1', 1454280261.29),
                             ('FP_2', 1454280261.29),
                             ('HOS_1', 1454280261.28),
                             ('HOS_2', 1454280261.29),
                             ('AIC_1', 1454280261.2700002),
                             ('AIC_2', 1454280261.26)]}

    comp_tr = {'aad': 0.014814747704399956,
               'bootmadmode': 0.009999990463256836,
               'bootmean': UTCDateTime(2016, 1, 31, 22, 44, 21, 285500),
               'bootmode': UTCDateTime(2016, 1, 31, 22, 44, 21, 290000),
               'mad': 0.009999990463256836,
               'mean': UTCDateTime(2016, 1, 31, 22, 44, 21, 286667),
               'mean-median': -0.003333,
               'median': UTCDateTime(2016, 1, 31, 22, 44, 21, 290000),
               'mmd': 0.07999992370605469,
               'mode': UTCDateTime(2016, 1, 31, 22, 44, 21, 290000),
               'modealt': UTCDateTime(2016, 1, 31, 22, 44, 21, 290000),
               'std': 0.021602407516798438,
               'var': 0.00046666401052182965}
    #
    pd = PickContainer.from_dictionary(_tmp_pick)
    valuable = Weighter(pd["A365A"],
                        ("MP1", "BAIT_EVAL_1"),
                        wintest_thr=0.3,
                        interphase_thr=0.6)
    valuable.triage_jk()

    if not valuable.get_triage_dict() == comp_tp:
        errors.append("TRIAGE DICT don't match")
    if not valuable.get_triage_results() == comp_tr:
        errors.append("TRIAGE RESULTS don't match")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
