import os
import sys
import pprint
from obspy import read
from obspy import UTCDateTime
#
import adapt.utils as QU
from adapt.database import PickContainer
from adapt.picks.analysis import Judge
from adapt import errors as QE


_tmp_pick = QU.loadPickleObj('./tests_data/test_analysis_pd.pkl')
ST = read('./tests_data/KP201606231437.A366A.A362A.stream')


def load_and_process():
    """Utility function to load and filter the trace"""
    raw = ST
    proc = raw.copy()
    proc.detrend('demean')
    proc.detrend('simple')
    proc.taper(max_percentage=0.05, type='cosine')
    proc.filter("bandpass",
                freqmin=1,
                freqmax=30,
                corners=2,
                zerophase=True)
    return proc, raw

def test_judge_init():
    errors = []
    # KP201606231437 / A366A A362A
    pst, rst = load_and_process()
    pd = PickContainer.from_dictionary(_tmp_pick)
    try:
        jud = Judge(pd,
                    ('A366A',),
                    pst,
                    rst,
                    channel="*Z",
                    extract_picktime="median",
                    extract_pickerror='std',
                    weigther_dict={},
                    polarizer_dict={},
                    evaluator_dict={},
                    storing_pick_tag="JusticeForAll")
    except (TypeError, AttributeError):
        errors.append("Judge Class not correctly initialized")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_judge_deliberate_1():
    errors = []
    # KP201606231437 / A366A
    pst, rst = load_and_process()
    pd = PickContainer.from_dictionary(_tmp_pick)
    statlist = ('A366A', 'A362A')

    # Work
    jud = Judge(pd,
                statlist,
                pst,
                rst,
                channel="*Z",
                extract_picktime="median",
                extract_pickerror='std',
                weigther_dict={'analysis_key': ('BAIT_EVAL',
                                                'BK_ROUND1',
                                                'BAIT_ROUND1',
                                                'MyAIC_ROUND1'),
                               'wintest_thr': 0.3,
                               'interphase_thr': 0.6},
                polarizer_dict={'definition_method': "conservative",
                                'sec_after_pick': 0.1,
                                'use_raw': False},
                evaluator_dict={},
                storing_pick_tag="JusticeForAll")

    jud.deliberate()
    final_dict = jud.get_final_pd()

    # ---------- Checks
    if len(final_dict) != len(statlist):
        errors.append("Final Pick dimensions don't match the length of " +
                      "length of station queries")

    # Error
    if (final_dict['A366A']['JusticeForAll'][0]['pickerror'] !=
       0.05188129034558818):
        errors.append("Final Pick error of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickerror'])

    if (final_dict['A362A']['JusticeForAll'][0]['pickerror'] !=
       0.011180329225731478):
        errors.append("Final Pick error of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickerror'])

    # PickTime
    if (final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 56, 135000)):
        errors.append("Pick Time of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    if (final_dict['A362A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 39, 1, 920000)):
        errors.append("Pick Time of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    # Polarity
    if (final_dict['A366A']['JusticeForAll'][0]['pickpolar'] != "-"):
        errors.append("Polarity of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickpolar'])

    if (final_dict['A362A']['JusticeForAll'][0]['pickpolar'] != "-"):
        errors.append("Polarity of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickpolar'])

    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_judge_deliberate_1a():
    errors = []
    # KP201606231437 / A366A
    pst, rst = load_and_process()
    pd = PickContainer.from_dictionary(_tmp_pick)
    statlist = ('A366A',)

    # Work
    jud = Judge(pd,
                statlist,
                pst,
                rst,
                channel="*Z",
                extract_picktime="median",
                extract_pickerror='std',
                weigther_dict={'analysis_key': ('BAIT_EVAL',
                                                'BK_ROUND1',
                                                'BAIT_ROUND1',
                                                'MyAIC_ROUND1'),
                               'wintest_thr': 0.3,
                               'interphase_thr': 0.6},
                polarizer_dict={'definition_method': "conservative",
                                'sec_after_pick': 0.1,
                                'use_raw': False},
                evaluator_dict={},
                storing_pick_tag="JusticeForAll")

    jud.deliberate()
    final_dict = jud.get_final_pd()

    # ---------- Checks
    if len(final_dict) != len(statlist):
        errors.append("Final Pick dimensions don't match the length of " +
                      "length of station queries")

    # Error
    if (final_dict['A366A']['JusticeForAll'][0]['pickerror'] !=
       0.05188129034558818):
        errors.append("Final Pick error of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickerror'])

    # PickTime
    if (final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 56, 135000)):
        errors.append("Pick Time of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    # Polarity
    if (final_dict['A366A']['JusticeForAll'][0]['pickpolar'] != "-"):
        errors.append("Polarity of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickpolar'])

    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_judge_deliberate_2():
    errors = []
    # KP201606231437 / A366A
    pst, rst = load_and_process()
    pd = PickContainer.from_dictionary(_tmp_pick)
    statlist = ('A366A', 'A362A')

    jud = Judge(pd,
                statlist,
                pst,
                rst,
                channel="*Z",
                extract_picktime="mean",
                extract_pickerror='var',
                weigther_dict={'analysis_key': ('BAIT_EVAL',
                                                'BK_ROUND1',
                                                'BAIT_ROUND1',
                                                'MyAIC_ROUND1'),
                               'wintest_thr': 0.3,
                               'interphase_thr': 0.6},
                polarizer_dict={'definition_method': "simple",
                                'sec_after_pick': 0.1,
                                'use_raw': False},
                evaluator_dict={},
                storing_pick_tag="JusticeForAll")

    jud.deliberate()
    final_dict = jud.get_final_pd()

    # ---------- Checks
    if len(final_dict) != len(statlist):
        errors.append("Final Pick dimensions don't match the length of " +
                      "length of station queries")

    # Error
    if (final_dict['A366A']['JusticeForAll'][0]['pickerror'] !=
       0.0026916682879232212):
        errors.append("Final Pick error of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickerror'])

    if (final_dict['A362A']['JusticeForAll'][0]['pickerror'] !=
       0.00012499976159574544):
        errors.append("Final Pick error of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickerror'])

    # PickTime
    if (final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 56, 115000)):
        errors.append("Pick Time of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    if (final_dict['A362A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 39, 1, 925000)):
        errors.append("Pick Time of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    # Polarity
    if (final_dict['A366A']['JusticeForAll'][0]['pickpolar'] != "U"):
        errors.append("Polarity of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickpolar'])

    if (final_dict['A362A']['JusticeForAll'][0]['pickpolar'] != "-"):
        errors.append("Polarity of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickpolar'])

    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))

def test_judge_deliberate_3a_error():
    """ This test wants to change on the dictionary of the principal
        objects. See if they affact the final results
    """
    errors = []
    # KP201606231437 / A366A
    pst, rst = load_and_process()
    pd = PickContainer.from_dictionary(_tmp_pick)
    statlist = ('A366A',)

    # Work
    jud = Judge(pd,
                statlist,
                pst,
                rst,
                channel="*Z",
                extract_picktime="median",
                extract_pickerror='std',
                weigther_dict={'analysis_key': ('BAIT_EVAL',
                                                'BK_ROUND1',
                                                'BAIT_ROUND1',
                                                'MyAIC_ROUND1'),
                               'wintest_thr': 0.3,
                               'interphase_thr': 0.6},
                polarizer_dict={'definition_method': "conservative",
                                'sec_after_pick': 0.1,
                                'use_raw': False},
                evaluator_dict={},
                storing_pick_tag="JusticeForAll")

    jud.deliberate()
    median_std_conservative = jud.get_final_pd()

    # Change Error
    jud.set_extract_pickerror('var')
    jud.deliberate()
    median_var_conservative = jud.get_final_pd()

    # ---------- Checks
    if (len(median_std_conservative) != len(statlist) or
       len(median_var_conservative) != len(statlist)):
        errors.append("Final Pick dimensions don't match the length of " +
                      "length of station queries")

    # Error
    if (median_std_conservative['A366A']['JusticeForAll'][0]['pickerror'] !=
       0.05188129034558818):
        errors.append("Original error of A366A (mean) mismatch: %f - %f" %
            median_std_conservative['A366A']['JusticeForAll'][0]['pickerror'])

    if (median_var_conservative['A366A']['JusticeForAll'][0]['pickerror'] !=
       0.0026916682879232212):
        errors.append("Original error of A366A (var) mismatch: %f - %f" %
            median_std_conservative['A366A']['JusticeForAll'][0]['pickerror'])

    if (median_std_conservative['A366A']['JusticeForAll'][0]['timeUTC_pick'] !=
       median_var_conservative['A366A']['JusticeForAll'][0]['timeUTC_pick']):
        errors.append("PickTime changes after error method change: %f - %f" %
         (
        median_std_conservative['A366A']['JusticeForAll'][0]['timeUTC_pick'] !=
        median_var_conservative['A366A']['JusticeForAll'][0]['timeUTC_pick']))

    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_judge_deliberate_3b_timing():
    """ This test wants to change on the dictionary of the principal
        objects. See if they affact the final results
    """
    errors = []
    # KP201606231437 / A366A
    pst, rst = load_and_process()
    pd = PickContainer.from_dictionary(_tmp_pick)
    statlist = ('A366A',)

    jud = Judge(pd,
                statlist,
                pst,
                rst,
                channel="*Z",
                extract_picktime="median",
                extract_pickerror='std',
                weigther_dict={'analysis_key': ('BAIT_EVAL',
                                                'BK_ROUND1',
                                                'BAIT_ROUND1',
                                                'MyAIC_ROUND1'),
                               'wintest_thr': 0.3,
                               'interphase_thr': 0.6},
                polarizer_dict={'definition_method': "conservative",
                                'sec_after_pick': 0.1,
                                'use_raw': False},
                evaluator_dict={},
                storing_pick_tag="JusticeForAll")

    jud.deliberate()
    median_std_conservative = jud.get_final_pd()

    # Change Timing
    jud.set_extract_picktime('mean')
    jud.deliberate()
    mean_std_conservative = jud.get_final_pd()

    # ---------- Checks
    if (len(median_std_conservative) != len(statlist) or
       len(mean_std_conservative) != len(statlist)):
        errors.append("Final Pick dimensions don't match the length of " +
                      "length of station queries")

    if (median_std_conservative['A366A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 56, 135000)):
        errors.append("Original picktime of A366A mismatch: %s" %
        median_std_conservative['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    if (mean_std_conservative['A366A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 56, 115000)):
        errors.append("Original picktime of A366A (var) mismatch: %f - %f" %
        mean_std_conservative['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    if (median_std_conservative['A366A']['JusticeForAll'][0]['pickerror'] !=
       mean_std_conservative['A366A']['JusticeForAll'][0]['pickerror']):
        errors.append("Error changes after error method change: %f - %f" %
         (
        median_std_conservative['A366A']['JusticeForAll'][0]['pickerror'] !=
        mean_std_conservative['A366A']['JusticeForAll'][0]['pickerror']))
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_judge_deliberate_3c_polarity():
    """ This test wants to change on the dictionary of the principal
        objects. See if they affact the final results
    """
    errors = []
    # KP201606231437 / A366A
    pst, rst = load_and_process()
    pd = PickContainer.from_dictionary(_tmp_pick)
    statlist = ('A366A',)

    # Work

    jud = Judge(pd,
                statlist,
                pst,
                rst,
                channel="*Z",
                extract_picktime="median",
                extract_pickerror='std',
                weigther_dict={'analysis_key': ('BAIT_EVAL',
                                                'BK_ROUND1',
                                                'BAIT_ROUND1',
                                                'MyAIC_ROUND1'),
                               'wintest_thr': 0.3,
                               'interphase_thr': 0.6},
                polarizer_dict={'definition_method': "conservative",
                                'sec_after_pick': 0.1,
                                'use_raw': False},
                evaluator_dict={},
                storing_pick_tag="JusticeForAll")

    jud.deliberate()
    median_std_conservative = jud.get_final_pd()

    # Change Polarity
    jud.set_polarizer_dict({'definition_method': "simple",
                            'sec_after_pick': 0.1,
                            'use_raw': False})
    jud.deliberate()
    median_std_simple = jud.get_final_pd()

    # ---------- Checks
    if (len(median_std_conservative) != len(statlist) or
       len(median_std_simple) != len(statlist)):
        errors.append("Final Pick dimensions don't match the length of " +
                      "length of station queries")

    if (median_std_conservative['A366A']['JusticeForAll'][0]['pickpolar'] !=
       "-"):
        errors.append("Original picktime of A366A mismatch: %s" %
        median_std_conservative['A366A']['JusticeForAll'][0]['pickpolar'])

    if (median_std_simple['A366A']['JusticeForAll'][0]['pickpolar'] !=
       "U"):
        errors.append("Original picktime of A366A (var) mismatch: %f - %f" %
        median_std_simple['A366A']['JusticeForAll'][0]['pickpolar'])

    if (median_std_conservative['A366A']['JusticeForAll'][0]['pickerror'] !=
       median_std_simple['A366A']['JusticeForAll'][0]['pickerror']):
        errors.append("Error changes after error method change: %f - %f" %
         (
        median_std_conservative['A366A']['JusticeForAll'][0]['pickerror'] !=
        median_std_simple['A366A']['JusticeForAll'][0]['pickerror']))
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_judge_deliberate_4_fail_evaluation():
    """ This test wants to change on the dictionary of the principal
        objects. See if they affact the final results
    """
    errors = []
    # KP201606231437 / A366A
    pst, rst = load_and_process()
    pd = PickContainer.from_dictionary(_tmp_pick)
    statlist = ('A366A',)

    # Work

    jud = Judge(pd,
                statlist,
                pst,
                rst,
                channel="*Z",
                extract_picktime="median",
                extract_pickerror='std',
                weigther_dict={'analysis_key': ('BAIT_EVAL',
                                                'BK_ROUND1',
                                                'BAIT_ROUND1',
                                                'MyAIC_ROUND1'),
                               'wintest_thr': 0.3,
                               'interphase_thr': 0.6},
                polarizer_dict={'definition_method': "conservative",
                                'sec_after_pick': 0.1,
                                'use_raw': False},
                evaluator_dict={'functions_dict': {
                                    'max_signal2noise_ratio': {
                                          'noise_window': 0.5,
                                          'signal_window': 0.25,
                                          'threshold': 5.0,
                                          'use_raw': False,
                                          'normalize': False,
                                          'debug_plot': False}}},
                storing_pick_tag="JusticeForAll")

    jud.deliberate()
    median_std_conservative_eval = jud.get_final_pd()

    # ---------- Checks
    if median_std_conservative_eval['A366A']['JusticeForAll'][0]['evaluate']:
        errors.append("Evaluation of pick is failing ...")

    assert not errors, "Errors occured:\n{}".format("\n".join(errors))

# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================
# =====================================================================


def test_judge_deliberate_1_changePickerKeys():
    errors = []
    # KP201606231437 / A366A
    pst, rst = load_and_process()
    pd = PickContainer.from_dictionary(_tmp_pick)
    statlist = ('A366A', 'A362A')

    # Change keys of KURT and AIC
    pd["A366A"]["AIC_ROUND1"] = pd["A366A"].pop("MyAIC_ROUND1")

    # Change keys of KURT and AIC
    pd["A362A"]["AIC_ROUND1"] = pd["A362A"].pop("MyAIC_ROUND1")

    # Work
    jud = Judge(pd,
                statlist,
                pst,
                rst,
                channel="*Z",
                extract_picktime="median",
                extract_pickerror='std',
                weigther_dict={'analysis_key': ('BAIT_EVAL',
                                                'BK_ROUND1',
                                                'BAIT_ROUND1',
                                                'AIC_ROUND1'),
                               'wintest_thr': 0.3,
                               'interphase_thr': 0.6},
                polarizer_dict={'definition_method': "conservative",
                                'sec_after_pick': 0.1,
                                'use_raw': False},
                evaluator_dict={},
                storing_pick_tag="JusticeForAll")

    jud.deliberate()
    final_dict = jud.get_final_pd()

    # ---------- Checks
    if len(final_dict) != len(statlist):
        errors.append("Final Pick dimensions don't match the length of " +
                      "length of station queries")

    # Error
    if (final_dict['A366A']['JusticeForAll'][0]['pickerror'] !=
       0.05188129034558818):
        errors.append("Final Pick error of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickerror'])

    if (final_dict['A362A']['JusticeForAll'][0]['pickerror'] !=
       0.011180329225731478):
        errors.append("Final Pick error of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickerror'])

    # PickTime
    if (final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 56, 135000)):
        errors.append("Pick Time of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    if (final_dict['A362A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 39, 1, 920000)):
        errors.append("Pick Time of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    # Polarity
    if (final_dict['A366A']['JusticeForAll'][0]['pickpolar'] != "-"):
        errors.append("Polarity of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickpolar'])

    if (final_dict['A362A']['JusticeForAll'][0]['pickpolar'] != "-"):
        errors.append("Polarity of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickpolar'])

    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_judge_deliberate_2_changePickerKeys():
    errors = []
    # KP201606231437 / A366A
    pst, rst = load_and_process()
    pd = PickContainer.from_dictionary(_tmp_pick)
    statlist = ('A366A', 'A362A')

    # Change keys of KURT and AIC
    pd["A366A"]["AIC_ROUND1"] = pd["A366A"].pop("MyAIC_ROUND1")

    # Change keys of KURT and AIC
    pd["A362A"]["AIC_ROUND1"] = pd["A362A"].pop("MyAIC_ROUND1")

    jud = Judge(pd,
                statlist,
                pst,
                rst,
                channel="*Z",
                extract_picktime="mean",
                extract_pickerror='var',
                weigther_dict={'analysis_key': ('BAIT_EVAL',
                                                'BK_ROUND1',
                                                'BAIT_ROUND1',
                                                'AIC_ROUND1'),
                               'wintest_thr': 0.3,
                               'interphase_thr': 0.6},
                polarizer_dict={'definition_method': "simple",
                                'sec_after_pick': 0.1,
                                'use_raw': False},
                evaluator_dict={},
                storing_pick_tag="JusticeForAll")

    jud.deliberate()
    final_dict = jud.get_final_pd()

    # ---------- Checks
    if len(final_dict) != len(statlist):
        errors.append("Final Pick dimensions don't match the length of " +
                      "length of station queries")

    # Error
    if (final_dict['A366A']['JusticeForAll'][0]['pickerror'] !=
       0.0026916682879232212):
        errors.append("Final Pick error of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickerror'])

    if (final_dict['A362A']['JusticeForAll'][0]['pickerror'] !=
       0.00012499976159574544):
        errors.append("Final Pick error of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickerror'])

    # PickTime
    if (final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 56, 115000)):
        errors.append("Pick Time of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    if (final_dict['A362A']['JusticeForAll'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 39, 1, 925000)):
        errors.append("Pick Time of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['timeUTC_pick'])

    # Polarity
    if (final_dict['A366A']['JusticeForAll'][0]['pickpolar'] != "U"):
        errors.append("Polarity of A366A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickpolar'])

    if (final_dict['A362A']['JusticeForAll'][0]['pickpolar'] != "-"):
        errors.append("Polarity of A362A mismatch: %f" %
                      final_dict['A366A']['JusticeForAll'][0]['pickpolar'])

    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
