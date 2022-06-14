import pprint
from adapt.database import PickContainer, StatContainer_Event, StatContainer
import adapt.utils as QU
import adapt.errors as QE
#
from obspy import read
from adapt.picks.phaser import Spock


def convert_StatMeta2StatCont(statmeta):
    SC = StatContainer(source_id="test", contains="seismometer", tagstr="test")
    for eqid in statmeta.keys():
        for _stat, _metadict in statmeta[eqid].items():
            for _mn, _mv in _metadict.items():
                SC.append_meta(_stat, eqid, _mn, _mv)
    return SC


def mini_process(rst):
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


RawStream = read('./tests_data/KP201606231437.A366A.A362A.stream')
ProcStream = mini_process(RawStream)

_tmp_pick = QU.loadPickleObj('./tests_data/KP201606231437.A313A.pick')
PD = PickContainer.from_dictionary(_tmp_pick)

EV = QU.loadPickleObj('./tests_data/AllCatalog_ObsPyCatalog.pkl')

_tmp_meta = QU.loadPickleObj('./tests_data/KP201606231437_EventMeta.pkl')

_sd = StatContainer_Event.from_dictionary(_tmp_meta)
SD = convert_StatMeta2StatCont(_sd)



def test_initialization():
    errors = []
    #
    """ Initialization test """
    try:
        star_treck = Spock(ProcStream, RawStream, PD, SD, EV,
                           stat_key_list=('A313A', ),
                           eventid_key_list=('KP201606231437', ),
                           functions_dict_outliers={},
                           functions_dict_associator={})
    except QE.InvalidType:
        errors.append("Spock class not correctly initialized")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_shootme_outliers():
    """ Running OUTLIERS test """
    errors = []
    #
    fd = {'quake_bait_difference': {'bait_idx': 0,
                                    'threshold': 1}
          }
    #
    star_treck = Spock(ProcStream, RawStream, PD, SD, EV,
                       stat_key_list=('A313A', ),
                       eventid_key_list=('KP201606231437', ),
                       phases_key_list=("P1", ),
                       functions_dict_outliers=fd)
    star_treck.shoot_outliers()
    res = star_treck.get_result_dict_outliers()

    star_treck._extract_event('cacca')

    #
    if not res:
        errors.append("RESULTS DICT not returned")
    if res['KP201606231437']['A313A']['P1']['quake_bait_difference']['result']:
        errors.append("Something went wrong in the test")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_shootme_outliers_changePickerNames():
    """ Running OUTLIERS test """
    errors = []
    #
    fd = {'quake_bait_difference': {'bait_idx': 0,
                                    'threshold': 1}
          }

    # Change keys of KURT and AIC
    PD['A313A']["HOS_MP1"] = PD['A313A'].pop("KURT_MP1")
    PD['A313A']["AIC_MP1"] = PD['A313A'].pop("MyAIC_MP1")

    star_treck = Spock(ProcStream, RawStream, PD, SD, EV,
                       stat_key_list=('A313A', ),
                       eventid_key_list=('KP201606231437', ),
                       phases_key_list=("P1", ),
                       functions_dict_outliers=fd)
    star_treck.shoot_outliers()
    res = star_treck.get_result_dict_outliers()

    star_treck._extract_event('cacca')

    #
    if not res:
        errors.append("RESULTS DICT not returned")
    if res['KP201606231437']['A313A']['P1']['quake_bait_difference']['result']:
        errors.append("Something went wrong in the test")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_shootme_outliers_predictionDelta():
    """ Running OUTLIERS test """
    errors = []
    #
    fd = {'quake_prediction_delta': {'threshold': 1,
                                     'prediction_time_idx': 0,
                                     'threshold': 1.0,
                                     'check_delays_of': None}
          }

    star_treck = Spock(ProcStream, RawStream, PD, SD, EV,
                       stat_key_list=('A313A', ),
                       eventid_key_list=('KP201606231437', ),
                       phases_key_list=("P1", ),
                       functions_dict_outliers=fd)
    star_treck.shoot_outliers()
    res = star_treck.get_result_dict_outliers()

    star_treck._extract_event('cacca')

    #
    if not res:
        errors.append("RESULTS DICT not returned")
    if not res['KP201606231437']['A313A']['P1']['quake_prediction_delta']['result']:
        errors.append("Something went wrong in the test")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
