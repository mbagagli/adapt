import pprint
from obspy import UTCDateTime
from adapt import forward as FWD
from adapt.database import PickContainer
from adapt.utils import loadPickleObj


CAKEMODEL = "./tests_data/mymodel.nd"
STATDICT = loadPickleObj("./tests_data/KP201606231437.stat_dict.corr")
EVENT = loadPickleObj("./tests_data/KP201606231437.event")  # 10 km


def test_predict_phase_arrival():
    """ Define the prediction time of a certain phase """
    errors = []
    pick_dict = PickContainer.create_empty_container()
    pick_dict, rottenstat = FWD.PredictPhaseArrival_PYROCKOCAKE(
                                                EVENT,
                                                pick_dict,
                                                STATDICT,
                                                ['MAIM', 'GSCL', 'GRAM',
                                                 'A366A', 'A362A', 'A319A'],
                                                phaselist=["Pg", "Pn", "PmP"],
                                                vel1d_model=CAKEMODEL,
                                                use_stations_corrections=False)

    # Testing STATION ['A366A']
    if len(pick_dict['A366A']['Predicted_Pn']) > 1:
        errors.append("Too many Pn predictions on A366A!")
    if (pick_dict['A366A']['Predicted_Pn'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 51, 835586)):
        errors.append("Wrong predicted Pn on A366A")

    # Testing STATION ['A362A']
    if len(pick_dict['A362A']['Predicted_Pn']) > 1:
        errors.append("Too many Pn predictions on A362A!")
    if (pick_dict['A362A']['Predicted_Pn'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 58, 868825)):
        errors.append("Wrong predicted Pn on A362A")

    # Testing STATION ['A319A']
    if len(pick_dict['A319A']['Predicted_Pg']) > 1:
        errors.append("Too many Pg predictions on A319A!")
    if (pick_dict['A319A']['Predicted_Pg'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 10, 934679)):
        errors.append("Wrong predicted Pg on A319A")

    if len(pick_dict['A319A']['Predicted_Pn']) > 1:
        errors.append("Too many Pn predictions on A319A!")
    if (pick_dict['A319A']['Predicted_Pn'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 9, 620766)):
        errors.append("Wrong predicted Pn on A319A")

    # Testing STATION ['MAIM']
    if len(pick_dict['MAIM']['Predicted_Pg']) > 1:
        errors.append("Too many Pg predictions on MAIM!")
    if (pick_dict['MAIM']['Predicted_Pg'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 5, 397562)):
        errors.append("Wrong predicted Pg on MAIM")

    if len(pick_dict['MAIM']['Predicted_Pn']) > 1:
        errors.append("Too many Pn predictions on MAIM!")
    if (pick_dict['MAIM']['Predicted_Pn'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 5, 406003)):
        errors.append("Wrong predicted Pn on MAIM")

    if len(pick_dict['MAIM']['Predicted_PmP']) > 1:
        errors.append("Too many PmP predictions on MAIM!")
    if (pick_dict['MAIM']['Predicted_PmP'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 5, 763964)):
        errors.append("Wrong predicted PmP on MAIM")

    # Testing STATION ['GSCL']
    if len(pick_dict['GSCL']['Predicted_Pg']) > 1:
        errors.append("Too many Pg predictions on GSCL!")
    if (pick_dict['GSCL']['Predicted_Pg'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 7, 762167)):
        errors.append("Wrong predicted Pg on GSCL")

    if len(pick_dict['GSCL']['Predicted_Pn']) > 1:
        errors.append("Too many Pn predictions on GSCL!")
    if (pick_dict['GSCL']['Predicted_Pn'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 7, 205040)):
        errors.append("Wrong predicted Pn on GSCL")

    if len(pick_dict['GSCL']['Predicted_PmP']) > 1:
        errors.append("Too many PmP predictions on GSCL!")
    if (pick_dict['GSCL']['Predicted_PmP'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 7, 922834)):
        errors.append("Wrong predicted PmP on GSCL")

    # Testing STATION ['GRAM']
    if len(pick_dict['GRAM']['Predicted_Pg']) > 1:
        errors.append("Too many Pg predictions on GRAM!")
    if (pick_dict['GRAM']['Predicted_Pg'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 5, 218829)):
        errors.append("Wrong predicted Pg on GRAM")

    if len(pick_dict['GRAM']['Predicted_Pn']) > 1:
        errors.append("Too many Pn predictions on GRAM!")
    if (pick_dict['GRAM']['Predicted_Pn'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 5, 324339)):
        errors.append("Wrong predicted Pn on GRAM")

    if len(pick_dict['GRAM']['Predicted_PmP']) > 1:
        errors.append("Too many PmP predictions on GRAM!")
    if (pick_dict['GRAM']['Predicted_PmP'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 5, 632505)):
        errors.append("Wrong predicted PmP on GRAM")

    # Ensemble
    if len(pick_dict['A366A']) > 1 or len(pick_dict['A362A']) > 1:
        errors.append("Station A362A. Additional predicted phase, not possible"
                      " for given mod!")

    # FInal
    if rottenstat:
        errors.append("There are ROTTEN STAT! FATAL: it's not possible!")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_predict_phase_arrival_statcorr():
    """ Define the prediction time of a certain phase """
    errors = []

    STATDICT['A366A']["p_delay"] = 7.5
    STATDICT['A366A']["s_delay"] = 5.5
    STATDICT['A362A']["p_delay"] = -0.6
    STATDICT['A362A']["s_delay"] = 11.0

    pick_dict = PickContainer.create_empty_container()
    pick_dict, rottenstat = FWD.PredictPhaseArrival_PYROCKOCAKE(
                                                EVENT,
                                                pick_dict,
                                                STATDICT,
                                                ['A366A', 'A362A'],
                                                phaselist=["Pg", "Pn", "PmP"],
                                                vel1d_model=CAKEMODEL,
                                                use_stations_corrections=True)
    # Testing STATION ['A366A']
    if (pick_dict['A366A']['Predicted_Pn'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 59, 335586)):
        errors.append("Wrong predicted Pn on A366A")

    if (pick_dict['A362A']['Predicted_Pn'][0]['timeUTC_pick'] !=
       UTCDateTime(2016, 6, 23, 14, 38, 58, 268825)):
        errors.append("Wrong predicted Pn on A362A")

    if len(pick_dict['A366A']) > 1 or len(pick_dict['A362A']) > 1:
        errors.append("Additional predicted phase, not possible for given mod")

    if rottenstat:
        errors.append("There are ROTTEN STAT, and it's not possible!")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
