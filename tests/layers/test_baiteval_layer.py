import pprint
from obspy import read, UTCDateTime
import adapt.utils as QU
import adapt.processing as QPR
from adapt.layers import InitialBaitLayer

# In this file are contained all the tests for the InitialBait layer


def test_init():
    conf = QU.get_adapt_config("./tests_data/test_config_layer_v041.yml")
    st_raw = read()
    st_proc = QPR.processStream(st_raw,
                                copystream=True,
                                **conf)
    # Init
    hope = InitialBaitLayer.BaitBeautify(
                                    "testTAG",
                                    st_proc,
                                    st_raw,
                                    UTCDateTime("2009-08-24T00:20:07.7"),
                                    conf['BAIT_LAYER'],
                                    working_channel="*Z")


def test_simpleWorkflow():
    errors = []
    #
    conf = QU.get_adapt_config("./tests_data/test_config_layer_v041.yml")
    st_raw = read()
    st_proc = QPR.processStream(st_raw,
                                copystream=True,
                                **conf)
    # Init
    hope = InitialBaitLayer.BaitBeautify(
                                    "testTAG",
                                    st_proc,
                                    st_raw,
                                    UTCDateTime("2009-08-24T00:20:07.7"),
                                    conf['BAIT_LAYER'],
                                    working_channel="*Z")

    # Run layerclass
    hope.pickMe()
    # hope.itbk.plotPicks(show=True)

    # ========================================== Tests
    if (hope.extract_pick(0, "AIC")[0][0] !=
       UTCDateTime(2009, 8, 24, 0, 20, 7, 750000)):
        errors.append("P1 AIC not correct")

    if (hope.extract_pick(0, "BK")[0][0] !=
       UTCDateTime(2009, 8, 24, 0, 20, 7, 720000)):
        errors.append("P1 BK not correct")

    if (hope.extract_pick(1, "AIC")[0][0] !=
       UTCDateTime(2009, 8, 24, 0, 20, 8, 710000)):
        errors.append("P2 AIC not correct")

    if (hope.extract_pick(1, "BK")[0][0] !=
       UTCDateTime(2009, 8, 24, 0, 20, 8, 740000)):
        errors.append("P2 BK not correct")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
