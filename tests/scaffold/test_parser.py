import adapt.parser as QP
import adapt.database as QD
#
from obspy import UTCDateTime

TestDataDir = "./tests_data/"


def test_dict2Event():
    """ simple test """
    errors = []
    tmpDict = {'origintime': UTCDateTime(0),
               'eventscore': None,      # To be added
               'latGEO': '45.567N',     # Coordinates 'N'/'S' is [20]
               'lonGEO': '11.123E',     # Coordinates 'E'/'W' is [27]
               'lat': 45.567,           # Coordinates 'N'/'S' is [20]
               'lon': 11.123,           # Coordinates 'E'/'W' is [27]
               'dep': 15.0,             # km
               'mag': 3.5,
               'magType': 'Ml',
               'id': 'test',
               'eqlabel': 'tryerr',
               # 18-11-2019
               'rms': 0.15,
               'gap': 100.5}
    tmpEv = QP.dict2Event(tmpDict)
    # Multiple error checks
    if not tmpEv.origins[0].resource_id == "test":
        errors.append("Wrong ResourceID match")
    if not tmpEv.origins[0].time == UTCDateTime(1970, 1, 1, 0, 0):
        errors.append("Wrong OriginTime match")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_manuloc2Event():
    """
    Another test
    """
    errors = []
    #
    outEv = QP.manuloc2Event(TestDataDir + "KP201604071825.MANULOC",
                             "KP201604071825", "test")
    # Multiple error checks
    if not outEv.origins[0].resource_id == "KP201604071825":
        errors.append("Wrong ResourceID match")
    if not outEv.origins[0].time == UTCDateTime(
            2016, 4, 7, 18, 25, 58, 100000):
        errors.append("Wrong OriginTime match")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_manupick2PickContainer():
    """
    Testing the effectiveness of the function.

    PICKCONTAINER=  .eqid
                    .type
                    OrderedDict{STAT=dict{'PHASENAME'=[
                                    dict{'polarity':       (str)
                                         'onset':          (str)
                                         'weight':         (float)
                                         'class':          (int)
                                         'timeUTC_pick':   (UTCDateTime)
                                         'timeUTC_early':  (UTCDateTime)
                                         'timeUTC_late':   (UTCDateTime)
                                         }, ...
                                                      ]
                                      }
                                }
    """
    errors = []
    reftime = UTCDateTime("2016/04/07 18:24")
    #
    fakemetadata = QD.StatContainer_Event()
    pc, stalist = QP.manupick2PickContainer(
                                TestDataDir + "KP201604071825.MANUPICK",
                                "KP201604071825",
                                "test",
                                metastatdict=fakemetadata)

    # Multiple error checks
    if not stalist == ["A303A", "IMOL"]:
        errors.append(
            "Wrong station list match, should be alphabetically ordered")
    if not pc.eqid == "KP201604071825" or not pc.type == "test":
        errors.append("Wrong attribute match")
    if len(pc.keys()) != 2:
        errors.append("Wrong number of station list")
    if not len(pc["A303A"].keys()) == 2 or not len(pc["IMOL"].keys()) == 2:
        errors.append("Wrong station phase count match")
    # From v0.3.1 the default is Seiscomp_
    if not isinstance(pc["A303A"]["Seiscomp_P1"], list):
        errors.append("Phasename picks not 'list' object")
    if not pc["A303A"]["Seiscomp_P1"][0]["timeUTC_pick"] == reftime + 133.822:
        errors.append("Phasename UTCDateTime error")
    # Check after changing pickkeys 'class' --> 'pickclass' (07112018)
    for ii in pc.keys():
        for xx in pc[ii]:
            for ll in pc[ii][xx]:
                if ll["pickclass"] is None:
                    errors.append("Key-Change went wrong (pickclass)")
                    break
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
