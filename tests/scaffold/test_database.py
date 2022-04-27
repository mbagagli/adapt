import os
import adapt.utils as QU
import adapt.database as TM
import adapt.errors as QE
#
from obspy import read_inventory, UTCDateTime


def test_createStationInventory():
    """
    Test function for the creation of an
    obspy Inventory class
    """
    errors = []
    fakeinv = read_inventory()
    fakeinv.write("./tests_data/TMP.xml", format='STATIONXML')
    CONF = QU.getQuakeConf("./tests_data/test_config.yml")
    testinv = TM.createStationInventory(**CONF)

    if not testinv == fakeinv:
        errors.append("Importing existing inventory")
    os.remove("./tests_data/TMP.xml")    # Netting

    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_buildInventory():
    """
    Test the core of the inventory building function
    """
    errors = []
    #
    CONF = QU.getQuakeConf("./tests_data/test_config.yml")
    testinv = TM.buildInventory(**CONF)
    if not len(testinv[:]) == 2:
        errors.append("Network appending error")
    tmpidx = next(
        _ii for _ii, net in enumerate(
            testinv[:]) if "ST" == net.code)
    if not len(testinv[tmpidx].stations) == 2:
        errors.append("Stations appending error")
    if (not os.path.exists("./tests_data/TMP.xml") and
       not os.path.getsize("./tests_data/TMP.xml") > 0):
        errors.append("Output saving error")
    os.remove("./tests_data/TMP.xml")    # Netting
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_createStationContainer():
    """
    Another test
    """
    errors = []
    attributes = ("fullname", "alias", "lat", "lon", "elev_m", "network")
    #
    CONF = QU.getQuakeConf("./tests_data/test_config.yml")
    # v0.5.1 Stein ==> xml type Pbspy Inventory option added
    outdict = TM.createStationContainer(**CONF)

    # TEST - 1 (import existent)
    testdict = QU.loadPickleObj("./tests_data/TMP_small.pkl")

    if testdict != outdict:
        errors.append("Stored and loaded dict differ")

    # TEST - 2 (statcount)
    if len(outdict.keys()) != 3:
        errors.append("Stations count doesn't match")

    # TEST - 3 (keyssaving)
    for stat in outdict.keys():
        for kk in attributes:
            try:
                outdict[stat][kk]
            except KeyError:
                errors.append("Key storing differs")

    # TEST - 4 (keysloading)
    for stat in testdict.keys():
        for kk in attributes:
            try:
                testdict[stat][kk]
            except KeyError:
                errors.append("Key loading differs")
    #
    os.remove("./tests_data/TMP_small.pkl")    # Netting
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_checkpickdict():
    """ Test the Multipicker class-helper module of """
    errors = []
    #
    fake_pick = TM.PickContainer("test", "test1", "test2")
    fake_pick.addPick("RJOB", "testphase",
                      timeUTC_pick=UTCDateTime("2009-08-24T00:20:03"))
    expect_dict = {'polarity': None,
                   'onset': None,
                   'weight': None,
                   'pickclass': None,
                   'pickerror': None,
                   'pickpolar': None,
                   'boot_obj': None,       # (quake.picks.weight.Bootstrap/None)
                   'timeUTC_pick': UTCDateTime("2009-08-24T00:20:03"),
                   'timeUTC_early': None,
                   'timeUTC_late': None,
                   'evaluate': None,        # (bool/None)
                   'evaluate_obj': None,
                   'outlier': None,         # (bool/None)
                   'phaser_obj': None,      # (quake.picks.phaser.Spock/None)
                   'features': None,        # (dict/None)
                   'features_obj': None,    # (quake.picks.evaluation.Gandalf/None)
                   'backup_picktime': None,  # (UTCDateTime)
                   'general_infos': None     # dict for general porpuses
                   }
    # Test 1
    if not len(fake_pick.keys()) == 1:
        errors.append("station keys differ")
    # Test 2
    for ll in fake_pick["RJOB"]["testphase"]:
        for kk in ll.keys():
            try:
                if expect_dict[kk] != ll[kk]:
                    errors.append("inner list dict differs")
            except KeyError:
                errors.append("missing keys")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_addPick():
    """ Test the Multipicker class-helper module of """
    errors = []
    #
    fake_pick = TM.PickContainer("test", "test1", "test2")
    fake_pick.addPick("RJOB", "testphase",
                      timeUTC_pick=UTCDateTime("2009-08-24T00:20:03"))
    fake_pick.addPick("RJOB", "testphase",
                      timeUTC_pick=UTCDateTime("1987-03-02T02:02:02"))
    # Test 1
    if not len(fake_pick.keys()) == 1:
        errors.append("station keys differ")
    # Test 2
    if not len(fake_pick["RJOB"]) == 1:
        errors.append("phase-key differs")
    # Test 3
    if not len(fake_pick["RJOB"]["testphase"]) == 2:
        errors.append("multipick phase-key differs")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_addStat():
    """ Test the Multipicker class-helper module of """
    errors = []
    #
    fake_pick = TM.PickContainer("test", "test1", "test2")

    fake_pick.addStat("RJOB", {"testphase": [{
                      'timeUTC_pick': UTCDateTime("2009-08-24T00:20:03")}]})
    fake_pick.addStat("RJOB", {"testphase": [{
                      'timeUTC_pick': UTCDateTime("1987-03-02T02:02:02")}]})
    # Test 1
    if not len(fake_pick.keys()) == 1:
        errors.append("station keys differ")
    # Test 2
    if not len(fake_pick["RJOB"]) == 1:
        errors.append("phase-key differs")
    # Test 3
    if not len(fake_pick["RJOB"]["testphase"]) == 2:
        errors.append("multipick phase-key differs")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_addStat_removeStat():
    """ Test the Multipicker class-helper module of """
    errors = []
    #
    fake_pick = TM.PickContainer("test", "test1", "test2")

    fake_pick.addStat("RJOB", {"testphase": [{
                      'timeUTC_pick': UTCDateTime("2009-08-24T00:20:03")}]})
    fake_pick.addStat("RJOB", {"testphase": [{
                      'timeUTC_pick': UTCDateTime("1987-03-02T02:02:02")}]})
    # Test 1
    if not len(fake_pick.keys()) == 1:
        errors.append("station keys differ")
    # Test 2
    if not len(fake_pick["RJOB"]) == 1:
        errors.append("phase-key differs")
    # Test 3
    if not len(fake_pick["RJOB"]["testphase"]) == 2:
        errors.append("multipick phase-key differs")

    # -- Now remove stat
    fake_pick.addStat("RJOB", {"testphase": [{
                      'timeUTC_pick': UTCDateTime("1992-08-17T02:02:02")}]},
                      overwrite_stat=True)
    # Test 1
    if not len(fake_pick.keys()) == 1:
        errors.append("station keys differ ... after removal")
    # Test 2
    if not len(fake_pick["RJOB"]) == 1:
        errors.append("phase-key differs ... after removal")
    # Test 3
    if not len(fake_pick["RJOB"]["testphase"]) == 1:
        errors.append("multipick phase-key differs ... after removal")
    # Test 4
    if (not fake_pick["RJOB"]["testphase"][0]['timeUTC_pick'] ==
       UTCDateTime("1992-08-17T02:02:02")):
        errors.append("pick times differs ... after removal")
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_addStat_removePhasePick():
    """ Test the Multipicker class-helper module of """
    errors = []
    #
    fake_pick = TM.PickContainer("test", "test1", "test2")

    fake_pick.addStat("RJOB", {"testphase": [{
                      'timeUTC_pick': UTCDateTime("2009-08-24T00:20:03")}]})
    fake_pick.addStat("RJOB", {"testphase": [{
                      'timeUTC_pick': UTCDateTime("1987-03-02T02:02:02")}]})
    # Test 1
    if not len(fake_pick.keys()) == 1:
        errors.append("station keys differ")
    # Test 2
    if not len(fake_pick["RJOB"]) == 1:
        errors.append("phase-key differs")
    # Test 3
    if not len(fake_pick["RJOB"]["testphase"]) == 2:
        errors.append("multipick phase-key differs")

    # -- Now remove phase

    fake_pick.addStat("RJOB", {
                      'testphase': [{'timeUTC_pick': UTCDateTime("1992-08-17T02:02:02")}],
                      'testphase_new': [{'timeUTC_pick': UTCDateTime("1989-11-09T00:00:00")}]
                      }, overwrite_stat=False, overwrite_phase=True)

    # Test 1
    if not len(fake_pick.keys()) == 1:
        errors.append("station keys differ ... after removal")
    # Test 2
    if not len(fake_pick["RJOB"]) == 2:
        errors.append("phase-key differs ... after removal")
    # Test 3
    if not len(fake_pick["RJOB"]["testphase"]) == 1:
        errors.append("multipick phase-key differs ... after removal")
    # Test 4
    if (not fake_pick["RJOB"]["testphase"][0]['timeUTC_pick'] ==
       UTCDateTime("1992-08-17T02:02:02")):
        errors.append("pick times differs ... after removal")
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_stationContainer_aliashelper():
    """
    Another test
    """
    errors = []
    #
    CONF = QU.getQuakeConf("./tests_data/test_config.yml")
    outdict = TM.createStationContainer(**CONF)

    # Set Alias
    try:
        outdict.set_alias("PYPD", "dedema")
        errors.append("Failing to raise error for ALIAS > 4 char")
    except QE.CheckError:
        pass

    outdict.set_alias("PYPD", "dede")
    _tmp = outdict.get_alias("PYPD")
    if not _tmp == "dede":
        errors.append("Failing to properly set ALIAS")

    try:
        outdict.set_alias("ZIAN", "dede")
        errors.append("Failing to raise error for ALIAS already present")
    except QE.CheckError:
        pass

    try:
        outdict.set_alias("ZIAN", "dede", force=True)
    except QE.CheckError:
        errors.append("Failing to raise error for ALIAS already present FORCED")

    try:
        _ = outdict.get_statname_from_alias("dede")
        errors.append("Failing to raise error for multiple ALIAS")
    except QE.CheckError:
        pass

    #
    os.remove("./tests_data/TMP_small.pkl")    # Netting
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
