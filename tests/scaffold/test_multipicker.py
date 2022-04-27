from obspy import read, UTCDateTime
#
import adapt.pickers as QPK
from adapt.multipicker import MultiPicker
from adapt.database import StatContainer, PickContainer


def miniproc(st):
    prs = st.copy()
    prs.detrend('demean')
    prs.detrend('simple')
    prs.taper(max_percentage=0.05, type='cosine')
    prs.filter("bandpass",
               freqmin=1,
               freqmax=30,
               corners=2,
               zerophase=True)
    return prs


# ---------------------------------- MAINS

def test_sec2sample():
    """ Test the pickers module method """
    errors = []
    # Create fakes
    fake_st = read()
    fake_stat = StatContainer()
    fake_stat.addStat("RJOB", {"fullname": "RJOB",
                               "alias": None,
                               "network": "BW",
                               "lat": None,
                               "lon": None,
                               "elev_m": None})
    fake_pick = PickContainer("test", "test1", "test2")
    fake_pick.addPick("RJOB", "testphase",
                      timeUTC_pick=UTCDateTime("2009-08-24T00:20:03"))

    # Initialize class
    df = fake_st[0].stats.sampling_rate
    test_in = {"test1": 45.8,
               "test2": 1009.987}
    test_out = QPK._sec2sample(df, **test_in)
    #
    if not test_out.keys() == test_in.keys():
        errors.append("Returned keys doesn't match")
    if not test_out["test1"] == 4580 or not test_out["test2"] == 100999:
        errors.append("Wrong conversion")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_initializeclass():
    """ Test the Multipicker initialize method """
    errors = []
    # Create fakes
    fake_st = read()
    fake_stat = StatContainer()
    fake_stat.addStat("RJOB", {"fullname": "RJOB",
                               "alias": None,
                               "network": "BW",
                               "lat": None,
                               "lon": None,
                               "elev_m": None})
    fake_pick = PickContainer("test", "test1", "test2")
    fake_pick.addPick("RJOB", "testphase",
                      timeUTC_pick=UTCDateTime("2009-08-24T00:20:10"))
    # Initialize class
    try:
        QMPobj = MultiPicker("EQTEST",
                             "EQTAG",
                             "DICTTAG",
                             fake_st,
                             fake_st,
                             associatedUTC=None,
                             associatedTAG=None,
                             pickers=None,
                             slices_delta=None,
                             # enlargenoisewin=None,
                             # enlargesignalwin=None,
                             slices_container=None,
                             slices_container_raw=None,
                             pickEvalConfigDict=None)
    except TypeError as e:
        errors.append("Undefined input: %s" % e)
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_sliceItUp():
    """ Test the Multipicker class-helper module of """
    errors = []
    # Create fakes
    fake_st = read()
    fake_stat = StatContainer()
    fake_stat.addStat("RJOB", {"fullname": "RJOB",
                               "alias": None,
                               "network": "BW",
                               "lat": None,
                               "lon": None,
                               "elev_m": None})
    fake_pick = PickContainer("test", "test1", "test2")
    fake_pick.addPick("RJOB", "testphase",
                      timeUTC_pick=UTCDateTime("2009-08-24T00:20:10"))
    # Initialize class
    QMPobj = MultiPicker("EQTEST",
                         "EQTAG",
                         "DICTTAG",
                         fake_st,
                         fake_st)
    QMPobj.associatedTAG = "testphase"
    QMPobj.associatedUTC = UTCDateTime("2009-08-24T00:20:10")
    # v0.6.32 must be a list of iterable
    QMPobj.slices_delta = ((3.0, 3.0), (2.0, 2.0), (1.5, 1.5))

    # **** TEST CALL
    # avoid having it errorized for STA/LTA noise level detection
    try:
        QMPobj.pickableStream = True
        QMPobj.sliceItUp()
    except Exception as e:
        errors.append(e)

    # Test 1
    if not QMPobj.slices_container:
        errors.append("Not actually slicing")
    # Test 2
    try:
        QMPobj.slices_container
    except KeyError:
        errors.append("Uncorrect phases indexing for slicing")
    # Test 3
    if (not len(QMPobj.slices_container) ==
       len(QMPobj.slices_delta)):
        errors.append("asked and pursued number of slices differ")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_sliceItUp_RAW():
    """ Test the Multipicker class-helper module of """
    errors = []
    # Create fakes
    fake_st_raw = read()
    fake_st = miniproc(fake_st_raw)
    #
    fake_stat = StatContainer()
    fake_stat.addStat("RJOB", {"fullname": "RJOB",
                               "alias": None,
                               "network": "BW",
                               "lat": None,
                               "lon": None,
                               "elev_m": None})
    fake_pick = PickContainer("test", "test1", "test2")
    fake_pick.addPick("RJOB", "testphase",
                      timeUTC_pick=UTCDateTime("2009-08-24T00:20:10"))
    # Initialize class
    QMPobj = MultiPicker("EQTEST", "EQTAG", "DICTTAG",
                          fake_st, rawdatastream=fake_st_raw)
    QMPobj.associatedTAG = "testphase"
    QMPobj.associatedUTC = UTCDateTime("2009-08-24T00:20:10")
    # v0.6.32 must be a list of iterable
    QMPobj.slices_delta = ((3.0, 3.0), (2.0, 2.0), (1.5, 1.5))

    # **** TEST CALL
    # avoid having it errorized for STA/LTA noise level detection
    try:
        QMPobj.pickableStream = True
        QMPobj.sliceItUp()
    except Exception as e:
        errors.append(e)

    # Test 1
    if not QMPobj.slices_container:
        errors.append("Not actually slicing")
    # Test 2
    try:
        QMPobj.slices_container
    except KeyError:
        errors.append("Uncorrect phases indexing for slicing")
    # Test 3
    if (not len(QMPobj.slices_container) ==
       len(QMPobj.slices_delta)):
        errors.append("asked and pursued number of slices differ")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
