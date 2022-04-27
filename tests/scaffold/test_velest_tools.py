import pprint
from pathlib import Path
#
import adapt.utils as QU
from adapt.scaffold import velest_tools as VT
from adapt.utils import loadPickleObj
import adapt.errors as QE
#
from obspy import UTCDateTime
import shutil

# --------------------------------------------------------


def test_remove_stat_obs():
    """
    Check if removing stations observations from CNV work
    """
    errors = []
    _ = VT.remove_stations_from_cnv(
                                    "./tests_data/velest_tools/mytest.cnv",
                                    statlst=["I164", "I185", "GU20"])
    difflist = QU.filediff(
        "./tests_data/velest_tools/mytest.cnv.statsfiltered",
        "./tests_data/velest_tools/mytest.remove_stat_obs.match.cnv",
        print_out=False)
    if difflist:
        errors.append("Processed file don't match with true test-results")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_hypres2dict():
    """Test the expansion of VELEST_4.5.1 *hypres file into a dictionary
    """
    errors = []
    #
    checkdict = loadPickleObj("./tests_data/velest_tools/mytest.hypres.pkl")
    resdict = VT.hypres2dict("./tests_data/velest_tools/mytest.hypres")
    # Checks
    if checkdict != resdict:
        errors.append("hypres dict differs!")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_stares2dict():
    """Test the expansion of VELEST_4.5.1 *stares file into a dictionary
    """
    errors = []
    #
    checkdict = loadPickleObj("./tests_data/velest_tools/mytest.hypres.pkl")
    resdict = VT.stares2dict("./tests_data/velest_tools/mytest.stares")
    # Checks
    if checkdict != resdict:
        errors.append("stares dict differs!")
        mismatch = {key for key in resdict.keys() &
                    checkdict if resdict[key] != checkdict[key]}
        pprint.pprint(mismatch)
    #

    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_reloc():
    """Test the class RELOC of ADAPT from VELEST_4.5.1 *reloc
    """
    errors = []
    #
    rel = VT.RELOC()
    pd = rel.get_pick_list()
    cat = rel.get_catalog()

    # Checks
    if pd or cat:
        errors.append("PickDict and Catalog are already imported, IMPOSSIBLE!")

    rel.import_file("./tests_data/velest_tools/mytest.reloc")
    pd = rel.get_pick_list()
    cat = rel.get_catalog()
    ogtim = UTCDateTime("20190114 13:37:58.296")  # 69.043

    # Checks
    if not pd or not cat:
        errors.append("PickDict and Catalog are NOT imported!")
    if pd[0]['I001']['P'][0]['timeUTC_pick'] != ogtim+69.043:
        errors.append("Traveltime not correctly acquired")
    if pd[0]['I001']['P'][0]['general_infos']['t_student_velest_sem'] != -12.507:
        errors.append("T-Student not correctly acquired")

    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_reloc_filtering_TSTU():
    """Test the class RELOC of ADAPT from VELEST_4.5.1 *reloc
    """
    errors = []
    #
    rel = VT.RELOC()
    rel.import_file("./tests_data/velest_tools/mytest.reloc")
    rel.filter_tstudent(tstu_thr=0.5)
    pd = rel.get_pick_list()
    cat = rel.get_catalog()

    if 'I001' in pd[0].keys():
        errors.append("Filtering not correct")
    if 'SL16' not in pd[0].keys():
        errors.append("Filtering not correct")
    if len(pd[0].keys()) != 9:
        errors.append("Filtering not correct: length of final stations differ")
    if len(cat) != 1:
        errors.append("Fake events appended. Must be skipped")

    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_VELEST_obj():
    """ This tests must be done with your installation path of VELEST """
    errors = []
    #
    work_dir = "/tests_data/velest_tools/test.run.vel"
    bin_path = "/usr/local/GEOPHYSICS/velest/4.5.2_MB/bin/velest_452_MB"
    reg_names = "/usr/local/GEOPHYSICS/velest/4.5.2_MB/regionsnamen.dat"
    reg_coord = "/usr/local/GEOPHYSICS/velest/4.5.2_MB/regionskoord.dat"

    vel = VT.VELEST()
    #
    try:
        vel.set_work_dir(work_dir)
        test = vel.get_work_dir()
        if not isinstance(test, Path):
            errors.append("WORK-DIR is not instantiated as Path object")
        #
        vel.set_bin_path(bin_path)
        test = vel.get_bin_path()
        if not isinstance(test, Path):
            errors.append("BIN-PATH is not instantiated as Path object")
        #
        vel.set_reg_names(reg_names)
        test = vel.get_reg_names()
        if not isinstance(test, Path):
            errors.append("REG-NAMES is not instantiated as Path object")
        #
        vel.set_reg_coord(reg_coord)
        test = vel.get_reg_names()
        if not isinstance(test, Path):
            errors.append("REG-NAMES is not instantiated as Path object")

        assert not errors, "Errors occured:\n{}".format("\n".join(errors))

    except QE.BadConfigurationFile:
        print("WARNING: missing or erroneous paths for executable (or) variables")


def test_VELEST_cmn():
    """ This tests must be done with your installation path of VELEST """
    errors = []
    #
    work_dir = "./tests_data/velest_tools/test.run.vel"
    work_cmn = "./tests_data/velest_tools/test.run.vel/velest.cmn"
    bin_path = "/usr/local/GEOPHYSICS/velest/4.5.2_MB/bin/velest_452_MB"
    mod_path = "/home/matteo/polybox/ETH/work/Sublime_SFTP_bigstar01/quake/tests_data/velest_tools/runvel.input.mod"
    cnv_path = "/home/matteo/polybox/ETH/work/Sublime_SFTP_bigstar01/quake/tests_data/velest_tools/runvel.input.cnv"
    sta_path = "/home/matteo/polybox/ETH/work/Sublime_SFTP_bigstar01/quake/tests_data/velest_tools/runvel.input.sta"
    #
    reg_names = "/usr/local/GEOPHYSICS/velest/4.5.2_MB/regionsnamen.dat"
    reg_coord = "/usr/local/GEOPHYSICS/velest/4.5.2_MB/regionskoord.dat"

    try:
        vel = VT.VELEST()
        vel.set_work_dir(work_dir)
        vel.set_cmn(work_cmn)
        vel.set_bin_path(bin_path)
        vel.set_cnv(cnv_path)
        vel.set_mod(mod_path)
        vel.set_sta(sta_path)
        vel.set_reg_names(reg_names)
        vel.set_reg_coord(reg_coord)
        vel.prepare()
    except QE.BadConfigurationFile:
        print("WARNING: missing or erroneous paths for executable (or) variables")
        return True

    # JHD
    vel.create_cmn('jhd')  # defaults must be specified
    de = vel.get_cmn_par()
    if str(de['isingle']) != "0":
        errors.append("Erroneous CMN-JHD created")

    # SEM
    vel.create_cmn('sem')  # defaults must be specified
    de = vel.get_cmn_par()
    if str(de['isingle']) != "1":
        errors.append("Erroneous CMN-SEM created")

    # Change-Par
    vel.change_cmn_par(othet=100.0)  # defaults must be specified
    de = vel.get_cmn_par()
    if str(de['othet']) != "100.0":
        errors.append("Erroneous CMN changes")

    try:
        vel.change_cmn_par(bar='foo')
        errors.append("Erroneous PARKEY not recognized!")
    except QE.InvalidVariable:
        pass  # correct detection

    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
    # -- Close and clean
    work_dir = vel.get_work_dir()
    shutil.rmtree(work_dir.absolute(), ignore_errors=True)


def test_VELEST_run():
    """ This tests must be done with your installation path of VELEST """
    errors = []
    #
    work_dir = "./tests_data/velest_tools/test.run.vel"
    work_cmn = "./tests_data/velest_tools/test.run.vel/velest.cmn"
    bin_path = "/usr/local/GEOPHYSICS/velest/4.5.2_MB/bin/velest_452_MB"
    mod_path = "/home/matteo/polybox/ETH/work/Sublime_SFTP_bigstar01/quake/tests_data/velest_tools/runvel.input.mod"
    cnv_path = "/home/matteo/polybox/ETH/work/Sublime_SFTP_bigstar01/quake/tests_data/velest_tools/runvel.input.cnv"
    sta_path = "/home/matteo/polybox/ETH/work/Sublime_SFTP_bigstar01/quake/tests_data/velest_tools/runvel.input.sta"
    #
    reg_names = "/usr/local/GEOPHYSICS/velest/4.5.2_MB/regionsnamen.dat"
    reg_coord = "/usr/local/GEOPHYSICS/velest/4.5.2_MB/regionskoord.dat"

    try:
        vel = VT.VELEST()
        vel.set_work_dir(work_dir)
        vel.set_cmn(work_cmn)
        vel.set_bin_path(bin_path)
        vel.set_cnv(cnv_path)
        vel.set_mod(mod_path)
        vel.set_sta(sta_path)
        vel.set_reg_names(reg_names)
        vel.set_reg_coord(reg_coord)
        vel.prepare()

    except QE.BadConfigurationFile:
        print("WARNING: missing or erroneous paths for executable (or) variables")
        return True

    # SEM
    vel.create_cmn('sem')  # defaults must be specified
    vel.work()

    difflist = QU.filediff(
        "/home/matteo/polybox/ETH/work/Sublime_SFTP_bigstar01/quake/tests_data/velest_tools/runvel.output.cnv",
        work_dir+"/VELEST.out.cnv", print_out=False)
    if difflist:
        errors.append("Processed file don't match with true test-results")

    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
    # -- Close and clean
    work_dir = vel.get_work_dir()
    shutil.rmtree(work_dir.absolute(), ignore_errors=True)
