import os
import sys
import difflib
import logging.config
import yaml
import pickle
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from pkg_resources import parse_version
#
from operator import itemgetter  # to extract picks
#
import adapt.errors as QE
from obspy import Trace


# ===============================================================
# For colored logs
# FMT = "[{levelname:^9}] {name:^20} - {funcName:^20}:{lineno} --> {message}"
FMT = "[{levelname:^9}] {name} - {funcName}:{lineno} --> {message}"
FORMATS = {
    logging.DEBUG:     f"\33[36m{FMT}\33[0m",
    logging.INFO:      FMT,
    logging.WARNING:   f"\33[33m{FMT}\33[0m",
    logging.ERROR:     f"\33[31m{FMT}\33[0m",
    logging.CRITICAL:  f"\33[1m\33[31m{FMT}\33[0m"
}


class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_fmt = FORMATS[record.levelno]
        formatter = logging.Formatter(log_fmt, style="{")  # needed for custom styling
        return formatter.format(record)
# ===============================================================


logger = logging.getLogger(__name__)


# --------------------------------------------- Calculation


def calcEpiDist(ela, elo, sla, slo, outdist="km"):
    """
    Simple function to project and calculate the
    distance between two points in lat/lon dec.deg

    outdist parameter can be "km"/"m"

    #from math import sin, cos, sqrt, atan2, radians

    """
    if not isinstance(outdist, str) or outdist.lower() not in ('km', 'm'):
        raise QE.InvalidParameter("OUTDIST must be 'km' or 'm' only!")
    #
    R = 6373.0                               # Earth's radius (km)
    lat1, lon1 = radians(ela), radians(elo)
    lat2, lon2 = radians(sla), radians(slo)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    if outdist == "m":
        return distance * 1000
    else:
        return distance


def getFirstPickPhase(pickdict, station, searchkey="Pre", indexnum=0):
    """
    This util extract the first arrival phase-key name.

    *** NB: The search key cannot contain Regexp (to implement)

    :type pickdict:  <adapt.database.PickContainer>
    :type station:   <str>
    :type searchkey: <str>
    """
    garbage = []
    for kk in pickdict[station].keys():
        if kk[0:len(searchkey)] == searchkey:
            garbage.append((
                # kk.split("_")[-1], # to get only classic
                kk,
                pickdict[station][kk][indexnum]["timeUTC_pick"]
                ))
    return sorted(garbage, key=itemgetter(1))[0][0]


def common_power_of_two(numlist, increase_exp=None):
    """ Return an integer with the closest common_power of 2
        between 2 numbers.
    """
    if not isinstance(numlist, (list, tuple)):
        raise QE.InvalidParameter("Input must be a LIST or TUPLE of numbers![%s]" %
                                  type(numlist))
    #
    maxin = np.max(numlist)
    if increase_exp:
        cl = int(2**(np.ceil(np.log2(maxin)) + increase_exp))   # NEW
    else:
        cl = int(2**(np.ceil(np.log2(maxin))))   # NEW

    cutidx = int(np.floor(cl/2.0))
    return cl, cutidx, maxin


def nearest_power_of_two(number, increase_exp=None):
    """ Return an integer with the closest common_power of 2
        between 2 numbers.
    """
    if not isinstance(number, (int, float, np.float64, np.int64)):
        raise QE.InvalidParameter("Input must be a FLOAT or INT istance! [%s]" %
                                  type(number))
    #
    if increase_exp:
        cl = int(2**(np.ceil(np.log2(number)) + increase_exp))
    else:
        cl = int(2**(np.ceil(np.log2(number))))

    cutidx = int(np.floor(cl/2.0))
    return cl, cutidx


def get_pick_slice(pickdict,
                   station,
                   searchkey="Pre",
                   phase_pick_indexnum=0,
                   arrival_order=0):
    """
    This util willextract the matching phase-key name.
    If `aarival_order` == 'all', the complete array is returned

    OUTPUT:
        List of Tuple [0] Phasename [1] TimeUTC

    """
    picklist = pickdict.getMatchingPick(station,
                                        searchkey,
                                        indexnum=phase_pick_indexnum)

    if not picklist:
        # Return Empty --> no match found
        raise QE.MissingVariable(
                     "No matching phasename with '%s' as search key!" %
                     searchkey)

    else:
        # Sort in order of time ALL the cases.
        garbage = []
        for element in picklist:
            if element[1]["timeUTC_pick"]:
                '''
                Enter here only if you have a valid pick.
                If pick == None skip
                '''
                garbage.append((
                    # kk.split("_")[-1], # to get only classic
                    element[0],
                    element[1]["timeUTC_pick"]
                    ))
        #
        mysortlist = sorted(garbage, key=itemgetter(1))
        if isinstance(arrival_order, int):
            try:
                return [mysortlist[arrival_order], ]
            except IndexError:
                logger.error("Arrival order index out bound: MAX: %d" %
                             (len(picklist)-1))
                raise QE.CheckError()
        elif isinstance(arrival_order, str) and arrival_order.lower() == "all":
            return mysortlist
        else:
            logger.error("Wrong arrival_order input type [int/str]")
            raise QE.InvalidType()


# --------------------------------------------- Configuration


def configLoggers():
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        )


def get_adapt_config(filepath, check_version=False):
    """
    Simple function to unpack the QUAKE / QUAKE_ML
    configuration file and return as a dict.
    """
    from adapt import __version__
    # Create dict
    try:
        with open(filepath, "rt") as qfc:
            # MB: adapt v0.5.9 - pyalm >= 5.1 must specify Loader
            outDict = yaml.load(qfc, Loader=yaml.FullLoader)
    except KeyError as e:
        sys.stderr.write(e + os.linesep)
        raise QE.BadConfigurationFile
    # Check Versions
    if check_version:
        if (parse_version(__version__) ==
           parse_version(outDict['adaptversion'])):
            return outDict
        else:
            logger.error(("INSTALLED version: %s  /  CONFIG version: %s") %
                         (__version__, outDict['adaptversion']))
            raise QE.BadConfigurationFile(
                "%s: not a valid CONFIGFILE, check version!" % filepath)
    #
    return outDict

# --------------------------------------------- Project/Set-Up


def createProjectDir(opev, **kwargs):
    """
    This function will make sure that the config.workingrootdir exist,
    and create the subdirectory Event and Event/waveform
    """
    if not os.path.isabs(kwargs["GENERAL"]["workingrootdir"]):
        logger.warning("Config key: 'workingrootdir' is a relative path")
    #
    if not os.path.isdir(
            kwargs["GENERAL"]["workingrootdir"] + os.sep +
            str(opev.resource_id)):
        os.makedirs(kwargs["GENERAL"]["workingrootdir"] +
                    os.sep + str(opev.resource_id), exist_ok=True)
    else:
        logger.warning("Event dir and subdirs already exists")


# --------------------------------------------- Pickling

# ----- After v0.1.4 the QUAKE database class take care of storing!
#       Next function should be called only for standard python object,
#       and not for customs or QUAKEs class.
def savePickleObj(obj, pathName):
    """
    This function should be called only for standard python object,
    and not for customs or QUAKEs class.

    pathname: specify output path and filename ( *.pkl)

    """
    with open(pathName, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def loadPickleObj(pathName):
    """
    Usually saved with extension *.pkl
    """
    with open(pathName, 'rb') as f:
        return pickle.load(f)

# --------------------------------------------- Miscellaneous


def printLogo(bufferh):
    """
    Given a buffer, it will print out the logo.
    """
    from adapt import __author__, __version__, __date__
    bufferh.write(
            os.linesep +
            "===================================" + os.linesep +
            "                __            __   " + os.linesep +
            "     ____ _____/ /___ _____  / /_  " + os.linesep +
            "    / __ `/ __  / __ `/ __ \/ __/  " + os.linesep +
            "   / /_/ / /_/ / /_/ / /_/ / /_    " + os.linesep +
            "   \__,_/\__,_/\__,_/ .___/\__/    " + os.linesep +
            "                   /_/             " + os.linesep +
            "===================================" + os.linesep
        )

    bufferh.write(("   AUTHOR: %s" + os.linesep) % __author__)
    bufferh.write(("  VERSION: %s" + os.linesep) % __version__)
    bufferh.write(("     DATE: %s" + os.linesep) % __date__)
    bufferh.write("===================================" + os.linesep * 2)


def runtimeFormat(seconds):
    h = seconds//(60*60)
    m = (seconds-h*60*60)//60
    s = seconds-(h*60*60)-(m*60)
    return (int(h), int(m), float(s))


def strcmpi(str1, str2):
    """
    Case-insensitive string compare methods
    equivalent to the "strcmpi"  function in MATLAB.
    If str2 is a list or tuple, strcmpi return
    true if str1 is contained at least once.
    Return True/False
    """
    # Compare with list/tuple:
    if isinstance(str2, list) or isinstance(str2, tuple):
        for ii in range(0, len(str2)):
            if str1.lower() == str2[ii].lower():
                return True
            else:
                continue
        return False
    # Single word compare:
    else:
        if str1.lower() == str2.lower():
            return True
        else:
            return False


def progressBar(
        iteration,
        total,
        prefix='',
        suffix='',
        decimals=1,
        barLength=100):
    """
    Call in a loop to create terminal progress bar
        @params:
          iteration   - Required  : current iteration (Int)
          total       - Required  : total iterations (Int)
          prefix      - Optional  : prefix string (Str)
          suffix      - Optional  : suffix string (Str)
          decimals    - Optional  : number of decimals in percent (int)
          barLength   - Optional  : character length of bar (Int)
    """
    percents = 100 * (iteration / float(total))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write(
        '\r%s |%s| %5.1f%s %s' %
        (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write(os.linesep)
    sys.stdout.flush()


def filediff(file1, file2, line_end=os.linesep, print_out=True):
    """ Simply report the differences among files """
    with open(file1, 'r') as ONE:
        with open(file2, 'r') as TWO:
            diff = difflib.unified_diff(
                ONE.readlines(),
                TWO.readlines(),
                fromfile='file1',
                tofile='file2',
                lineterm=line_end)

    diff_cont = [_ll for _ll in diff]
    if diff_cont:
        if print_out:
            for _ll in diff_cont:
                sys.stdout.write(_ll)
    else:
        sys.stdout.write("Files have NO differences"+os.linesep)
    #
    return diff_cont


# --------------------------------------------- Synthetic Test


def createSyntheticWave(Amp=2, Freq=25, LenSec=5, df=100.0, mode="sin"):
    """
    This method will create a sin/cos waveform of a single frequency
    of a certain length in time sampled with a certain df
    """
    if Freq > df/2:
        sys.stderr.write("Warning! ALIAS INTRODUCED")
    #
    sample = round(LenSec*df)  # int
    t = np.array([])
    y = np.array([])
    for x in np.arange(sample):
        t = np.append(t, x / df)
        if mode == "sin":
            y = np.append(y, Amp*np.sin(2 * np.pi * Freq * x / df))
        elif mode == "cos":
            y = np.append(y, Amp*np.cos(2 * np.pi * Freq * x / df))
        else:
            sys.stderr.write("ERROR! Wrong mode specified ['sin' or 'cos']")
            return None
    #
    return t, y


def mergeMultiWave(*arg):
    """
    This helper method will merge different Synthetic Waveforms, each
    one provided with a dict.
    """
    fulltime = np.array([])
    fulldata = np.array([])
    for _ii, _dd in enumerate(arg):
        if isinstance(_dd, dict):
            if _ii > 0:
                ti = fulltime[-1]
            else:
                ti = 0.0
            t, y = createSyntheticWave(**_dd)
            fulltime = np.append(fulltime, t+ti)
            fulldata = np.append(fulldata, y)
        else:
            continue
    return fulltime, fulldata


def create_obspy_trace(data, stats):
    """ It will return an ObsPy.Trace instance from give array and
        stats dictionary
    """
    if not isinstance(data, np.ndarray):
        raise QE.InvalidParameter("DATA must be a numpy.ndarray instance!")
    if not isinstance(stats, dict):
        raise QE.InvalidParameter("STATS must be a dict instance!")
    #
    return Trace(data=data, header=stats)


def extract_catalog_event(catalog, eqid=None):
    """ Simply query EQID along the catalog attribute to search """
    if not catalog:
        raise QE.MissingVariable("Missing class catalog! abort")
    if not eqid or not isinstance(eqid, str):
        raise QE.MissingVariable("Missing EQID string!")
    #
    try:
        ev = [_ev for _ev in catalog.events
              if _ev.resource_id.id == eqid][0]
        return ev
    except IndexError:
        logger.error("EVID:  %s  missing in catalog!" % eqid)
        return False


def extract_pickdict(iter, eqid=None):
    """ It will return an ADAPT PickContainer from iterable (list/tuple)
        given an evid
    """
    if not iter:
        raise QE.MissingVariable("Missing list or tuple! abort")
    if not eqid or not isinstance(eqid, str):
        raise QE.MissingVariable("Missing EQID string!")
    #
    try:
        return [ii for ii in iter if ii.eqid == eqid][0]
    except IndexError:
        logger.error("EVID:  %s  missing in list/tuple!" % eqid)
        return False
