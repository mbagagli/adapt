import os
import sys
import copy
import numpy as np
import logging
from pathlib import Path
import subprocess
import shutil
#
import pandas as pd
from obspy import UTCDateTime
import obspy.core.event as opev
from obspy.geodetics.base import kilometer2degrees, degrees2kilometers
#
from adapt import __version__, __author__, __date__
from adapt.database import PickContainer, StatContainer
import adapt.parser as QP
import adapt.utils as QU
from adapt.utils import progressBar, loadPickleObj, savePickleObj
import adapt.errors as QE


logger = logging.getLogger(__name__)


"""
THIS IS THE OFFICIAL MODULE!!! USE THIS
"""


"""
This modules aims to interface the adapt modules with the VELEST
software. They also aim to modify cnv internally.


@@@develop: create classobject for CNV and STA file. (mod should be fine already)

"""

# =================== Constants
KM = 1000.0  # m
MT = 0.001   # km

# =================== Format Lines
FMT_HEADER_CNV = "%02d%02d%02d %02d%02d %05.2f %7.4f%s %8.4f%s%7.2f%7.2f    %3d%10.2f  EVID: %s"
FMT_BODY_CNV = "%s%s%1d%6.2f"


# ===========================================================
# ===========================================================
# ================================================  PRIVATE
# ===========================================================
# ===========================================================


def _count_file_lines(filepath):
    """Return number of rows in a file. OS independent

    Args:
        filepath (str): directory path (absolute or relative) of the
                        object file
    Returns:
        count (int): number of line-count

    """
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            count += 1
    return count


def _unpack_observation_residual(instr):
    """Expand in a dict the input obs.residual string

    This helper function returns a dict with the unpacked string of
    *hypres file.
    ```
 Station residuals for event=   1, KP201601011222   160101 1222 40.54
 sta ph wt  res   ttime delta     sta ph wt  res   ttime delta
 CH01 p  0  0.434 19.00109.71     CH02 p  0 -0.588 19.52114.44
 CH03 p  0  0.767 21.34126.21     CH04 p  0 -0.868 19.28112.43
 CH06 p  0  0.548 13.44 75.08     CH09 p  0  0.252 18.49106.76
    ```

    """
    out_dict = {}
    stnm = instr[0:4]
    phas = instr[5].upper()
    pcla = np.int(instr[7:9])
    pres = np.float(instr[10:16])
    trtm = np.float(instr[16:22])
    delt = np.float(instr[22:28])
    # Compilation
    out_dict = {
        'stat_name_velest': stnm,
        'phase_name': phas,
        'phase_weight': pcla,
        'obs_res': pres,
        'obs_tt': trtm,
        'obs_delta': delt
    }
    return out_dict


def _extract_event_catalog(obcat, eqid):
    """ Extract Event object matching resource_id tag

    Args:
        obcat (obspy.core.Catalog): ObsPy Catalog object where to search

    Returns:
        (obspy.core.Event): ObsPy Event object when the resource
            identifier is matched in the query

    """
    return [ev for ev in obcat.events if ev.resource_id.id == eqid][0]


def _perturbate_event_catalog(obcat, quantity, min_depth=False, max_depth=False,
                              create_copy=False):
    """ Utility to perturbate event's preferred origin

    Return an ObsPy.Catalog object. If you want a copy in return please
    provide the right boolean in input. Otherwise inplace modification
    will be made.

    Args:
        obcat (obspy.core.Catalog): ObsPy catalog object where to search

    Returns:
        catalog (obspy.core.Catalog): ObsPy catalog object perturbated

    Note:
        `quantity` must be in kilometers!

    """
    if create_copy:
        catalog = copy.deepcopy(obcat)
    else:
        catalog = obcat
    #
    logger.info("Perturbating catalog!")

    # ===
    EPI = kilometer2degrees(quantity)
    DEP = quantity * KM  # km ==> m
    if min_depth:
        min_depth = min_depth*KM
    if max_depth:
        max_depth = max_depth*KM
    # ===

    direction = "plus"
    for xx, ee in enumerate(catalog):
        changeidx = xx % 3
        #
        if changeidx == 0:
            # changelat
            if direction == "plus":
                catalog[xx].origins[0].latitude += EPI
            elif direction == "minus":
                catalog[xx].origins[0].latitude -= EPI
            else:
                raise ValueError

        elif changeidx == 1:
            # changelon
            if direction == "plus":
                catalog[xx].origins[0].longitude += EPI
            elif direction == "minus":
                catalog[xx].origins[0].longitude -= EPI
            else:
                raise ValueError

        elif changeidx == 2:
            # changedep
            if direction == "plus":
                catalog[xx].origins[0].depth += DEP
                #
                if min_depth and catalog[xx].origins[0].depth <= min_depth:
                    catalog[xx].origins[0].depth = min_depth
                elif max_depth and catalog[xx].origins[0].depth >= max_depth:
                    catalog[xx].origins[0].depth = max_depth
                else:
                    pass
                #
                direction = "minus"

            elif direction == "minus":
                catalog[xx].origins[0].depth -= DEP
                #
                if min_depth and catalog[xx].origins[0].depth <= min_depth:
                    catalog[xx].origins[0].depth = min_depth
                elif max_depth and catalog[xx].origins[0].depth >= max_depth:
                    catalog[xx].origins[0].depth = max_depth
                else:
                    pass
                #
                direction = "plus"
            else:
                raise ValueError
    #
    return catalog


def expand_stat_alias(statCont, stnm):
    """ Expand the given alias to station full name

    This method simply return a str or a list of Stations full names
    based on input alias.

    Args:
        statCont (adapt.database.StationContainer): station container
        stnm (str, list): input list or string of stations alias

    Returns:
        out_stat_names (list): list of tuple with Station Dict

    """
    if isinstance(stnm, str):
        aa = [(ss, dd) for ss, dd in statCont.items() if dd['alias'] == str(stnm)]
        return aa[0]
    elif isinstance(stnm, list):
        aa = [(ss, dd) for ss, dd in statCont.items() if dd['alias'] in stnm]
        return aa
    else:
        print("ERROR: alias param. must be either string or list")
        return False


def _write_event_cnv(intuple, outbuff=sys.stdout):
    """ Private module function to stream CNV tuple object into buffer

    Will write the (HEADER, PHASELIST) Tuple into the given buffer being
    either a fileID or a system buffer.

    Args:
        intuple (tuple/list): object with

    Returns:
        (bool): True if everything went good.

    """
    outbuff.write(intuple[0]+os.linesep)
    #
    if not intuple[1]:
        # empty event
        outbuff.write(os.linesep)
        return False
    #
    for _i, _p in enumerate(intuple[1]):
        outbuff.write(_p)
        if (_i+1) % 6 == 0:
            outbuff.write(os.linesep)

    if (_i+1) % 6 != 0:
        outbuff.write(os.linesep*2)
    else:
        outbuff.write(os.linesep)

    return True


def _extract_event_cnv(filepath, id):
    """ Function to parse single event in CNV file into python

    Return a list of 2 elements with the header string and a list of
    phases (12 chars)

    Args:
        filepath (str): path to the CNV file containing the event.

    Returns:
        outList (list): list containig 2 index, the first being the
        header, the second being  a list of phase list.

    """
    outList = []
    phaselist = []
    inside_event = False
    with open(filepath, "r") as CNV:

        for _i, _l in enumerate(CNV):
            fields = _l.strip().split()

            if not fields:
                if inside_event:
                    break
                else:
                    continue

            if inside_event:
                lstr = _l.strip()
                obs = [lstr[idx:idx + 12] for idx in range(0, len(lstr), 12)]
                phaselist = phaselist + obs

            cnvid = fields[-1]
            if cnvid == id and not inside_event:
                # HEADER
                outList.append(_l.strip())
                inside_event = True

    outList.append(phaselist)
    return outList


def _extract_all_events_cnv(filepath):
    """ Function to parse multi-events in CNV file into python

    Return a dictionary of EQID keys each one with 2 elements with
    the header string and a list of phases (12 chars)

    Args:
        filepath (str): path to the CNV file containing the event.

    Returns:
        outDict (dict): sdict containig 2 index, the first being the
        header, the second being  a list of phase list.

    """
    outDict = {}
    #
    inside_event = False
    with open(filepath, "r") as CNV:

        for _i, _l in enumerate(CNV):
            fields = _l.strip().split()

            if fields and not inside_event:
                cnvid = fields[-1]
                header = _l.strip()
                inside_event = True
                phaselist = []
                continue

            if inside_event:
                lstr = _l.strip()
                obs = [lstr[idx:idx + 12] for idx in range(0, len(lstr), 12)]
                phaselist = phaselist + obs

            if not fields:
                # Storing the EVENT
                outDict[cnvid] = [header, phaselist]
                # closing an event
                inside_event = False
                continue
    return outDict


def _eventpickdict2cnv(eve, pickd, stations_dict=None, phase_list=['P1', ],
                       fixed_class=False, use_buffer=False,
                       out_file="event2cnv.out", use_alias=True):
    """ Main function where magic happens, it requires:
            - Event object (obspy)
            - Quake_StationMETADATA (NB: must contain ALIAS attribute)
            - Quake_PickDict
            - phaselist ->  List of phases to be converted
    """
    if use_buffer and out_file:
        raise ValueError("Only one among an output filepath or a buffer object "
                         "must be specified! Default to output-file")
    if use_alias and not stations_dict:
        raise ValueError("The use_alias set to True, but no station "
                         "dictionary found [station_dict] !!!")
    #
    if out_file:
        with open(out_file, "w") as OUT:
            # ====================================   HEADER
            lat = eve.origins[0]['latitude']
            lon = eve.origins[0]['longitude']
            dep = eve.origins[0]['depth']  # output in meteres
            dep = dep*10**(-3)  # conversion to km
            ot = eve.origins[0]['time']   # UTCDateTime
            mag = eve.magnitudes[0]['mag']
            kpid = eve.origins[0].resource_id.id
            #
            if lat >= 0.0:
                lat_comp = "N"
            else:
                lat_comp = "S"

            if lon >= 0.0:
                lon_comp = "E"
            else:
                lon_comp = "W"

            # 19112019 added GAP and RMS (optional)
            try:
                gap = np.int(np.round(eve.origins[0].quality.azimuthal_gap))
            except AttributeError:
                # GAP is None, or not stored
                gap = 0

            try:
                res = eve.origins[0].quality.standard_error
            except AttributeError:
                # RMS is None, or not stored
                res = 0.0

            OUT.write((FMT_HEADER_CNV+os.linesep) % (
                        np.int(str(ot.year)[-2:]), np.int(ot.month), np.int(ot.day),
                        np.int(ot.hour), np.int(ot.minute),
                        (ot.second + ot.microsecond*10**(-6)),
                        np.abs(lat), lat_comp, np.abs(lon), lon_comp,
                        np.float(dep), np.float(mag),
                        gap, res, kpid))

            # ====================================   BODY
            idx = 0
            for stat, phasedict in pickd.items():
                for phase, phasepickslist in phasedict.items():
                    if phase in phase_list:
                        if "p" in phase.lower():
                            phn = "P"
                        elif "s" in phase.lower():
                            phn = "S"
                        else:
                            raise TypeError("Unrecognized PHASE: %s" % phase)

                        for pck in phasepickslist:
                            if fixed_class:
                                cla = np.int(fixed_class)
                            else:
                                cla = np.int(np.round(pck['pickclass']))

                            try:
                                pht = pck['timeUTC_pick'] - ot
                            except TypeError:
                                # Missing phasepick
                                continue

                            if use_alias:
                                if ("alias" not in stations_dict[stat].keys() or
                                   not stations_dict[stat]['alias']):
                                    raise TypeError(
                                        "ALIAS must be given for station: %s" %
                                        stat)
                                printstat = stations_dict[stat]['alias']
                            else:
                                printstat = stat
                            #
                            if len(printstat) > 4:
                                raise ValueError("Stations in CNV file must be "
                                                 "4 char long only!")
                            #
                            OUT.write((FMT_BODY_CNV) % (printstat, phn, cla, pht))
                            idx += 1
                            if idx == 6:
                                OUT.write(os.linesep)
                                idx = 0
            # --- End of EVENT
            if idx == 0:
                OUT.write(os.linesep)
            else:
                OUT.write(os.linesep*2)

    elif use_buffer:
        # ====================================   HEADER
        lat = eve.origins[0]['latitude']
        lon = eve.origins[0]['longitude']
        dep = eve.origins[0]['depth']  # output in meteres
        dep = dep*10**(-3)  # conversion to km
        ot = eve.origins[0]['time']   # UTCDateTime
        mag = eve.magnitudes[0]['mag']
        kpid = eve.origins[0].resource_id.id
        #
        if lat >= 0.0:
            lat_comp = "N"
        else:
            lat_comp = "S"

        if lon >= 0.0:
            lon_comp = "E"
        else:
            lon_comp = "W"

        # 19112019 added GAP and RMS (optional)
        try:
            gap = np.int(np.round(eve.origins[0].quality.azimuthal_gap))
        except AttributeError:
            # GAP is None, or not stored
            gap = 0

        try:
            res = eve.origins[0].quality.standard_error
        except AttributeError:
            # RMS is None, or not stored
            res = 0.0

        use_buffer.write((FMT_HEADER_CNV+os.linesep) % (
                    np.int(str(ot.year)[-2:]), np.int(ot.month), np.int(ot.day),
                    np.int(ot.hour), np.int(ot.minute),
                    (ot.second + ot.microsecond*10**(-6)),
                    np.abs(lat), lat_comp, np.abs(lon), lon_comp,
                    np.float(dep), np.float(mag),
                    gap, res, kpid))

        # ====================================   BODY
        idx = 0
        for stat, phasedict in pickd.items():
            for phase, phasepickslist in phasedict.items():
                if phase in phase_list:
                    if "p" in phase.lower():
                        phn = "P"
                    elif "s" in phase.lower():
                        phn = "S"
                    else:
                        raise TypeError("Unrecognized PHASE: %s" % phase)

                    for pck in phasepickslist:
                        if fixed_class:
                            cla = np.int(fixed_class)
                        else:
                            cla = np.int(np.round(pck['pickclass']))

                        try:
                            pht = pck['timeUTC_pick'] - ot
                        except TypeError:
                            # Missing phasepick
                            continue

                        if use_alias:
                            if ("alias" not in stations_dict[stat].keys() or
                               not stations_dict[stat]['alias']):
                                raise TypeError(
                                  "ALIAS must be given for station: %s" % stat)
                            printstat = stations_dict[stat]['alias']
                        else:
                            printstat = stat
                        #
                        if len(printstat) > 4:
                            raise ValueError("Stations in CNV file must be "
                                             "4 char long only!")
                        #
                        use_buffer.write((FMT_BODY_CNV) %
                                         (printstat, phn, cla, pht))
                        idx += 1
                        if idx == 6:
                            use_buffer.write(os.linesep)
                            idx = 0
        # --- End of EVENT
        if idx == 0:
            use_buffer.write(os.linesep)
        else:
            use_buffer.write(os.linesep*2)


def _manupick2PickContainer(filepath, eqid, eqtag):
    """
    This Function will convert a MANUPICK SED/ETHZ
    phase format into a PickContainer class
    inside the framework itself.

    It return a `<adapt.database.PickContainer>`

    v0.3.1 if eqtag == 'reference' than we use 'Reference_'
           else 'Seiscomp_' is used

    """
    stationlist = []
    pickdict = {}
    # Fastest way to count InputLines
    with open(filepath, "r") as IN:
        totlines = sum(1 for _ in IN)
    ###
    with open(filepath, "r") as IN:
        for xx, line in enumerate(IN):             # take origin times
            if xx == 3:
                line = line.strip()
                refertime = UTCDateTime(year=int(line[0:4]),
                                        month=int(line[5:7]),
                                        day=int(line[8:10]),
                                        hour=int(line[11:13]),
                                        minute=int(line[14:16])
                                        )
                # if inkey=="INPUT":
                # PickDict[eqid]["referenceUTC"]=refertime
                #    ( Reference TIME for MANUPICK [done only @ INPUT] )
            if xx > 3 and xx < (totlines - 1):
                # take picks, avoid the *SKIP line at bottom file
                # Exctract Info from Line
                station = line[0:7].strip()

                # Next line mod. to improve tag extraction 15.11.2018
                #   phaseName = line[8:12].strip()
                # v0.3.1
                phaseName = line[8:12].strip()

                phasett = float(line[19:26].strip())
                emim = line[16]
                polarity = line[17]
                weight = None

                if line[27] == "0":
                    usepick = False
                elif line[27] == "1":
                    usepick = True

                try:
                    classNum = int(line[88])
                except (ValueError, IndexError):
                    classNum = None

                # Check Dict Keys
                if station not in pickdict:
                    pickdict[station] = {}
                if phaseName not in pickdict[station]:
                    pickdict[station][phaseName] = []

                # Add Vals to Dict
                simplepickdict = {}
                simplepickdict["onset"] = emim
                simplepickdict["usepick"] = usepick
                simplepickdict["pickpolar"] = polarity
                simplepickdict["pickerror"] = weight
                simplepickdict["pickclass"] = classNum
                #
                simplepickdict["timeUTC_pick"] = refertime + phasett
                try:
                    earlytt = float(line[72:79].strip())
                    simplepickdict["timeUTC_early"] = refertime + \
                        phasett + earlytt
                # missing (could not convert str to float)
                except (ValueError, IndexError):
                    earlytt = None
                    simplepickdict["timeUTC_early"] = None
                try:
                    latett = float(line[80:87].strip())
                    simplepickdict["timeUTC_late"] = refertime + \
                        phasett + latett
                except (ValueError, IndexError):
                    latett = None
                    simplepickdict["timeUTC_late"] = None
                # Append loop results
                pickdict[station][phaseName].append(simplepickdict)
                if station not in stationlist:
                    stationlist.append(station)
    # Convert it
    stationlist.sort()
    outdict = QP.dict2PickContainer(pickdict, eqid, eqtag, "STANDARD")
    return outdict, stationlist


# ===========================================================
# ===========================================================
# ================================================  MODIFY
# ===========================================================
# ===========================================================


def perturbate_events_cnv(cnvfile, perturb, outcnv="perturbated.cnv"):
    """ Perturbate the hypocenters in a cnv and return the cnv

    Args:
        cnvfile (str): path to cnv file
        perturb (int, float): quantity to perturbate

    Returns:
        retcat (obspy.core.Catalog)
        evpdcollection (list): list of adapt.database.PickContainer
    """
    print("Loading!")
    catalog, evpdcollection = cnv2quake(cnvfile)
    retcat = _perturbate_event_catalog(catalog, perturb, min_depth=0.05)
    # retcat = _perturbate_event_catalog(catalog, perturb)
    quake2cnv(retcat, evpdcollection, statdict=None,
              phase_list=["VEL_P", "VEL_S", ],
              fixed_class=False, use_alias=False, out_file=outcnv)
    return retcat, evpdcollection


def import_stations_corrections(statdict, velstafile, minp=1, mins=1,
                                ignore_missing_stations=False,
                                export_obj=False):
    """ This function will add the station corrections into the given
        QUAKE.StatContainer object.
        It needs a VELEST *statistic.sta file (generated on output)

        NB: Will override p_delay , s_delay keys if present!

        If ignore_missing_stations is TRUE, the function WILL NOT
        raise an error when NO ALIAS is found

        'export_obj' must be a string to store the object

    """
    logging.info("Importing Station Corrections from: %s" % velstafile)
    if isinstance(statdict, str):
        STATCONT = loadPickleObj(statdict)
    else:
        STATCONT = statdict
    #
    kp = []  # list of modified p_delay stationsdelay
    ks = []  # list of modified s_delay stationsdelay
    with open(velstafile, "r") as IN:
        # sta phase  nobs avres  avwres    std     wsum    delay
        for xx, ll in enumerate(IN):
            if xx == 0:
                continue
            #
            fields = ll.strip().split()
            statnm = fields[0]
            statph = fields[1]
            nobs = np.int(fields[2])
            corr = np.float(fields[-1])
            # Check
            if ((statph.lower() == "p" and nobs < minp) or
               (statph.lower() == "s" and nobs < mins)):
                continue
            #
            skmatch = [k for k, s in STATCONT.items()
                       if s['alias'] == statnm]
            if not skmatch:
                if ignore_missing_stations:
                    continue
                raise QE.MissingVariable(
                    "Station Alias not found for [%s]! Wrong inventory?" %
                    statnm)

            elif len(skmatch) > 1:
                raise QE.MissingVariable(
                    "Multiple Alias found for [%s]: %s , Wrong inventory?" %
                    (statnm, skmatch))
            else:
                # Exist and unique
                sk = skmatch[0]

            if statph.lower() == "p":
                STATCONT[sk]['p_delay'] = corr
                kp.append(sk)
            elif statph.lower() == "s":
                STATCONT[sk]['s_delay'] = corr
                ks.append(sk)
            else:
                raise ValueError("Unrecongnized phase: %s" % statph)
    # Reset to zero the other stations, delays
    for _kk, _ss in STATCONT.items():
        if _kk not in kp:
            STATCONT[_kk]['p_delay'] = np.float(0.00)
        if _kk not in ks:
            STATCONT[_kk]['s_delay'] = np.float(0.00)
    #
    if export_obj:
        savePickleObj(STATCONT, export_obj)
    return STATCONT


def remove_obs_from_cnv(cnvfile, staresfile, threshold=1.0):
    """ This function, will remove the precise obs-stations given in
        *statistic.sta from CNV. The threshold is the RMS per obs
        It will return a new CNV (*.filtered)
    """
    cnvDict = _extract_all_events_cnv(cnvfile)
    tl = _count_file_lines(staresfile)
    with open(staresfile, "r") as INSTAT:
        logger.info("Removing faulty obs")
        for _xx, _ll in enumerate(INSTAT):
            if _xx == 0 or _ll[0] == "#":
                continue
            #
            fields = _ll.strip().split()
            eqid = fields[-1]
            res = fields[3]
            statname = fields[0]
            if np.abs(np.float(res)) >= threshold:
                # Seek the event, remove the obs
                cnvDict[eqid][1] = [obs for obs in cnvDict[eqid][1]
                                    if obs[0:4] != statname]
            # Progress Bar
            progressBar(_xx, tl-1, prefix='Removing OBS:', suffix='Complete',
                        barLength=15)

    # Write new file
    logger.info("Creating a NEW CNV")
    with open(cnvfile+".filtered", "w") as OUT:
        for _kk, _ev in cnvDict.items():
            _write_event_cnv(_ev, outbuff=OUT)
    #
    return True


def remove_stations_from_cnv(cnvfile, statlst=None, export=True):
    """ This function, will remove ALL the observations from the given
        station lists. No matter if P or S obs at the moment.
        It will write a new CNV (*.filtered)
    """
    if not statlst or not isinstance(statlst, (list, tuple)):
        raise QE.MissingVariable("I need a stations alias list to work!")
    #
    cnvDict = _extract_all_events_cnv(cnvfile)
    for _kk, _ev in cnvDict.items():
        _ev[1] = [ii for ii in _ev[1] if ii[0:4] not in statlst]

    # Write new file
    if export:
        logger.info("Creating a NEW CNV")
        with open(cnvfile+".statsfiltered", "w") as OUT:
            for _kk, _ev in cnvDict.items():
                _write_event_cnv(_ev, outbuff=OUT)
    return cnvDict

# ===========================================================
# ===========================================================
# ================================================  PARSING
# ===========================================================
# ===========================================================


def quake2cnv(cat, pickdlist, statdict=None,
              phase_list=["P1", ], fixed_class=False,
              out_file="quake2cnv_out.cnv", use_alias=False,
              split_single=False):
    """ Routine to convert back to CNV

    pickdlist is a list of pickobject. Make sure the resource_id and the
        eqid attribute of PickCOntainer coincide

    split_single will create a separate - single CNV file for each event

    """
    if isinstance(cat, str):
        cat = loadPickleObj(cat)
    if isinstance(statdict, str):
        statdict = loadPickleObj(statdict)
    if isinstance(pickdlist, str):
        pickdlist = loadPickleObj(pickdlist)

    with open(out_file, "w") as OUT:

        for _xx, ee in enumerate(cat):

            if len(cat) > 1:
                progressBar(_xx, len(cat)-1, prefix='Converting: ',
                            decimals=2, barLength=15)

            eqid = ee.origins[0].resource_id.id
            try:
                # evpickd = [ii for ii in pickdlist if ii.eqid == eqid][0]
                evpickd = QU.extract_pickdict(pickdlist, eqid)
            except IndexError:
                print("PickContainer of %s is MISSING! Skipping ..." % eqid)

            _eventpickdict2cnv(ee, evpickd, stations_dict=statdict,
                               fixed_class=fixed_class,
                               use_buffer=OUT, out_file=None,
                               phase_list=phase_list,
                               use_alias=use_alias)


def cnv2quake(cnvfile, info_store="VelestRun",
              alias_convert=None, fixed_depth=None):
    """Function to convert a CNV file back into ADAPT/ObsPy classes

        This function will convert back a CNV (VELEST) earthquake file
        back into a list of adapt.PickContainer and an ObsPy catalog.

        Args:
            cnvfile (str): a CNV file path
            info_store (str): tag of the resulting PickContainer
            alias_convert (str, adapt.StationContainer): must be either
                a path to StatContainer file or a StatContainer object
                itself. If provided, make sure the key `alias` is
                present for each stations. Default will not convert
                the stations name and will leave the VELEST one intact.
            fixed_depth (int, float): will store the events with a fixed
                depth in the header.

        Note:
            If the CNV file contains both P and S phases, these will be
            stored underneath the phase tag VEL_P and VEL_S respectively

            Maybe update the function with PickContainer class methods
            for storing.

    """
    opcat = opev.Catalog()   # QuakeCatalog.append(ev)
    pkcnt = []
    #
    cnvDict = _extract_all_events_cnv(cnvfile)

    #
    if alias_convert:
        if isinstance(alias_convert, str):
            STATCONT = loadPickleObj(alias_convert)
        else:
            STATCONT = alias_convert

    #
    logger.info("Converting CNV to QUAKE PickContainer and ObsPy Catalog")
    for _eqid, _eqtpl in cnvDict.items():

        # --- Convert HEADER to valid dict for quake tools
        # The subsequent keys ('id', 'origintime', 'lat', 'lon', 'dep',
        #                      'rms', 'gap', 'mag', 'magType')
        headDict = {}
        hd = _eqtpl[0]
        # ID
        headDict['id'] = _eqid
        # OT
        dd = "-".join(['20' +
                       '{:02d}'.format(np.int(hd[0:2])),
                       '{:02d}'.format(np.int(hd[2:4])),
                       '{:02d}'.format(np.int(hd[4:6]))
                       ])
        tt = ":".join([
                       '{:02d}'.format(np.int(hd[7:9])),
                       '{:02d}'.format(np.int(hd[9:11])),
                       '{:f}'.format(np.float(hd[12:17]))
                       ])
        evot = UTCDateTime(dd+"T"+tt)
        headDict['origintime'] = evot

        # @@@ Lat
        if hd[25] == "N":
            headDict['lat'] = np.float(hd[18:25])
        elif hd[25] == "S":
            headDict['lat'] = -np.float(hd[18:25])
        else:
            raise TypeError("Wrong input CNV for LAT %s" % hd[18:25])
        # @@@ Lon
        if hd[35] == "E":
            headDict['lon'] = np.float(hd[27:35])
        elif hd[35] == "W":
            headDict['lon'] = -np.float(hd[27:35])
        else:
            raise TypeError("Wrong input CNV for LON %s" % hd[27:35])
        # @@@ Dep
        headDict['dep'] = np.float(hd[37:43]) * 1000  # unit: meters
        # For backward compatibility and as utility
        if fixed_depth and isinstance(fixed_depth, (float, int)):
            headDict['dep'] = fixed_depth * 1000  # unit: meters
        # @@@ Mag
        headDict['mag'], headDict['magType'] = np.float(hd[45:50]), "Mlv"
        # @@@ GAP
        headDict['gap'] = np.float(hd[54:57])
        # @@@ RMS
        headDict['rms'] = np.float(hd[60:67])
        # ... finally append
        opcat.append((QP.dict2Event(headDict)))

        # --- Convert observation to valid dict for adapt tools
        # The USERDICT should have the subsequent keys:
        # userdict['STATION']={'PHSNM'=[
        #                         { 'polarity',       # (adapt.picks.polarity.Polarizer/None)
        #                           'onset',          # (str/None)
        #                           'weight',         # (adapt.picks.weight.Weighter/None)
        #                           'pickerror',      # (float/None)
        #                           'pickclass',      # (int/None)
        #                           'pickpolar',      # (str/None)
        #                           'evaluate',       # (bool/None)
        #                           'evaluate_obj',   # (adapt.picks.evaluation.Gandalf/None)
        #                           'features',       # (dict/None)
        #                           'features_obj',   # (adapt.picks.featuring.Miner/None)
        #                           'outlier',        # (bool/None)
        #                           'phaser_obj',     # (adapt.picks.phaser.Spock/None)
        #                           'timeUTC_pick',   # (UTCDateTime/None)
        #                           'timeUTC_early',  # (UTCDateTime/None)
        #                           'timeUTC_late',   # (UTCDateTime/None)
        #                           'backup_picktime' # (UTCDateTime/None)
        #                         },
        #                              ]
        #                     }

        obsDict = PickContainer(_eqid, info_store, "VELEST")
        for _ob in _eqtpl[1]:
            # 'I041P0  9.67'
            statnm = _ob[0:4]
            statnm_err = _ob[0:4]
            if alias_convert:
                try:
                    statnm = [ss['fullname'] for _, ss in STATCONT.items()
                              if ss['alias'] == statnm][0]
                except IndexError:
                    raise QE.MissingVariable("Missing alias for [%s]! "
                                             "Wrong Inventory?" % statnm_err)

            statph = _ob[4]
            statcl = np.int(_ob[5])
            stattt = evot + np.float(_ob[6:])
            #
            obsDict.addPick(statnm, 'VEL_' + statph,
                            pickclass=statcl,
                            timeUTC_pick=stattt)
        pkcnt.append(obsDict)
    #
    return opcat, pkcnt


def seiscomp_sed2cnv(manupdefile,
                     manupickfile,
                     statdictwithalias,
                     eventid="dunno",
                     outputfile="seiscomp_sed2cnv.out",
                     fixed_depth=None):
    """Convert a MANUPDE and MANUPICK (from SED-SC3) file

    This function will convert the *MANUPDE and *MANULOC files from
    SeiscComp3 output (only for SED version) hypocenter locations into
    a VELEST CNV file. This function will create an output CNV-file.

    Args:
        manupdefile  (str): filepath of a SED *MANUPDE file
        manupickfile (str): filepath of a SED *MANUPICK file
        statdictwithalias (str, adapt.database.StatContainer): must be
            either a path to StatContainer file or a StatContainer
            object itself.
    Returns:
        bool: Description of return value

    Notes:
        add an option for alias-conversion confirm. If set to None or
        False, only the first 4 carachters of the original station name
        will be used.
    """
    logger.info("Converting %s and %s --> %s" % (manupdefile,
                                                 manupickfile,
                                                 outputfile))
    #
    if isinstance(statdictwithalias, str):
        statobj = loadPickleObj(statdictwithalias)
    else:
        statobj = statdictwithalias
    #
    evobj = QP.manupde2Event(manupdefile,
                             eventid,
                             "tmp_useless", fixed_depth=fixed_depth)
    pickdict, _ = _manupick2PickContainer(
                                manupickfile, eventid, "tmp_useless")
    #
    _eventpickdict2cnv(evobj,
                       pickdict,
                       stations_dict=statobj,
                       # Next line must represent 1st arrival ONLY!
                       phase_list=['P', 'Pg', 'Pn', 'P1', 'S', 'S1', 'Sg'],
                       out_file=outputfile, fixed_class=0)
    #
    return True


def sort_my_cnv(incnv, outcnv="sort_my_cnv.sorted.cnv"):
    """ Simple function to order the CNV file in time order """

    # Extract
    cnvDict = _extract_all_events_cnv(incnv)

    # Write new file
    logger.info("Creating a NEW CNV")
    with open(outcnv, "w") as OUT:
        for _kk, _ev in sorted(cnvDict.items()):
            _write_event_cnv(_ev, outbuff=OUT)
    #
    return True


# ===========================================================
# ===========================================================
# ================================================  EXTRACT
# ===========================================================
# ===========================================================


def hypres2dict(file_path, store_pickle=False):
    """Return a dictionary of the given *hypres file

    This function will return a dictionary containing all the stations
    residual divided by EVID keys. The *hyperes file is produced only
    by VELEST 4.5

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        hypresdct (dict): residual observation dict split by EQID and
            STATION names

    """
    out_dict = {}
    inside_event = False
    with open(file_path, "r") as IN:
        for _xx, _ll in enumerate(IN):
            _lls = _ll.strip()
            _fields = _ll.strip().split()
            # colpisci
            if not _fields:
                # event close or useless header
                inside_event = False
                continue
            elif _fields[0] == "sta":
                continue
            #
            elif _fields[0] == "Station":
                # event open
                evid = _ll.strip().split(",")[1].split()[0]
                out_dict[evid] = {}
                inside_event = True
            elif inside_event:
                # collect
                tmpDict = _unpack_observation_residual(_lls[0:28])
                out_dict[evid][tmpDict['stat_name_velest']] = tmpDict

                if len(_lls) > 28:
                    # go for the second
                    tmpDict = _unpack_observation_residual(_lls[33:61])
                    out_dict[evid][tmpDict['stat_name_velest']] = tmpDict
            else:
                print("Unrecognized switch: %s" % _lls)
    #
    if store_pickle:
        logger.debug("storing the HypRes pickle")
        savePickleObj(out_dict, "hypores_dict.pkl")
    return out_dict


def stares2dict(file_path, store_pickle=False):
    """Return a dictionary of the given *stares file

    This function will return a dictionary containing all the stations
    residual divided by EVID keys. The *hyperes file is produced only
    by VELEST 4.5

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        hypresdct (dict): residual observation dict split by EQID and
            STATION names

 sta ph wt res ttime delta yyyymmdd hhmm ss evID
CH49 p  0  -0.602    4.67    8.26 160101 1222 40.24 KP201601011222
CH57 p  0  -0.390    4.64    9.18 160101 1222 40.24 KP201601011222
CH36 p  0  -0.305    4.80   10.90 160101 1222 40.24 KP201601011222
CH66 p  0  -0.154    6.46   28.09 160101 1222 40.24 KP201601011222

    """
    out_dict = {}
    with open(file_path, "r") as IN:
        for _xx, _ll in enumerate(IN):
            # Skip header " sta ph wt res ttime delta yyyymmdd hhmm ss evID"
            if _xx == 0:
                continue
            #
            _fields = _ll.strip().split()
            eqid = _fields[-1]
            stnm = str(_fields[0])
            phas = str(_fields[1]).upper()
            pcla = np.int(_fields[2])
            pres = np.float(_fields[3])
            trtm = np.float(_fields[4])
            delt = np.float(_fields[5])
            #
            if eqid not in out_dict.keys():
                # create new eqid-key
                out_dict[eqid] = {}
            # Compilation
            out_dict[eqid][stnm] = {
                            'stat_name_velest': stnm,
                            'phase_name': phas,
                            'phase_weight': pcla,
                            'obs_res': pres,
                            'obs_tt': trtm,
                            'obs_delta': delt
                        }
    #
    if store_pickle:
        logger.debug("storing the StaRes pickle")
        savePickleObj(out_dict, "stares_dict.pkl")
    return out_dict

# ===========================================================
# ===========================================================
# ================================================  INFOs
# ===========================================================
# ===========================================================


"""
# -> VELEST CNV (v4.5 or higher)
'16 1 3 1554 29.67 44.0271N  11.9400E  19.25   2.10    181      0.12  EVID: KP201601031554'
'I041P0  9.67I063P0  8.47I102P0  5.78I105P0  9.98I106P0  6.91I127P0  5.47'
'I164P0  7.27I185P0  9.16'
"""

"""
# -> VELEST *statistic.sta (v4.5 or higher)
 sta phase  nobs avres  avwres    std     wsum    delay
 8X03   P     6  0.0750  0.0750  0.2047     6.0000    0.1097
 8X07   P     2  0.0483  0.0483  0.4667     2.0000   -1.1205
 BE01   P     2  0.0562  0.0562  0.0523     2.0000    1.1081
 BE04   P     3  0.0499  0.0499  0.1140     3.0000    0.6267
 BW01   P    37  0.0638  0.0638  0.4176    37.0000   -1.1540
 BW02   P    24  0.0181  0.0181  0.3468    24.0000   -1.8920
 BW03   P     6  0.0419  0.0419  0.3835     6.0000   -1.8560
 BW04   P     5  0.0794  0.0794  0.2603     5.0000    1.3907
 BW05   P     8  0.0608  0.0608  0.3265     8.0000    1.1908
"""


"""
# -> SC3-SED *.manupde
  E V E N T   N R . 201709001                 MODEL: Alpine Regional 1D (Diehl 2009) - ManualMode AUTHOR: irene MAGNITUDE-METHOD: median

 Grid Search Solution: NonLinLoc(EDT)

   DATE  ORIGIN TIME    LAT      LON    Z   AZ  D-MIN  Ml   RMS GAP  NO NI Q
 20170906 12:22:29.8 46.262N  11.991E  0.0  -1    5.5 3.8  0.45  28 129 99 D
   ACCURACY: (+/-)    0.569    0.513  2.04            0.3

 L+T NR:-999  Bolzano I                                KOORDINATEN: 999/999 KM


STA    PHASE  RMK W MN P-SEC P-CAL P-RES   DT   WT     AMP    PER MAG    DELTA  AZIM ANGLE
AGOR   Pg      I  1 22 31.88 30.88  1.00 9.90 0.54 1.7E+02 W 0.00 3.7-     5.5 125.9 -99.0
VARN   Pg      ID 1 22 35.41 35.58 -0.18 9.90 1.41 2.2E+01 W 0.00 3.4-    31.6 161.6 -99.0
ZIAN   Pg      IU 1 22 36.07 35.82  0.24 9.90 1.41 6.5E+01 W 0.00 3.9-    33.0 272.8 -99.0
CTI    Pg      ID 1 22 36.14 36.15 -0.01 9.90 1.49 5.1E+01 W 0.00 3.9     34.8 227.0 -99.0
CTI    Sg      I  1 22 41.14 40.62  0.52 9.90 1.09 5.1E+01 W 0.00 3.9-    34.8 227.0 -99.0
CIMO   Pg      IU 1 22 35.95 36.16 -0.21 9.90 1.37 2.0E+01 W 0.00 3.5     35.1  80.9 -99.0
CIMO   Sg      E  1 22 41.17 40.64  0.52 9.90 1.09 2.0E+01 W 0.00 3.5-    35.1  80.9 -99.0
FRE8   Pg      ID 1 22 36.92 36.99 -0.07 9.90 1.47 2.3E+01 W 0.00 3.6-    40.0 133.3 -99.0
A291A  Pg      ID 1 22 37.43 37.34  0.09 9.90 1.48 3.1E+01 W 0.00 3.8-    41.5 346.1 -99.0
"""


def _define_class(delta, velres, classdict):

    ### classdct_original = {
    ###     '0': 0.25,
    ###     '1': 0.5,
    ###     '2': 1.0,
    ###     '3': 2.0,
    ### }
    ### classdct_shrink = {
    ###     '0': 0.15,
    ###     '1': 0.3,
    ###     '2': 0.6,
    ###     '3': 1.2,
    ### }

    # # ---- TD
    # classdict = {
    #     '0': 0.1,
    #     '1': 0.2,
    #     '2': 0.4,
    #     '3': 0.8,
    # }

    # --- IM-SC3
    # classdict={'0': 0.05,
    #            '1': 0.1,
    #            '2': 0.2,
    #            '3': 0.4,
    #            '4': 0.8},

    dist = np.sqrt(delta**2 + velres**2)
    #
    cc = 4
    if dist < classdict['0']:
        cc = 0
    elif classdict['0'] <= dist < classdict['1']:
        cc = 1
    elif classdict['1'] <= dist < classdict['2']:
        cc = 2
    elif classdict['2'] <= dist < classdict['3']:
        cc = 3
    elif classdict['3'] <= dist:
        cc = 4
    else:
        raise ValueError("Something Fishy --> %f" % dist)
    #
    return cc, dist


def _classify_cnv(cnvfile, staresfile, csvfile,
                  outfilecsv="classify_debug.csv", epidistmax=1000.0,
                  classdict={'0': 0.1, '1': 0.2, '2': 0.4, '3': 0.8}):
    cat, pdl = cnv2quake(cnvfile)
    resdict = stares2dict(staresfile, store_pickle=False)  # dict[eqid][statname]   #ID 'I022': {'stat_name_velest': 'I022', 'phase_name': 'P', 'phase_weight': 0, 'obs_res': 0.202, 'obs_tt': 44.39, 'obs_delta': 298.06},

    df = pd.read_csv(csvfile)
    missing_events = []

    with open(outfilecsv, 'w') as OUT:
        OUT.write("EQID, STAT, EPIDIST, DELTA, RESID"+os.linesep)

        ln = len(resdict.keys())
        for xx, (ev, ssdct) in enumerate(resdict.items()):
            try:
                evpickd = [ii for ii in pdl if ii.eqid == ev][0]
            except IndexError:
                #print(("PickContainer of %s is MISSING! Skipping ..."+os.linesep) % ev)
                missing_events.append(ev)
                continue
            #
            for ss, vss in ssdct.items():
                resid = vss['obs_res']
                tetl = np.float(df[(df['id'] == ev) &
                                (df['alias'] == ss)]['tlatetearly'])
                epid = np.float(df[(df['id'] == ev) &
                                (df['alias'] == ss)]['epidist'])
                OUT.write(("%s,%s,%8.4f, %8.4f, %7.3f"+os.linesep) %
                          (ev, ss, epid, tetl, resid))
                mycc, _ = _define_class(tetl, resid, classdict)
                #
                if epid <= epidistmax:
                    evpickd[ss]["VEL_P"][0]['pickclass'] = mycc
                else:
                    del evpickd[ss]
            #
            progressBar(xx, ln-1, 'Classifying: ', 'DONE',
                        decimals=2, barLength=15)

    quake2cnv(cat, pdl, statdict=None,
              phase_list=["VEL_P", "VEL_S", ],
              fixed_class=False, use_alias=False, out_file="classified.cnv")
    #
    logger.warning("List of missing EVENTS:")
    for _ev in missing_events:
        logger.warning("- %s" % _ev)


# def _classify_reloc(
#                   relocfile, csvfile,
#                   outfilecsv="classify_debug.csv",
#                   classdict={'0': 0.1, '1': 0.2, '2': 0.4, '3': 0.8}):
#     relobj = RELOC(relocfile)  # dict[eqid][statname]   #ID 'I022': {'stat_name_velest': 'I022', 'phase_name': 'P', 'phase_weight': 0, 'obs_res': 0.202, 'obs_tt': 44.39, 'obs_delta': 298.06},
#     cat, pdl = relobj.reloc_catalog, relobj.reloc_pickList
#     print("Reading CSV")
#     df = pd.read_csv(csvfile)
#     missing_events = []
#     #
#     with open(outfilecsv, 'w') as OUT:
#         OUT.write("EQID, STAT, EPIDIST, PHASE, DELTA, RESID, CLASS"+os.linesep)
#         ln = len(relobj.reloc_catalog)
#         for xx, ev in enumerate(cat):
#             eqid = ev.resource_id.id
#             respckdct = QU.extract_pickdict(pdl, eqid=eqid)
#             if not respckdct:
#                 missing_events.append(ev)
#                 continue
#             #
#             for ss, pds in respckdct.items():
#                 for ph, php in pds.items():
#                     # Usually in RELOC file there's should be only one obs per phase
#                     resid = php[0]['general_infos']['residual']
#                     tetl = np.float(df[(df['id'] == eqid) &
#                                     (df['alias'] == ss)]['tlatetearly'])
#                     epid = np.float(df[(df['id'] == eqid) &
#                                     (df['alias'] == ss)]['epidist'])
#                     mycc = _define_class(tetl, resid, classdict)
#                     #
#                     OUT.write(("%s,%s,%8.4f,%s,%8.4f,%7.3f,%d"+os.linesep) %
#                               (eqid, ss, epid, ph, tetl, resid, mycc))
#                     php[0]['pickclass'] = mycc
#                 #
#             progressBar(xx, ln-1, 'Classifying: ', 'DONE',
#                         decimals=2, barLength=15)
#     #
#     quake2cnv(cat, pdl, statdict=None,
#               phase_list=["P", "S", ],
#               fixed_class=False, use_alias=False, out_file="classified.cnv")
#     #
#     logger.warning("List of missing EVENTS:")
#     for _ev in missing_events:
#         logger.warning("- %s" % _ev)


"""
Let's use the next function as official:
- It should be used if you classify CNV after SEM (Step_7) and after
new magnitudes have been estimated.
- Use the final CNV (final catalog) together + RELOC files as
error file + ADAPT_newstats_CSV for the Tlate-Tearly values

"""


def classify_cnv_ajv(
                  cnvfile, relocfile, csvfile,
                  outfilecsv="classify_debug.csv", skip_evid=[],
                  classdict={'0': 0.1, '1': 0.2, '2': 0.4, '3': 0.8}):

    cnvobj = CNV(cnvfile)
    cat, pdl = cnvobj.get_catalog(), cnvobj.get_pick_dict_list()
    #
    relobj = RELOC(relocfile)  # dict[eqid][statname]   #ID 'I022': {'stat_name_velest': 'I022', 'phase_name': 'P', 'phase_weight': 0, 'obs_res': 0.202, 'obs_tt': 44.39, 'obs_delta': 298.06},
    #
    logger.info("Reading CSV")
    df = pd.read_csv(csvfile)
    missing_events = []
    #
    with open(outfilecsv, 'w') as OUT:
        OUT.write("EQID, STAT, EPIDIST, PHASE, DELTA, RESID, CLASS, ERRORNORM"+os.linesep)

        for xx, ev in enumerate(cat):
            eqid = ev.resource_id.id
            _work_pd = QU.extract_pickdict(pdl, eqid=eqid)
            if not _work_pd:
                missing_events.append(ev)
                continue
            if eqid in skip_evid:
                logger.warning("EVENT:  %s  ---> Skipped" % eqid)
                continue
            #
            for ss, pds in _work_pd.items():
                for ph, php in pds.items():
                    # Usually, as the dict is from VELEST, the cnv2quake
                    # script is linking only VEL_P and VEL_S for P and S.
                    # In particular, only the first occurence (index 0)
                    resid = relobj.extract_residual(eqid, ss, ph[-1], indexnum=0)
                    tetl = np.float(df[(df['id'] == eqid) &
                                    (df['alias'] == ss)]['tlatetearly'])
                    epid = np.float(df[(df['id'] == eqid) &
                                    (df['alias'] == ss)]['epidist'])
                    mycc, errnorm = _define_class(tetl, resid, classdict)
                    #
                    OUT.write(("%s,%s,%8.4f,%s,%8.4f,%7.3f,%d,%.3f"+os.linesep) %
                              (eqid, ss, epid, ph, tetl, resid, mycc, errnorm))
                    php[0]['pickclass'] = mycc
                #
            progressBar(xx, len(cat)-1, 'Classifying: ', 'DONE',
                        decimals=2, barLength=15)
    #
    quake2cnv(cat, pdl, statdict=None,
              phase_list=["VEL_P", "VEL_S", ],
              fixed_class=False, use_alias=False, out_file="classified.cnv")
    #
    logger.warning("List of missing EVENTS:")
    for _ev in missing_events:
        logger.warning("- %s" % _ev)


class CNV(object):
    """ opev = OP.Event """
    def __init__(self, pathname=None):
        self.cnvpath = pathname
        self.cnv_catalog = opev.Catalog()
        self.cnv_pickList = []
        self.tot_eq = 0
        #
        if pathname:
            self.import_file(pathname)
            self.tot_eq = len(self.cnv_catalog)

    def _remove_empty_stations(self):
        for _pd in self.cnv_pickList:
            _pd.delete_empty_station()

    def _convert_geographic_coordinate(self, instr):
        """ Simply put N and E positive. Last carachter analyzed """
        direction = instr[-1]
        if direction.upper() in ("N", "E"):
            return np.float(instr[:-1])
        elif direction.upper() in ("S", "W"):
            return -np.float(instr[:-1])
        else:
            raise ValueError("Unknown DIRECTION:  %s" % direction.upper())

    def import_file(self, pathname):
        """ Import reloc file and split it into OPCATALOG and PickDict

        After some testing, I went for a full charachter-wise selection,
        rather than field based one.
        """

        self.cnv_catalog, self.cnv_pickList = cnv2quake(pathname,
                                                        info_store="cnvv-obj")
        self.tot_eq = len(self.cnv_catalog)

    def count_obs_per_stat(self, store_file=False):
        """ Utility to calculate the amount of observations per stations
            is contained in the CNV.
            Return a dictionary: {'statname': count, ...}
        """
        obs_dict = {}
        #
        for _pd in self.cnv_pickList:
            for stat in _pd.keys():
                if stat not in obs_dict.keys():
                    obs_dict[stat] = 1
                else:
                    obs_dict[stat] += 1
        #
        if store_file:
            with open("count_obs_per_stat.txt", "w") as OUT:
                for _k, _v in obs_dict.items():
                    OUT.write(("%5s %3d"+os.linesep) % (_k, _v))
        return obs_dict

    def remove_obs_by_residual(self, stares_file,
                               threshold=1.0):
        """ Remove by VELEST residual """
        if not self.cnv_catalog or not self.cnv_pickList:
            logger.error("ERROR: please import a CNV first !!!")
        else:
            with open(stares_file, "r") as INSTAT:
                logger.info("Removing faulty obs")
                for _xx, _ll in enumerate(INSTAT):
                    if _xx == 0 or _ll[0] == "#":
                        continue
                    #
                    fields = _ll.strip().split()
                    eqid = fields[-1]
                    res = fields[3]
                    statname = fields[0]
                    phasename = fields[1].upper()
                    if np.abs(np.float(res)) >= threshold:
                        # Seek the event, remove the obs
                        _pd = QU.extract_pickdict(self.cnv_pickList, eqid=eqid)
                        _pd.delete_pick(statname, "VEL_"+phasename, 0)  # MB only one phase per station when reading CNV, always! --> do not change 0

    def keep_first_arrival(self):
        """ This wrapper function will internally call the private
            PickContainer._sort_by_time and
            PickContainer._keep_first_only methods to keep only the
            first pick occurence in time for each station-phase pair
        """
        for pickdict in self.cnv_pickList:
            pickdict._sort_by_time()
            pickdict._keep_first_only()

    def export_to_file(self, outname, use_alias=False):
        """ Simply export out """

        if use_alias:
            use_alias_switch = True
            if isinstance(use_alias, str):
                alias_asc = loadPickleObj(use_alias)
            elif isinstance(use_alias, StatContainer):
                alias_asc = use_alias
            else:
                raise TypeError("If specified, 'use_alias' par must be a string "
                                "to an adapt StatContainer pickle, or a StatContainer itself!")
        else:
            use_alias_switch = False
            alias_asc = None
        #
        self._remove_empty_stations()
        quake2cnv(self.cnv_catalog, self.cnv_pickList, statdict=alias_asc,
                  phase_list=["VEL_P", "VEL_S"],
                  fixed_class=False, use_alias=use_alias_switch, out_file=outname)

    def get_catalog(self):
        if self.cnv_catalog:
            return self.cnv_catalog
        else:
            logger.warning("Missing CATALOG. Run `import_file` method first!")

    def get_pick_dict_list(self):
        if self.cnv_pickList:
            return self.cnv_pickList
        else:
            logger.warning("Missing PICKDICTLIST. Run `import_file` method first!")


class RELOC(object):
    """ opev = OP.Event """
    def __init__(self, pathname=None):
        self.relocpath = pathname
        self.reloc_catalog = opev.Catalog()
        self.reloc_pickList = []
        #
        self.tot_eq = 0
        self.mod_eq = 0
        self.same_eq = 0
        #
        if pathname:
            self.import_file(pathname)

    def _count_lines(self, filepath):
        """Return number of rows in a file. OS independent

        Args:
            filepath (str): directory path (absolute or relative) of the
                            object file
        Returns:
            count (int): number of line-count

        """
        count = 0
        with open(filepath, 'r') as f:
            for line in f:
                count += 1
        return count

    def _convert_geographic_coordinate(self, instr):
        """ Simply put N and E positive """
        direction = instr[-1]
        if direction.upper() in ("N", "E"):
            return np.float(instr[:-1])
        elif direction.upper() in ("S", "W"):
            return -np.float(instr[:-1])
        else:
            raise ValueError("Unknown DIRECTION:  %s" % direction.upper())

    def _cnv_export(self, rel_cat, rel_pick_list, outfile_name="cnv_eport.cnv"):
        """ Simply export catalog and picks into CNV format """
        quake2cnv(self.reloc_catalog, self.reloc_pickList,
                  statdict=None,
                  phase_list=["P", "S"],
                  fixed_class=False, use_alias=False, out_file=outfile_name)

    # def _sort_picklist_epicentral_distance():
    #     if not self.reloc_catalog or not self.reloc_pickList:
    #         raise QE.MissingAttribute("Missing RELOC catalog or pickdict!")
    #     #
    #     for pickd in self.reloc_pickList:

    def import_file(self, pathname):
        """ Import reloc file and split it into OPCATALOG and PickDict

        After some testing, I went for a full charachter-wise selection,
        rather than field based one.
        """
        linecount = self._count_lines(pathname)

        def __runprogressbar(idx):
            """ Simply run the progress bar """
            progressBar(idx, linecount-1, prefix='Import RELOC:',
                        suffix='Complete', barLength=15)

        with open(pathname, "r") as IN:
            inside_event = False
            eventlineidx = 1
            for xx, line in enumerate(IN):
                if line[0] == "1":
                    inside_event = True
                    eventlineidx = 1
                    eqid = line.strip().split()[-1]
                    # Init
                    ev = opev.Event(resource_id=eqid)
                    og = opev.Origin(resource_id=eqid)
                    oq = opev.OriginQuality()
                    mg = opev.Magnitude(resource_id=eqid)
                    pd = PickContainer(eqid, "VELEST-SEM", "RELOC")
                    pd.eqid = eqid

                elif line[0:5] == "  $$$":
                    inside_event = False
                    # Store all
                    self.reloc_catalog = self.reloc_catalog + ev
                    self.reloc_pickList.append(pd)
                    #
                    eventlineidx = 1  # reset event count
                    __runprogressbar(xx)
                    continue

                elif line == os.linesep or line[0] == "#":
                    # Empty line
                    __runprogressbar(xx)
                    continue

                if line.strip().split()[0] == "DELETED:":
                    __runprogressbar(xx)
                    continue

                if line.strip() == "ERROR: insufficient data to locate the quake!":
                    # False Alarm, reset everything
                    inside_event = False
                    eventlineidx = 1  # reset event count
                    __runprogressbar(xx)
                    continue

                # lls = line.strip().split()
                if inside_event and eventlineidx == 3:
                    # ============= Header
                    og.time = UTCDateTime("20"+line[1:20])
                    og.latitude = self._convert_geographic_coordinate(
                                    line[20:29])
                    og.longitude = self._convert_geographic_coordinate(
                                    line[29:38])
                    og.depth = np.float(line[39:46]) * KM  # converting to meters

                    # Magnitude
                    mg.magnitude_type = "MLv"
                    try:
                        mg.mag = np.float(line[48:51])
                    except ValueError:
                        # usually is it because
                        mg.mag = -9.99

                    # Origin Quality
                    oq.standard_error = np.float(line[63:68])
                    oq.azimuthal_gap = np.float(line[60:63])
                    og.quality = oq

                    # Append event and mag
                    ev.origins.append(og)
                    ev.magnitudes.append(mg)

                elif inside_event and eventlineidx >= 9:
                    # ============= Observations
                    # stat = lls[0]
                    # ppha = lls[4]
                    # pcla = np.int(lls[5])
                    # ptrt = np.float(lls[8])  # # too risky for number like 83.813100.685
                    # tstu = np.float(lls[-1])  # too risky for number like -46.098
                    stat = line[:6].strip()
                    ppha = line[21:22]
                    epid = np.float(line[7:11])
                    pcla = np.int(line[23:24])
                    ptrt = np.float(line[36:43])
                    pres = np.float(line[59:65])
                    tstu = np.float(line[79:])

                    # Create Dict
                    stat_dict = {
                        ppha: [{
                            'timeUTC_pick': og.time + ptrt,
                            'pickclass': pcla,
                            'general_infos': {'t_student_velest_sem': tstu,
                                              'epicentral_distance': epid,
                                              'residual': pres}
                           },
                        ]
                    }
                    pd.addStat(stat, stat_dict)

                # At the end of iteration
                eventlineidx += 1
                __runprogressbar(xx)
        # Close to READ
        self.tot_eq = len(self.reloc_catalog)
        self.same_eq = len(self.reloc_catalog)
        self.mod_eq = 0

    def filter_epicentral_distance(self, epi_thr=250, create_copy=False):
        """ Filter by Epicentral Distance:
            VELEST-SEM after a certain epicentral distance can show
            weird behavior on RELOC output that could mess the I/O on
            subsequent runs.
            This method aims to filter out station obs. based on their
            epicentral distance (epi_thr).
            If create_copy True a new picklist is created and returned
            otherwise (False) it will modify directly the
            self.reloc_picklist !!!
        """
        if not self.reloc_catalog or not self.reloc_pickList:
            raise QE.MissingAttribute("Missing RELOC catalog or pickdict!")
        #
        list_to_delete = []
        for pickd in self.reloc_pickList:
            eqtag = pickd.eqid
            for stat, statdct in pickd.items():
                for phnm, phpklst in statdct.items():
                    for idx, phpkdct in enumerate(phpklst):
                        epi = phpkdct['general_infos']['epicentral_distance']
                        if not epi <= epi_thr:
                            list_to_delete.append(
                                (eqtag, stat, phnm, idx, epi))

        # ---- Removing obs
        if create_copy:
            work_picklist = copy.deepcopy(self.reloc_pickList)
        else:
            work_picklist = self.reloc_pickList
        #
        if not list_to_delete:
            logger.info("EPI-DIST Filtering (%3.1f sec.): NO-OBS" %
                        np.abs(epi_thr))
            print("EPI-DIST Filtering (%3.1f sec.): NO-OBS" %
                  np.abs(epi_thr))
        else:
            logger.info("EPI-DIST Filtering (%3.1f sec.): %d obs." %
                        (np.abs(epi_thr), len(list_to_delete)))
            print("EPI-DIST Filtering (%3.1f sec.): %d obs." %
                  (np.abs(epi_thr), len(list_to_delete)))

            for xx in list_to_delete:
                eqtag, stat, phnm, idx, _ = xx[:]
                evpickd = [ii for ii in work_picklist if ii.eqid == eqtag][0]
                del evpickd[stat][phnm][idx]
                if len(evpickd[stat][phnm]) == 0:
                    del evpickd[stat][phnm]
                    if len(evpickd[stat]) == 0:
                        del evpickd[stat]
        #
        if create_copy:
            return work_picklist
        else:
            return True

    def filter_tstudent(self, tstu_thr=0.5,
                        create_stats=False,
                        split_exports=False):
        """ Filter by T-student --> modify directly the
            self.reloc_picklist !!!
        """

        def __pass_tstu__(pkdct):
            if (np.abs(
                 pkdct['general_infos']['t_student_velest_sem']) <= tstu_thr):
                return True
            else:
                return False

        if not self.reloc_catalog or not self.reloc_pickList:
            raise QE.MissingAttribute("Missing RELOC catalog or pickdict!")
        #
        list_to_delete = []
        for pickd in self.reloc_pickList:
            eqtag = pickd.eqid
            for stat, statdct in pickd.items():
                for phnm, phpklst in statdct.items():
                    for idx, phpkdct in enumerate(phpklst):
                        if not __pass_tstu__(phpkdct):
                            list_to_delete.append(
                                (eqtag, stat, phnm, idx,
                                 phpkdct['general_infos']['t_student_velest_sem']))

        # ---- Removing obs
        if not list_to_delete:
            logger.info("T-STUDENT Filtering (%3.1f sec.): NO-OBS" %
                        np.abs(tstu_thr))
            print("T-STUDENT Filtering (%3.1f sec.): NO-OBS" %
                  np.abs(tstu_thr))
        else:
            logger.info("T-STUDENT Filtering (%3.1f sec.): %d obs." %
                        (np.abs(tstu_thr), len(list_to_delete)))
            print("T-STUDENT Filtering (%3.1f sec.): %d obs." %
                  (np.abs(tstu_thr), len(list_to_delete)))

            for xx in list_to_delete:
                eqtag, stat, phnm, idx, _ = xx[:]
                evpickd = [ii for ii in self.reloc_pickList if ii.eqid == eqtag][0]
                del evpickd[stat][phnm][idx]
                if len(evpickd[stat][phnm]) == 0:
                    del evpickd[stat][phnm]
                    if len(evpickd[stat]) == 0:
                        del evpickd[stat]

        # --- Check which events remain the same
        all_id = sorted(set([eq.resource_id.id for eq in self.reloc_catalog]))
        mod_id = sorted(set([xx[0] for xx in list_to_delete]))
        same_id = sorted(set([eqid for eqid in all_id if eqid not in mod_id]))

        # --- Update attributes
        self.tot_eq = len(all_id)
        self.same_eq = len(same_id)
        self.mod_eq = len(mod_id)
        # -----------------------

        if split_exports:
            sameCat, modCat = opev.Catalog(), opev.Catalog()
            samePickList, modPickList = [], []
            for _id in same_id:
                sameCat += [_ev for _ev in self.reloc_catalog.events
                            if _ev.resource_id.id == _id][0]
                samePickList.append(
                    [ii for ii in self.reloc_pickList if ii.eqid == _id][0])

            for _id in mod_id:
                modCat += [_ev for _ev in self.reloc_catalog.events
                           if _ev.resource_id.id == _id][0]
                modPickList.append(
                    [ii for ii in self.reloc_pickList if ii.eqid == _id][0])

            # Export
            self._cnv_export(sameCat, samePickList,
                             outfile_name="RELOC.T-Student.filtering.same.cnv")
            self._cnv_export(modCat, modPickList,
                             outfile_name="RELOC.T-Student.filtering.modified.cnv")

        # --- Finally create stats
        if create_stats:
            with open("RELOC.filtering_tstudent.stats.txt", "w") as OUT:
                OUT.write(
                    ("@@@ Total Events:  %5d"+os.linesep) % len(all_id))
                OUT.write(
                    ("@@@ Modified Events:  %5d"+os.linesep) % len(mod_id))
                OUT.write(
                    ("@@@ Untouched Events:  %5d"+os.linesep) % len(same_id))

                OUT.write(
                    ("### Modified EQID"+os.linesep))
                for _id in mod_id:
                    OUT.write(("%s"+os.linesep) % _id)
                OUT.write(
                    ("### Untouched EQID"+os.linesep))
                for _id in same_id:
                    OUT.write(("%s"+os.linesep) % _id)

                if len(list_to_delete) > 0:
                    OUT.write(
                        ("@@@ T-STUDENT Filtering (%3.1f km): %d obs"+os.linesep) %
                        (np.abs(tstu_thr), len(list_to_delete)))
                    OUT.write(
                        ("### EQID  STATNAME  PHASE  IDX  TSTUD"+os.linesep))
                    for xx in list_to_delete:
                        eqtag, stat, phnm, idx, tstu = xx[:]
                        OUT.write(
                            ("%20s %6s %3s %2d %7.3f"+os.linesep) % (
                             eqtag, stat, phnm, idx, tstu))
        #
        return True

    def get_pick_list(self):
        return self.reloc_pickList

    def get_catalog(self):
        return self.reloc_catalog

    def export_to_cnv(self, outfile_name="reloc2cnv.cnv"):
        """ Export the entire RELOC in MEMORY to CNV file """
        self._cnv_export(outfile_name=outfile_name)

    def extract_residual(self, eqid, stat, phase, indexnum=0):
        """ This method returns the residual of a given station-phase
            pair for a given event. If event not included, will return
            None and raise a warning
        """

        if phase.lower() not in ("p", "s"):
            raise ValueError("Phase must be either 'P' or 'S' only!")
        #
        _tmp_pd = QU.extract_pickdict(self.reloc_pickList, eqid)
        _, _pk = _tmp_pd.getMatchingPick(stat, phase.upper(),
                                         indexnum=indexnum)[0]  # first occurence only
        return _pk['general_infos']['residual']


class STA(object):
    """ opev = OP.Event """
    def __init__(self, pathname=None):
        self.stapath = pathname
        # self.statcontainer = StatContainer()
        self.statcontainer = {}
        self.staheader = None
        self.tot_stat = 0
        #
        if pathname:
            self.import_file(pathname)

    def _count_lines(self, filepath):
        """Return number of rows in a file. OS independent

        Args:
            filepath (str): directory path (absolute or relative) of the
                            object file
        Returns:
            count (int): number of line-count

        """
        count = 0
        with open(filepath, 'r') as f:
            for line in f:
                count += 1
        return count

    def _convert_geographic_coordinate(self, instr):
        """ Simply put N and E positive """
        direction = instr[-1]
        if direction.upper() in ("N", "E"):
            return np.float(instr[:-1])
        elif direction.upper() in ("S", "W"):
            return -np.float(instr[:-1])
        else:
            raise ValueError("Unknown DIRECTION:  %s" % direction.upper())

    def _create_geografic_coordinate(self, infloat, direction):
        """ Simply put N and E positive """
        if infloat >= 0.0:
            if direction.lower() in ('lon', 'longitude'):
                return "%8.4fE" % np.abs(infloat)
            elif direction.lower() in ('lat', 'latitude'):
                return "%7.4fN" % np.abs(infloat)
            else:
                raise ValueError("Direction must be either 'lon' or 'lat' !!!")
        elif infloat < 0.0:
            if direction.lower() in ('lon', 'longitude'):
                return "%8.4fW" % np.abs(infloat)
            elif direction.lower() in ('lat', 'latitude'):
                return "%7.4fS" % np.abs(infloat)
            else:
                raise ValueError("Direction must be either 'lon' or 'lat' !!!")

    def _import_station_alias(infile):
        """ InFile must be a 2 column file

        ALIAS  FULLNAME
        ALIAS  FULLNAME ...

        """

        aliases = []
        with open(infile, 'r') as IN:
            for xx, ll in enumerate(IN):
                ff = ll.strip().split()
                aliases.append((ff[0], ff[1]))
        #
        print("successfully read %d Station Alias" % xx+1)
        return aliases

    def _sta2pydict(self, infile):
        """ dai

        (a4,f7.4,a1,1x,f8.4,a1,1x,i5,1x,i1,1x,i4,1x,f5.2,2x,f5.2)
        8X0142.7750N  17.3627E    77 1    1 -0.35   0.0           5     0
        8X0243.6176N  19.3690E   482 1    2  0.00   0.0           0     0

        """
        with open(infile, 'r') as IN:
            for xx, line in enumerate(IN):
                ll = line.strip()
                if xx == 0:
                    self.staheader = ll
                    continue
                if not ll:
                    continue

                # @@@ Name
                statname = ll[0:4]
                self.statcontainer[statname] = {}

                # @@@ Lat
                self.statcontainer[statname]['lat'] = self._convert_geographic_coordinate(ll[4:12])

                # @@@ Lon
                self.statcontainer[statname]['lon'] = self._convert_geographic_coordinate(ll[13:22])

                # @@@ Dep
                self.statcontainer[statname]['elev'] = np.int(ll[23:28])
                self.statcontainer[statname]['elevkm'] = np.int(ll[23:28]) / 1000.0

                self.statcontainer[statname]['used'] = np.int(ll[28:30])
                self.statcontainer[statname]['idx'] = np.int(ll[30:35])

                if self.statcontainer[statname]['idx'] == 9999:
                    self.statcontainer[statname]['isreference'] = True
                else:
                    self.statcontainer[statname]['isreference'] = False

                # @@@ Pcorr
                self.statcontainer[statname]['pcorr'] = np.float(ll[36:41])
                self.statcontainer[statname]['scorr'] = np.float(ll[42:47])

                # @@@ Pobs
                self.statcontainer[statname]['pobs'] = np.int(ll[54:59])
                self.statcontainer[statname]['sobs'] = np.float(ll[60:65])
        #
        return True

    def reset_stat_corr(self, stations_list=None, group="P", value=0.0):
        """ reset given stat """
        if stations_list:
            for ss in stations_list:
                if group.lower() == "p":
                    self.statcontainer[ss]["pcorr"] = value
                elif group.lower() == "s":
                    self.statcontainer[ss]["scorr"] = value
                else:
                    raise ValueError("Group must be either 'P' or 'S' --> %s" %
                                     group)
            return True
        else:
            return False

    def import_file(self, pathname):
        """ Load into memory and parsing stafile
        """
        self._sta2pydict(pathname)
        return True

    def export_sta(self, pathname):
        """ Write out statfile

        (a4,f7.4,a1,1x,f8.4,a1,1x,i5,1x,i1,1x,i4,1x,f5.2,2x,f5.2)
        8X0142.7750N  17.3627E    77 1    1 -0.35   0.0           5     0
        8X0243.6176N  19.3690E   482 1    2  0.00   0.0           0     0

        """
        REFSTAPASSED = False
        with open(pathname, "w") as OUT:
            OUT.write(("%s"+os.linesep) % self.staheader)
            for xx, ss in enumerate(self.statcontainer):
                if not self.statcontainer[ss]['isreference']:
                    if not REFSTAPASSED:
                        self.statcontainer[ss]['idx'] = xx+1
                    else:
                        self.statcontainer[ss]['idx'] = xx
                else:
                    REFSTAPASSED = True

                #
                OUT.write(("%4s%s %s%6d%2d%5d%6.02f%6.02f%12d%6d"+os.linesep) %
                          (ss,
                           self._create_geografic_coordinate(self.statcontainer[ss]['lat'], direction="lat"),
                           self._create_geografic_coordinate(self.statcontainer[ss]['lon'], direction="lon"),
                           self.statcontainer[ss]['elev'],
                           self.statcontainer[ss]['used'],
                           self.statcontainer[ss]['idx'],
                           self.statcontainer[ss]['pcorr'],
                           self.statcontainer[ss]['scorr'],
                           self.statcontainer[ss]['pobs'],
                           self.statcontainer[ss]['sobs']))
            #
            OUT.write(os.linesep*2)
        return True

    def import_station_corrections(self, staobj, full_match=True):
        """ This method will take the pcorr and scorr of
            the given stat-object into the current instance

            If full_match is given, in addition to the station code also
            the lon-lat-dep parameters will be checked. otherwise only
            the station-code will be used.

            Please note that the matching is done case-sensitive)
        """
        if not isinstance(staobj, sys.modules[__name__].STA):
            raise TypeError("Need a STA class instance. Given:  %s" %
                            type(staobj))
        #
        for ss, vals in self.statcontainer.items():
            # seek dictionary
            try:
                other_station = staobj.statcontainer[ss]
            except KeyError:
                logger.warning("Station %s not find in input. Skipping ..." % ss)
                continue
            #
            if full_match:
                if (other_station['elev'] != vals['elev'] or
                   other_station['lon'] != vals['lon'] or
                   other_station['lat'] != vals['lat']):
                    logger.warning("Station %s found. No Full-match! Skipping ..." %
                                   ss)
                    continue
            # Override
            vals['pcorr'] = other_station['pcorr']
            vals['scorr'] = other_station['scorr']

    def reset_station_corrections(self, stations_list=None,
                                  pobs_thr=20, sobs_thr=20,
                                  group="P", value=0.0):
        """ It will reset the pcorr and scorr of station with LESS than
            pobs/sobs observations
        """
        if group.lower() not in ("p", "s"):
            raise ValueError("Group must be either 'P' or 'S' --> %s" %
                             group)
        #
        if stations_list:
            keystat = [
                ss for ss in self.statcontainer.keys() if ss in stations_list]
        else:
            keystat = self.statcontainer.keys()
        #
        for ss in keystat:
            if (group.lower() == "p" and
               self.statcontainer[ss]['pobs'] < pobs_thr):
                self.statcontainer[ss]["pcorr"] = value
            elif (group.lower() == "s" and
                  self.statcontainer[ss]['sobs'] < sobs_thr):
                self.statcontainer[ss]["scorr"] = value


class VELEST(object):
    """ Utility class to handle VELEST runs in Python environment
        NB: Works for VELEST 4.5 bin!
    """

    def __init__(self, work_dir=None, bin_path=None,
                 reg_names=None, reg_coord=None,
                 cmn=None, mod=None, sta=None, cnv=None, cmn_par={},
                 project_name="ADAPT-VELEST", outputs_prefix="VELEST.out"):

        """
        #  -------  Class Attributes
        self.work_dir = work_dir
        self.bin_path = bin_path
        self.reg_names = reg_names
        self.reg_coord = reg_coord
        self.cmn = cmn
        self.mod = mod
        self.sta = sta
        self.cnv = cnv
        self.project_name = project_name
        self.outputs_prefix = outputs_prefix
        self.cmn_par = {}
        """
        self.ALLOWEDPARKEYS = (
            'olat', 'olon', 'icoordsystem', 'zshift', 'itrial', 'ztrial', 'ised',
            'neqs', 'nshot', 'rotate',
            'isingle', 'iresolcalc',
            'dmax', 'itopo', 'zmin', 'veladj', 'zadj', 'lowveloclay',
            'nsp', 'swtfac', 'vpvs', 'nmod',
            'othet', 'xythet', 'zthet', 'vthet', 'stathet',
            'iqc', 'iwt0', 'iwt1', 'iwt2', 'iwt3', 'iwt4', 'iwt5',
            'iqf', 'iwf0', 'iwf1', 'iwf2', 'iwf3', 'iwf4', 'iwf5',
            'nsinv', 'nshcor', 'nshfix', 'iuseelev', 'iusestacorr',
            'iturbo', 'icnvout', 'istaout', 'ismpout', 'ihypores',
            'irayout', 'idrvout', 'ialeout', 'idspout', 'irflout', 'irfrout', 'iresout',
            'delmin', 'ittmax', 'invertratio')

        # Allocate Attributes
        self.set_work_dir(work_dir)
        self.set_bin_path(bin_path)
        self.set_reg_names(reg_names)
        self.set_reg_coord(reg_coord)
        self.set_cmn(cmn)
        self.set_mod(mod)
        self.set_sta(sta)
        self.set_cnv(cnv)
        self.set_project_name(project_name)
        self.set_outputs_prefix(outputs_prefix)
        self.set_cmn_par(cmn_par)

    # ================================  PRIVATE
    def _count_events_cnv(self):
        cnt = 0
        if self.cnv:
            with self.cnv.open('r') as IN:
                for line in IN:
                    try:
                        _ = int(line[0])
                        cnt += 1
                    except ValueError:
                        pass
        return cnt

    def _default_cmn_parameters(self):
        """ Defaults values for VELEST, change it at your own peace """

        outdict = {}
        #
        outdict['olat'], outdict['olon'], outdict['icoordsystem'], outdict['zshift'], outdict['itrial'], outdict['ztrial'], outdict['ised'] =\
            46.0, -9.0, 0, 0.000, 0, 0.00, 0
        outdict['neqs'], outdict['nshot'], outdict['rotate'] =\
            1, 0, 0
        outdict['isingle'], outdict['iresolcalc'] =\
            0, 0
        outdict['dmax'], outdict['itopo'], outdict['zmin'], outdict['veladj'], outdict['zadj'], outdict['lowveloclay'] =\
            1000.0, 0, 0.01, 0.20, 5.00, 0
        outdict['nsp'], outdict['swtfac'], outdict['vpvs'], outdict['nmod'] =\
            1, 0.50, 1.730, 1
        outdict['othet'], outdict['xythet'], outdict['zthet'], outdict['vthet'], outdict['stathet'] =\
            0.01, 0.01, 0.01, 10.0, 0.10
        outdict['iqc'], outdict['iwt0'], outdict['iwt1'], outdict['iwt2'], outdict['iwt3'], outdict['iwt4'], outdict['iwt5'] =\
            0, 1, 1, 1, 1, 4, 4
        outdict['iqf'], outdict['iwf0'], outdict['iwf1'], outdict['iwf2'], outdict['iwf3'], outdict['iwf4'], outdict['iwf5'] =\
            1, 1.00, 1.00, 1.00, 1.00, 0.00, 0.00
        outdict['nsinv'], outdict['nshcor'], outdict['nshfix'], outdict['iuseelev'], outdict['iusestacorr'] =\
            1, 0, 0, 1, 1
        outdict['iturbo'], outdict['icnvout'], outdict['istaout'], outdict['ismpout'], outdict['ihypores'] =\
            0, 1, 1, 0, 1
        outdict['irayout'], outdict['idrvout'], outdict['ialeout'], outdict['idspout'], outdict['irflout'], outdict['irfrout'], outdict['iresout'] =\
            0, 0, 0, 0, 0, 0, 0
        outdict['delmin'], outdict['ittmax'], outdict['invertratio'] =\
            0.010, 3, 1
        #
        return outdict

    # ================================  PUBLIC

    # ---- Set Functions
    def set_cmn(self, pth):
        # -- Bad Instance
        try:
            self.cmn = Path(pth)
            self.cmn = self.cmn.resolve()
        except TypeError:
            self.cmn = None
            return False

    def set_cmn_par(self, obj):
        # -- Bad Instance
        if obj and isinstance(obj, dict):
            self.cmn_par = obj
        else:
            self.cmn_par = self._default_cmn_parameters()

    def set_mod(self, pth):
        # -- Bad Instance
        try:
            tmp = Path(pth)
        except TypeError:
            self.mod = None
            return False
        # -- Missing File
        if not tmp.is_file() or not tmp.exists():
            raise QE.BadConfigurationFile("MOD file not found: [%s]" %
                                          tmp.absolute())
        self.mod = tmp
        self.mod = self.mod.resolve()

    def set_sta(self, pth):
        # -- Bad Instance
        try:
            tmp = Path(pth)
        except TypeError:
            self.sta = None
            return False
        # -- Missing File
        if not tmp.is_file() or not tmp.exists():
            raise QE.BadConfigurationFile("STA file not found: [%s]" %
                                          tmp.absolute())
        self.sta = tmp
        self.sta = self.sta.resolve()

    def set_cnv(self, pth):
        # -- Bad Instance
        try:
            tmp = Path(pth)
        except TypeError:
            self.cnv = None
            return False
        # -- Missing File
        if not tmp.is_file() or not tmp.exists():
            raise QE.BadConfigurationFile("CNV file not found: [%s]" %
                                          tmp.absolute())
        self.cnv = tmp
        self.cnv = self.cnv.resolve()

    def set_work_dir(self, pth):
        """ Here we just set the string path, checks are done inside the
            work() method
        """
        # -- Bad Instance
        try:
            self.work_dir = Path(pth)
            self.work_dir = self.work_dir.resolve()
        except TypeError:
            self.work_dir = None
            return False

    def set_bin_path(self, pth):
        """ Here we just set the string path, checks are done inside the
            work() method
        """
        # --- Bad Instance
        try:
            tmp = Path(pth)
        except TypeError:
            self.bin_path = None
            return False

        # --- Missing File
        if not tmp.is_file() or not tmp.exists():
            raise QE.BadConfigurationFile(
                "BIN-PATH file not found: [%s]" % tmp.absolute())

        # --- Not Executable
        if not os.access(tmp, os.X_OK):
            raise QE.BadConfigurationFile(
                "BIN-PATH not executable: [%s]" % tmp.absolute())
        self.bin_path = tmp
        self.bin_path = self.bin_path.resolve()

    def set_reg_names(self, pth):
        # -- Bad Instance
        try:
            tmp = Path(pth)
        except TypeError:
            self.reg_names = None
            return False
        # -- Missing File
        if not tmp.is_file() or not tmp.exists():
            raise QE.BadConfigurationFile(
                "REGION-NAMES file not found: [%s]" % tmp.absolute())
        self.reg_names = tmp

    def set_reg_coord(self, pth):
        # -- Bad Instance
        try:
            tmp = Path(pth)
        except TypeError:
            self.reg_coord = None
            return False
        # -- Missing File
        if not tmp.is_file() or not tmp.exists():
            raise QE.BadConfigurationFile(
                "REGION-COORD file not found: [%s]" % tmp.absolute())
        self.reg_coord = tmp

    def set_project_name(self, strname):
        if not isinstance(strname, str):
            raise QE.InvalidType(
                "PROJECT-NAME must be a string type: [%s]" % type(strname))
        self.project_name = strname

    def set_outputs_prefix(self, strname):
        if not isinstance(strname, str):
            raise QE.InvalidType(
                "OUTPUT-PREFIX must be a string type: [%s]" % type(strname))
        self.outputs_prefix = strname

    # ---- Get Functions
    def get_cmn(self):
        return self.cmn

    def get_cmn_par(self):
        return self.cmn_par

    def get_mod(self):
        return self.mod

    def get_sta(self):
        return self.sta

    def get_cnv(self):
        return self.cnv

    def get_work_dir(self):
        return self.work_dir

    def get_bin_path(self):
        return self.bin_path

    def get_reg_names(self):
        return self.reg_names

    def get_reg_coord(self):
        return self.reg_coord

    def get_project_name(self):
        return self.project_name

    def get_outputs_prefix(self):
        return self.outputs_prefix

    # ---- Set-Up
    def prepare(self, create_missing_dir=True):
        """ Build Up the experiment. You just need to call this once
            per experiment.
        """

        # -- Missing File
        # if not tmp.is_file() or not tmp.exists():
        #     raise QE.BadConfigurationFile("CMN file not found: [%s]" %
        #                                   tmp.absolute())

        if not self.work_dir:
            raise QE.MissingAttribute("Missing WORKDIR! Specify it beforehand")
        if self.work_dir.is_dir() and self.work_dir.exists():
            raise QE.InvalidParameter(
                "WORK-DIR already exists: [%s]" % self.work_dir.absolute())
        else:
            self.work_dir.mkdir(parents=create_missing_dir)

        # If CMN is already a file, copy it in folder
        if self.cmn and self.cmn.is_file() and self.cmn.exists():
            shutil.copy(self.cmn, self.work_dir / 'velest.cmn')
            self.cmn = self.work_dir / 'velest.cmn'
        else:
            self.cmn = self.work_dir / 'velest.cmn'
        #
        logger.info("Now the CMN class attribute points at: [%s]" %
                    self.cmn)

    def change_cmn_par(self, **kwargs):
        """ Update the CMN """
        def __check_keys__(kk):
            if kk.lower() not in self.ALLOWEDPARKEYS:
                raise QE.InvalidVariable("Not a valid CMN-PAR key: [%s]" % kk)
            else:
                return True
        #
        for _k, _v in kwargs.items():
            if __check_keys__(_k):
                self.cmn_par[_k] = _v
        #
        logger.warning("Changes to velest.cmn are loaded. To be effective call"
                       " the `create_cmn()` method!")

    def create_cmn(self, mode):
        """ This method will always create a velest.cmn file inside the
            self.work_dir. If already present, it will override it !!!

https://stackoverflow.com/questions/58839678/how-to-get-the-relative-path-between-two-absolute-paths-in-python-using-pathlib
        """
        if not self.cmn_par:
            raise QE.MissingAttribute(
                "The CMN-PAR is missing! Set-it beforehand")
        #
        if mode.lower() in ('jhd', 'coupled', 'join-hypocenter-determination'):
            self.change_cmn_par(neqs=self._count_events_cnv(),
                                isingle=0,
                                nsinv=1,
                                iturbo=0,
                                ismpout=0)
        elif mode.lower() in ('sem', 'single', 'single-event-mode'):
            self.change_cmn_par(neqs=self._count_events_cnv(),
                                isingle=1,
                                nsinv=0,
                                iturbo=1,
                                ismpout=1,
                                invertratio=0,
                                ittmax=99)
        else:
            raise QE.InvalidVariable(
                "VELEST mode can be either 'JHD' or 'SEM' !!!")
        #
        with self.cmn.open('w') as OUT:
            now = UTCDateTime()
            OUT.write("***      V E L E S T   4.5    control-file"+os.linesep)
            OUT.write("***  created by: ADAPT-VelestTools library"+os.linesep)
            OUT.write(("***     version: %s"+os.linesep) % __version__)
            OUT.write(("***        date: %4d-%02d-%02d @ %02d:%02d:%02d"+os.linesep) % (
                        now.year, now.month, now.day,
                        now.hour, now.minute,now.second))
            OUT.write("*"*65+os.linesep)

            # line 1
            OUT.write(("%s"+os.linesep) % self.project_name)

            # line 2
            OUT.write("***  olat  olon  icoordsystem  zshift  itrial  ztrial  ised"+os.linesep)
            OUT.write(
                ("  %8.4f  %9.4f  %1d  %5.4f  %1d  %4.2f  %1d"+os.linesep) % (
                        self.cmn_par['olat'], self.cmn_par['olon'],
                        self.cmn_par['icoordsystem'], self.cmn_par['zshift'],
                        self.cmn_par['itrial'], self.cmn_par['ztrial'],
                        self.cmn_par['ised']))
            # line 3
            OUT.write("***  neqs  nshot  rotate"+os.linesep)
            OUT.write(("  %1d  %1d  %1d"+os.linesep) % (
                self.cmn_par['neqs'], self.cmn_par['nshot'],
                self.cmn_par['rotate']))
            # line 4
            OUT.write("***  isingle  iresolcalc"+os.linesep)
            OUT.write(("  %1d  %1d"+os.linesep) % (
                self.cmn_par['isingle'], self.cmn_par['iresolcalc']))
            # line 5
            OUT.write("***  dmax  itopo  zmin  veladj  zadj  lowveloclay"+os.linesep)
            OUT.write(("  %7.2f  %1d  %5.2f  %5.2f  %5.2f  %1d"+os.linesep) % (
                self.cmn_par['dmax'], self.cmn_par['itopo'],
                self.cmn_par['zmin'], self.cmn_par['veladj'],
                self.cmn_par['zadj'], self.cmn_par['lowveloclay']))
            # line 6
            OUT.write("*** nsp    swtfac   vpvs       nmod"+os.linesep)
            OUT.write(("  %1d  %5.2f  %5.3f  %1d"+os.linesep) % (
                self.cmn_par['nsp'], self.cmn_par['swtfac'],
                self.cmn_par['vpvs'], self.cmn_par['nmod']))
            # line 7
            OUT.write("***   othet   xythet    zthet    vthet   stathet"+os.linesep)
            OUT.write(("  %5.2f  %5.2f  %5.2f  %6.2f  %5.2f"+os.linesep) % (
                self.cmn_par['othet'], self.cmn_par['xythet'],
                self.cmn_par['zthet'], self.cmn_par['vthet'],
                self.cmn_par['stathet']))
            # line 8
            OUT.write("***  iqc  iwt0  iwt1  iwt2  iwt3  iwt4  iwt5"+os.linesep)
            OUT.write(("  %1d  %1d  %1d  %1d  %1d  %1d  %1d"+os.linesep) % (
                self.cmn_par['iqc'], self.cmn_par['iwt0'],
                self.cmn_par['iwt1'], self.cmn_par['iwt2'],
                self.cmn_par['iwt3'], self.cmn_par['iwt4'],
                self.cmn_par['iwt5']))
            # line 9
            OUT.write("***  iqf  iwf0  iwf1  iwf2  iwf3  iwf4  iwf5"+os.linesep)
            OUT.write(("  %1d  %1d  %1d  %1d  %1d  %1d  %1d"+os.linesep) % (
                self.cmn_par['iqf'], self.cmn_par['iwf0'],
                self.cmn_par['iwf1'], self.cmn_par['iwf2'],
                self.cmn_par['iwf3'], self.cmn_par['iwf4'],
                self.cmn_par['iwf5']))
            # line 10
            OUT.write("***  nsinv  nshcor  nshfix  iuseelev  iusestacorr"+os.linesep)
            OUT.write(("  %1d  %1d  %1d  %1d  %1d"+os.linesep) % (
                self.cmn_par['nsinv'], self.cmn_par['nshcor'],
                self.cmn_par['nshfix'], self.cmn_par['iuseelev'],
                self.cmn_par['iusestacorr']))
            # line 11
            OUT.write("***  iturbo  icnvout  istaout  ismpout  ihypores"+os.linesep)
            OUT.write(("  %1d  %1d  %1d  %1d  %1d"+os.linesep) % (
                self.cmn_par['iturbo'], self.cmn_par['icnvout'],
                self.cmn_par['istaout'], self.cmn_par['ismpout'],
                self.cmn_par['ihypores']))
            # line 12
            OUT.write("***  irayout  idrvout  ialeout  idspout  irflout  irfrout   iresout"+os.linesep)
            OUT.write(("  %1d  %1d  %1d  %1d  %1d  %1d  %1d"+os.linesep) % (
                self.cmn_par['irayout'], self.cmn_par['idrvout'],
                self.cmn_par['ialeout'], self.cmn_par['idspout'],
                self.cmn_par['irflout'], self.cmn_par['irfrout'],
                self.cmn_par['iresout']))
            # line 13
            OUT.write("***  delmin  ittmax  invertratio"+os.linesep)
            OUT.write(("  %5.3f  %1d  %1d"+os.linesep) % (
                self.cmn_par['delmin'], self.cmn_par['ittmax'],
                self.cmn_par['invertratio']))

            # MOD
            OUT.write("*** Input 1D velocity model(s):"+os.linesep)
            if not self.mod:
                OUT.write(os.linesep)
                logger.warning("Empy MOD path --> set it before run VELEST")
            else:
                if len(str(self.mod)) > 65:
                    relpath = os.path.relpath(self.mod, start=self.work_dir)
                    OUT.write(("%s"+os.linesep) % relpath)
                else:
                    OUT.write(("%s"+os.linesep) % self.mod)

            # STA
            OUT.write("*** Station file:"+os.linesep)
            if not self.sta:
                OUT.write(os.linesep)
                logger.warning("Empy STA path --> set it before run VELEST")
            else:
                if len(str(self.sta)) > 65:
                    relpath = os.path.relpath(self.sta, start=self.work_dir)
                    OUT.write(("%s"+os.linesep) % relpath)
                else:
                    OUT.write(("%s"+os.linesep) % self.sta)

            # SeismoFile (empty)
            OUT.write("*** Seismo file:"+os.linesep*2)

            # REG-NAMES
            OUT.write("*** File with region names:"+os.linesep)
            if not self.reg_names:
                OUT.write(os.linesep)
                logger.warning("Empy REG-NAMES path --> set it before run VELEST")
            else:
                if len(str(self.reg_names)) > 65:
                    relpath = os.path.relpath(self.reg_names, start=self.work_dir)
                    OUT.write(("%s"+os.linesep) % relpath)
                else:
                    OUT.write(("%s"+os.linesep) % self.reg_names)

            # REG-COORD
            OUT.write("*** File with region coordinates:"+os.linesep)
            if not self.reg_coord:
                OUT.write(os.linesep)
                logger.warning("Empy REG-COORD path --> set it before run VELEST")
            else:
                if len(str(self.reg_coord)) > 65:
                    relpath = os.path.relpath(self.reg_coord, start=self.work_dir)
                    OUT.write(("%s"+os.linesep) % relpath)
                else:
                    OUT.write(("%s"+os.linesep) % self.reg_coord)

            # TopoFile (empty)
            OUT.write("*** File #1 with topo data:"+os.linesep*2)
            OUT.write("*** File #2 with topo data:"+os.linesep*2)

            # CNV
            OUT.write("*** Earthquake input file:"+os.linesep)
            if not self.cnv:
                OUT.write(os.linesep)
                logger.warning("Empy CNV path --> set it before run VELEST")
            else:
                if len(str(self.cnv)) > 65:
                    relpath = os.path.relpath(self.cnv, start=self.work_dir)
                    OUT.write(("%s"+os.linesep) % relpath)
                else:
                    OUT.write(("%s"+os.linesep) % self.cnv)

            # Shot data (empy)
            OUT.write("*** File with Shot data:"+os.linesep*2)

            # OUTPUT-PREFIX
            OUT.write("*** Output files prefix:"+os.linesep)
            OUT.write(("%s"+os.linesep) % self.outputs_prefix)

            # Close
            OUT.write("*"*65+os.linesep)

    def work(self, log_file=""):
        # ---- Work Functions
        """ Check everything is ok and Run
        docu: https://docs.python.org/3/library/subprocess.html
        subprocess.run()
        """
        nowdir = Path.cwd()
        os.chdir(self.work_dir)
        if log_file and isinstance(log_file, str):
            with open(self.work_dir / "vel.log", "w") as LOG:
                result = subprocess.run(str(self.bin_path), stdout=LOG)
        else:
            result = subprocess.run(str(self.bin_path))
        os.chdir(nowdir.resolve())
        return result


# from obspy.core.event import Origin
# origin = Origin()
# origin.resource_id = 'smi:ch.ethz.sed/origin/37465'
# origin.time = UTCDateTime(0)
# origin.latitude = 12
# origin.latitude_errors.uncertainty = 0.01
# origin.latitude_errors.confidence_level = 95.0
# origin.longitude = 42
# origin.depth_type = 'from location'
# print(origin)


"""
1 E V E N T   N R .        1                            0                    0  EVID: KP201601262326
0 DATE  ORIGIN   TIME   LAT      LON     DEPTH  MAG  NO  DM GAP  RMS   ALE D-SPR
 160126 23:26:21.783 47.0736N 12.4738E  43.231  2.5  11  67  82 0.68  1.75  0.00
0  ERX  ERY  ERZ Q SQD  ADJ  IN NR  AVR   AAR  NM AVXM  SDXM IT
  -0.1  0.1 -0.0 B C/A  0.41  0 11  0.00  0.53  0  0.0  0.0  99
0 F-E NR: 546 AUSTRIA
0 STN  DIST AZM AIN PRMK HRMN  P-SEC  TPOBS  TPCAL  -TSCOR  P-RES   P-WT IMP STURES
        AMX PRX     SRMK XMAG  S-SEC  TSOBS  TSCAL  -TSCOR  S-RES   S-WT IMP STURES
  OE04   67 270 109  P 0 2326 34.120 12.337 12.814  0.080  -0.557   1.00 0.3986 -0.718
  BW31   76 328 105  P 0 2326 35.160 13.377 13.834  0.170  -0.627   1.00 0.1776 -0.691
  BW07   77 347 105  P 0 2326 37.170 15.387 13.949  0.340   1.098   1.00 0.1783  1.211
  BW22   79 341 104  P 0 2326 36.710 14.927 14.275  0.530   0.122   1.00 0.1732  0.134
  BW13   79 339 104  P 0 2326 36.450 14.667 14.274  0.330   0.063   1.00 0.1727  0.069
  BW32   82 341 103  P 0 2326 36.860 15.077 14.636  0.620  -0.178   1.00 0.1735 -0.196
  I176   93 193  69  P 0 2326 38.170 16.387 15.985 -0.430   0.832   1.00 0.4648  1.137
  I050  187 127  69  P 0 2326 50.530 28.747 28.070  0.000   0.677   1.00 0.2607  0.787
  GR08  194  70  69  P 0 2326 50.550 28.767 28.784  0.370  -0.387   1.00 0.2923 -0.460
  OE02  200  83  69  P 0 2326 51.980 30.197 29.687  0.360   0.150   1.00 0.2738  0.176
  I167  201 136  69  P 0 2326 50.410 28.627 29.717  0.060  -1.150   1.00 0.2691 -1.345
  $$$   VELEST-Version 4.5 ETH-AUGUST2017 located at: May 13 14:27:57 2021


1 E V E N T   N R .        2                            0                    0  EVID: KP201604100219
"""
