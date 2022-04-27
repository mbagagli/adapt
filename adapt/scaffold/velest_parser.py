import os
import sys
import copy
import numpy as np
import logging
#
import pandas as pd
from obspy import UTCDateTime
import obspy.core.event as opev
from obspy.geodetics.base import kilometer2degrees, degrees2kilometers
#
import adapt.parser as QP
from adapt.utils import progressBar, loadPickleObj, savePickleObj
import adapt.errors as QE


logger = logging.getLogger(__name__)


"""
This modules aims to interface the adapt modules with the VELEST
software. They also aim to modify cnv internally.

In particular this module contains the parsing routines to import/export
file formats.

"""

# =================== Format Lines
FMT_HEADER_CNV = "%2d%02d%02d %02d%02d %05.2f %7.4f%s %8.4f%s%7.2f%7.2f    %3d%10.2f  EVID: %s"
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


def _perturbate_event_catalog(obcat, quantity, create_copy=False):
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

    EPI = kilometer2degrees(quantity)
    DEP = quantity * 1000  # km ==> m

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
                direction = "minus"
            elif direction == "minus":
                catalog[xx].origins[0].depth -= DEP
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

                        if fixed_class:
                            cla = np.int(fixed_class)
                        else:
                            cla = np.int(np.round(phasepickslist[0]['pickclass']))

                        try:
                            pht = phasepickslist[0]['timeUTC_pick'] - ot
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

                    if fixed_class:
                        cla = np.int(fixed_class)
                    else:
                        cla = np.int(np.round(phasepickslist[0]['pickclass']))

                    try:
                        pht = phasepickslist[0]['timeUTC_pick'] - ot
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
    retcat = _perturbate_event_catalog(catalog, perturb)
    quake2cnv(retcat, evpdcollection, statdict=None,
              phase_list=["VEL_P", "VEL_S", ],
              fixed_class=False, use_alias=False, out_file=outcnv)
    return retcat, evpdcollection


def import_stations_corrections(statdict, velstafile, minp=1, mins=1,
                                ignore_missing_stations=False,
                                export_obj=False):
    """ This function will add the station corrections into the given
        ADAPT.StatContainer object.
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
                        barLength=20)

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
              out_file="quake2cnv_out.cnv", use_alias=False):
    """ Routine to convert back to CNV

    pickdlist is a list of pickobject. Make sure the resource_id and the
        eqid attribute of PickCOntainer coincide

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
                            decimals=2, barLength=20)

            eqid = ee.origins[0].resource_id.id
            try:
                evpickd = [ii for ii in pickdlist if ii.eqid == eqid][0]
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
    logger.info("Converting CNV to ADAPT PickContainer and ObsPy Catalog")
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
        obsDict = {}
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
            obsDict[statnm] = {('VEL_' + statph): [
                                            {'pickclass': statcl,
                                             'timeUTC_pick': stattt},
                                                  ]
                               }
        # ... finally append
        pkcnt.append(QP.dict2PickContainer(
                                obsDict, _eqid, info_store, "VELEST"))
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
