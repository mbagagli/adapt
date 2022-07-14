"""
In this module are contained all the necessary
funtions and methods to convert from
"""

import os
import logging
import copy
from pathlib import Path
#
import adapt.database as QD
import adapt.errors as QE
#
from obspy import read_inventory
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
import obspy.core.event as opev
from obspy import UTCDateTime
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------


def manupde2PickContainer(filepath, eqid, eqtag, metastatdict=None):
    """ This method willl return a PickContainer object to be used
        throughtout the picker.
        Using ManuPDE is more precise in terms of numbers.

        This method should be preferred for ease of use and for future
        maintenance, plus more accuracy on some fields (i.e. DEPTH, RMS)
    """
    stationlist = []
    pickdict = {}
    #
    with open(filepath, "r") as IN:
        for xx, line in enumerate(IN):
            if xx == 5:
                line = line.strip()
                pieces = line.split()  # split by space and strip each
                otm = UTCDateTime(pieces[0] + pieces[1])

            elif xx >= 12:
                line = line.strip()
                if not line:
                    # EOF reached
                    break
                pieces = line.split()
                #####################

                station = pieces[0]
                # Next line mod. to improve tag extraction 15.11.2018
                #   phaseName = line[8:12].strip()
                # v0.3.1
                if eqtag.lower() in ("reference", "ref"):
                    phaseName = "Reference_" + pieces[1]
                else:
                    phaseName = "Seiscomp_" + pieces[1]

                phasett = copy.deepcopy(otm)
                phasett.minute = int(pieces[4])
                phasett.second = int(pieces[5].split(".")[0])
                phasett.microsecond = int(pieces[5].split(".")[1]) * 10000

                groupphase = pieces[2]
                if len(groupphase) == 1:
                    emim = groupphase
                    polarity = None
                elif len(groupphase) == 2:
                    emim = groupphase[0]
                    polarity = groupphase[1]
                else:
                    raise QE.InvalidParameter("Missing GROUPPHASE var")

                weight = None
                classNum = None

                if pieces[3] == "0":
                    usepick = False
                elif pieces[3] == "1":
                    usepick = True

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
                simplepickdict["timeUTC_pick"] = phasett

                # --- MB: The next part seems to relate only on MANUPICK
                # try:
                #     earlytt = float(line[72:79].strip())
                #     simplepickdict["timeUTC_early"] = refertime + \
                #         phasett + earlytt
                # # missing (could not convert str to float)
                # except (ValueError, IndexError):
                #     earlytt = None
                #     simplepickdict["timeUTC_early"] = None
                # try:
                #     latett = float(line[80:87].strip())
                #     simplepickdict["timeUTC_late"] = refertime + \
                #         phasett + latett
                # except (ValueError, IndexError):
                #     latett = None
                #     simplepickdict["timeUTC_late"] = None
                # ----------------------------------------------------

                # Append loop results
                pickdict[station][phaseName].append(simplepickdict)
                if station not in stationlist:
                    stationlist.append(station)
    # Convert it
    stationlist.sort()

    # v0.3.1
    if eqtag.lower() in ("reference", "ref"):
        outdict = dict2PickContainer(pickdict, eqid, eqtag, "PREDREF")
    else:
        outdict = dict2PickContainer(pickdict, eqid, eqtag, "SC3AUTO")

    # 27042019 modifyied
    if isinstance(metastatdict, QD.StatContainer_Event):
        if eqtag.lower() in ("reference", "ref"):
            for ii in stationlist:
                metastatdict.addStat(
                    eqid, ii, isreference=True, isautomatic=False)
        else:
            for ii in stationlist:
                metastatdict.addStat(
                    eqid, ii, isreference=False, isautomatic=True)
    #
    return outdict, stationlist


def manupde2Event(filepath, eqid, eqlabel, fixed_depth=None):
    """ This method willl return an ObsPy.Event object to be used
        throughtout the picker. Using ManuPDE is more precise in terms
        of floats numbers.

        This method should be preferred for ease of use and for future
        maintenance, plus more accuracy on some fields (i.e. DEPTH, RMS)
    """
    tmpDict = {}
    with open(filepath, "r") as IN:
        for xx, line in enumerate(IN):
            if xx == 5:
                line = line.strip()
                pieces = line.split()  # split by space and strip each
                #
                tmpDict = {'origintime': UTCDateTime(pieces[0] +
                                                     pieces[1]),
                           'eventscore': None,             # To be added
                           # Coordinates 'N'/'S' is [20]
                           'latGEO': pieces[2],
                           # Coordinates 'E'/'W' is [27]
                           'lonGEO': pieces[3],
                           # m
                           'dep': float(pieces[4]) * 1000,  # meters
                           'mag': float(pieces[7]),
                           'magType': 'Mlv',
                           'id': eqid,
                           'eqlabel': eqlabel
                           }
                #
                if tmpDict['latGEO'][-1] == "N":
                    tmpDict['lat'] = float(tmpDict['latGEO'][:-1])
                elif line[20] == "S":
                    tmpDict['lat'] = float("-"+tmpDict['latGEO'][:-1])
                else:
                    logger.error(("wrong LAT direction %s --> " +
                                  tmpDict['latGEO'][-1]) %
                                 (filepath.split(os.sep)[-1]))
                    raise QE.InvalidParameter

                # LON Coordinates 'E'/'W' is [27]
                if tmpDict['lonGEO'][-1] == "W":
                    tmpDict['lon'] = float("-"+tmpDict['lonGEO'][:-1])
                elif tmpDict['lonGEO'][-1] == "E":
                    tmpDict['lon'] = float(tmpDict['lonGEO'][:-1])
                else:
                    logger.error(("wrong LON direction %s --> " +
                                 tmpDict['lonGEO'][-1]) %
                                 (filepath.split(os.sep)[-1]))
                    return None

                # v0.5.1 add GAP e RMS
                # skipping the first string charachetr after SED
                tmpDict['rms'] = float(pieces[8])
                tmpDict['gap'] = float(pieces[9])

                # v0.6.16 switch for production
                if fixed_depth:
                    if isinstance(fixed_depth, (int, float)):
                        logger.info("Event depth fixed at: %6.2f (km)" %
                                    fixed_depth)
                        tmpDict['dep'] = float(fixed_depth) * 1000
                    else:
                        raise QE.InvalidParameter(
                            "fixed_depth parameter must be numeric!")
    #
    outEv = dict2Event(tmpDict)
    return outEv


def manuloc2Event(filepath, eqid, eqlabel, fixed_depth=None):
    """
    This parsing method will convert input MANULOC
    information inside an ObsPy 'Event' Object

    INPUT:
        :type filepath: str
        :param filepath: path of the MANULOC (SED/ETHZ) ascii file

    e.g:
    "20000110053009846145N007385E00831Ml0000000000000000000SEDE000000000000           s12cmix"
     YYYYMMDDhhmnssm  lat lon    dddMM
                 seconds.m            depth(km) magnitude(M.M)
    """
    # Extract info from file
    tmpDict = {}
    with open(filepath, "r") as IN:
        for xx, line in enumerate(IN):
            if xx == 0:
                line = line.strip()
                tmpDict = {'origintime': UTCDateTime(line[0:14] + "." +
                                                     line[14]),
                           'eventscore': None,             # To be added
                           # Coordinates 'N'/'S' is [20]
                           'latGEO': line[15:17] + "." + line[17:21],
                           # Coordinates 'E'/'W' is [27]
                           'lonGEO': line[21:24] + "." + line[24:28],
                           # m
                           'dep': float(line[28:31]) * 1000,
                           'mag': float(line[31] + "." + line[32]),
                           'magType': line[33:35],
                           'id': eqid,
                           'eqlabel': eqlabel
                           }
                # LAT Coordinates 'N'/'S' is [20]
                if line[20] == "N":
                    tmpDict['lat'] = float(line[15:17] + "." + line[17:20])
                elif line[20] == "S":
                    tmpDict['lat'] = float(
                        "-" + line[15:17] + "." + line[17:20])
                else:
                    logger.error(("wrong LAT direction %s --> " + line[20]) %
                                 (filepath.split(os.sep)[-1]))
                    return None

                # LON Coordinates 'E'/'W' is [27]
                if line[27] == "W":
                    tmpDict['lon'] = float(
                        "-" + line[21:24] + "." + line[24:27])
                elif line[27] == "E":
                    tmpDict['lon'] = float(line[21:24] + "." + line[24:27])
                else:
                    logger.error(("wrong LON direction %s --> " + line[27]) %
                                 (filepath.split(os.sep)[-1]))
                    return None

                # v0.5.1 add GAP e RMS
                line_after_SED = line.split('SED')[-1]
                # skipping the first string charachetr after SED
                tmpDict['rms'] = float(line_after_SED[1:4])*0.01
                tmpDict['gap'] = float(line_after_SED[4:7])

                # v0.6.16 switch for production
                if fixed_depth:
                    if isinstance(fixed_depth, (int, float)):
                        logger.info("Event depth fixed at: %6.2f (km)" %
                                    fixed_depth)
                        tmpDict['dep'] = float(fixed_depth) * 1000
                    else:
                        raise QE.InvalidParameter(
                            "fixed_depth parameter must be numeric!")
    #
    outEv = dict2Event(tmpDict)
    return outEv


def manupick2PickContainer(filepath, eqid, eqtag, metastatdict=None, original_phase_naming=False):
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

                # MB: to allow the use only the real phase-name
                if original_phase_naming:
                    phaseName = line[8:12].strip()
                else:
                    # Next line mod. to improve tag extraction 15.11.2018
                    #   phaseName = line[8:12].strip()
                    # v0.3.1
                    if eqtag.lower() in ("reference", "ref"):
                        phaseName = "Reference_" + line[8:12].strip()
                    else:
                        phaseName = "Seiscomp_" + line[8:12].strip()

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

    # v0.3.1
    if eqtag.lower() in ("reference", "ref"):
        outdict = dict2PickContainer(pickdict, eqid, eqtag, "PREDREF")
    else:
        outdict = dict2PickContainer(pickdict, eqid, eqtag, "SC3AUTO")

    # 07122018 add metadata
    # 27042019 modifyied
    if isinstance(metastatdict, QD.StatContainer_Event):
        if eqtag.lower() in ("reference", "ref"):
            for ii in stationlist:
                metastatdict.addStat(
                    eqid, ii, isreference=True, isautomatic=False)
        else:
            for ii in stationlist:
                metastatdict.addStat(
                    eqid, ii, isreference=False, isautomatic=True)
    #
    return outdict, stationlist


def dict2Event(userdict):
    """
    Internal Module to pass from an user created Dict
    to a custom Obspy Event object.

    :type userdict: dict
    :param userdict: usermade dictionary that MUST contain
    the subsequent keys ('id', 'origintime', 'lat', 'lon', 'dep',
                         'rms', 'gap', 'mag', 'magType')

    The method return an <obspy Event> object
    """
    # Create Origin class
    origin = opev.Origin()
    origin.resource_id = userdict['id']
    origin.time = userdict['origintime']
    origin.latitude = userdict['lat']
    origin.longitude = userdict['lon']
    origin.depth = round(userdict['dep'])  # unit meter
    # Create OriginError class
    origin_qual = opev.OriginQuality(standard_error=userdict['rms'],
                                     azimuthal_gap=userdict['gap'])
    # Fill GAP and RMS
    origin.quality = origin_qual
    # Create Magnitude class
    magnitude = opev.Magnitude()
    magnitude.mag = userdict['mag']
    magnitude.magnitude_type = userdict['magType']
    # Create/Populate Event class
    outEv = opev.Event()
    outEv.resource_id = userdict['id']
    outEv.origins.append(origin)
    outEv.magnitudes.append(magnitude)
    return outEv


def dict2PickContainer(userdict, eqid, eqtag, picktag):
    """
    Utility function to convert user dict from previus
    parsing into a new class object.

    The USERDICT should have the subsequent keys:

    userdict['STATION']={'PHSNM'=[
                                    {'polarity':      (str/None)
                                    'onset':          (str/None)
                                    'weight':         (float/None)
                                    'pickclass':          (int/None)
                                    'timeUTC_pick':   (UTCDateTime/None)
                                    'timeUTC_early':  (UTCDateTime/None)
                                    'timeUTC_late':   (UTCDateTime/None)
                                    },
                                 ]
                        }
    *** NB: PHASENAME is a list of Dictionary, in
        order to preserve the order and the information
    """
    od = QD.PickContainer(eqid, eqtag, picktag)
    for stat in userdict.keys():
        od.addStat(stat, userdict[stat])
    return od


def Event2dict(event):
    """
    Internal Module to pass from an user created Dict
    to a custom Obspy Event object.

    Args:
        event (obspy.core.Event): obspy event object
    Returns:
        outdict (dict): event-dictionary with the following keys
                        ('id', 'origintime', 'lat', 'lon', 'dep',
                         'rms', 'gap', 'mag', 'magType')
    NB: The depth is returned in km!!!
    """
    outdict = {}
    KM = 0.001  # m
    outdict['id'] = event.resource_id.id
    #
    outdict['origintime'] = event.origins[0].time
    outdict['lon'] = event.origins[0].longitude
    outdict['lat'] = event.origins[0].latitude
    outdict['dep'] = event.origins[0].depth * KM
    outdict['rms'] = event.origins[0].quality.standard_error
    outdict['gap'] = event.origins[0].quality.azimuthal_gap
    #
    outdict['mag'] = event.magnitudes[0].mag
    outdict['magType'] = event.magnitudes[0].magnitude_type
    #
    return outdict


def extract_catalog_event(catalog, eqid=None):
    """ Simply query EQID along the catalog attribute to search """
    if not catalog:
        raise QE.MissingVariable("Missing class catalog! abort")
    if not eqid or not isinstance(eqid, str):
        raise QE.MissingVariable("Missing EQID string!")
    #
    ev = [_ev for _ev in catalog.events
          if _ev.resource_id.id == eqid][0]
    return ev


def obspy_inventory2adapt_inventory(opinv, create_alias=False):
    """  Parsing function OBSPY to ADAPT
    It converts obspy inventory object into a StatContainer object.

    Args:
        opinv (obspy.Inventory):

    Other Parameters:
        create_alias (bool): if true, the returned catalog waveforms is also store as
            QUAKEML format
        **kwargs: _kwargs_ are used to specify the query parameter. For
            more informations check
            http://www.fdsn.org/webservices/fdsnws-station-1.1.pdf

    Returns:
        sd (database.StatDict): ADAPT inventory object
    """
    IDXCOUNT = {}

    def _set_alias(netnm):
        if netnm not in IDXCOUNT.keys():
            IDXCOUNT[netnm] = 1
        else:
            IDXCOUNT[netnm] += 1
        #
        return netnm[0] + '{:03d}'.format(IDXCOUNT[netnm])

    sd = QD.StatContainer(source_id=None, contains="seismometer", tagstr="info-string")
    for net in opinv:
        for ss in net:
            newst = {}
            newst['fullname'] = ss.code
            newst['lon'] = ss.longitude
            newst['lat'] = ss.latitude
            newst['elev_m'] = ss.elevation
            newst['elev_km'] = ss.elevation / 1000.0
            newst['network'] = net.code
            #
            if create_alias:
                newst['alias'] = _set_alias(net.code)
            else:
                newst['alias'] = None
            #
            sd.addStat(newst['network']+"."+newst['fullname'], newst)

    # self.statdict_keys = ('fullname',         # (adapt.picks.polarity.Polarizer/None)
    #                       'alias',            # (str/None)
    #                       'lat',           # (adapt.picks.weight.Weighter/None)
    #                       'lon',        # (float/None)
    #                       'elev_m',        # (int/None)
    #                       'elev_km',        # (str/None)
    #                       'network',         # (bool/None)
    #                       'channels',        # (adapt.picks.evaluation.Gandalf/None)
    #                       'general_infos'
    #                       )

    if create_alias:
        checklst = [stdct['alias'] for stnm, stdct in sd.items()]
        checkset = set(checklst)
        logger.info("CHECKLIST: %d - SET: %d (if number differs not unique the alias)" %
                    (len(checklst),len(checkset)))
        failed_list = [ff for ff, vv in Counter(checklst).items() if vv > 1]
        if len(failed_list) > 0:
            raise QE.CheckError("There are duplicates in ALIAS list: %r" % failed_list)
    #
    return sd


# =========================================================  TIPS
#
# Useful links
#
# - https://wiki.seismo.ethz.ch/doku.php?id=pro:net:flow:pick&s[]=manupick
