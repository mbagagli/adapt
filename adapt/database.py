import os
import logging
import glob
import re
import pickle
import copy
#
import adapt.utils as QU
import adapt.errors as QE
from obspy import read_inventory
from obspy.core.util.attribdict import AttribDict
from collections import OrderedDict

logger = logging.getLogger(__name__)

"""ADAPT main database module

This module contains all the main and necessary container classes
that let this library works.
Briefly it contains:
    - PickContainer class to store all the picks infos
    - StatContainer class to store all the stations information
    - StatContainer_Event class to store the station event metadata

"""

# --------------------------------------------- Main Class


class PickContainer(OrderedDict):
    """Main class to internally store all picks informations

    Note:
        In this class REFERENCE and PREDICTED picks list
        must be only 1 element long !!! (or at least the same repeated)
    """
    def __init__(self, eqid, eqtag, dicttag):
        super().__init__()
        self.eqid = eqid
        self.type = eqtag.lower()
        self.tag = dicttag.upper()
        self.pickdict_keys = ('polarity',         # (adapt.picks.polarity.Polarizer/None)
                              'onset',            # (str/None)
                              'weight',           # (adapt.picks.weight.Weighter/None)
                              'pickerror',        # (float/None)
                              'pickclass',        # (int/None)
                              'pickpolar',        # (str/None)
                              'evaluate',         # (bool/None)
                              'evaluate_obj',     # (adapt.picks.evaluation.Gandalf/None)
                              'features',         # (dict/None)
                              'features_obj',     # (adapt.picks.featuring.Miner/None)
                              'outlier',          # (bool/None)
                              'phaser_obj',       # (adapt.picks.phaser.Spock/None)
                              'boot_obj',         # (adapt.picks.weight.Bootstrap/None)
                              'timeUTC_pick',     # (UTCDateTime/None)
                              'timeUTC_early',    # (UTCDateTime/None)
                              'timeUTC_late',     # (UTCDateTime/None)
                              'backup_picktime',  # (UTCDateTime/None)
                              'general_infos'
                              )

    # The next Method has been added to let load/store with pickle
    # modules. Full answer at:
    # https://stackoverflow.com/questions/45860040/pickling-a-subclass-of-an-ordereddict
    # https://stackoverflow.com/questions/6190331/can-i-do-an-ordered-default-dict-in-python
    def __reduce__(self):
        """
        In order to be able to pickle object we need to override
        the __reduce__ method.

        OrderedDict.__reduce__ returns a 5 tuple
        the FIRST and last can be kept
        the FOURTH is None and needs to stay None
        the SECOND is a tuple containing UserInput argument in __init__

        *** NB the internal attributes are saved anyway, only custom
               ones needs to be stored properly
        """
        state = super().__reduce__()
        myargs = (self.eqid, self.type, self.tag)
        return (state[0], myargs, None, None, state[4])

    def copy(self):
        return copy.deepcopy(self)

    def getStats(self, **kwargs):
        """
        Class method to extract the station names
        keys contained in the object.

        Return a string list
        """
        return self.keys()

    def getStatPick(self, stat, **kwargs):
        """
        This small method return all the phase definition contained
        in the class for a given STAT name

        RETURN: list
        """
        return self[stat].keys()

    def getMatchingPick(self, stat, picktag, indexnum=0):
        """
        This method let query the dict keys with regexp.
        It returns a dictionary.

        This method is called in main, and statistic.py

        REFERENCE:
        - https://stackoverflow.com/questions/21024822/python-accessing-dictionary-with-wildcards
        """
        outlist = []
        regexp = re.compile(picktag)
        try:
            for key in self[stat].keys():
                if regexp.search(key):
                    try:
                        outlist.append((key, self[stat][key][indexnum]))
                    except IndexError:
                        logger.error("Index number out of bounds! MAX: %d" %
                                     (len(self[stat][key])-1))
                        raise QE.CheckError()
        except KeyError:
            logger.error("Missing station key! %s" % stat)
            raise QE.MissingVariable()
        #
        return outlist

    def addStat(self, stat, statdict,
                overwrite_stat=False,
                overwrite_phase=False):
        """
        Class method to add the station names
        keys and relative phase picks information.

        At the moment the class is structured
        that each phase name is a list of dict,
        containing all the possible picks with a
        certain tag.

        STATION={'PHASENAME'=[
                              {'polarity':         (str/None)
                                'onset':           (str/None)
                                'weight':          (float/None)
                                'pickclass':       (int/None)
                                'timeUTC_pick':    (UTCDateTime/None)
                                'timeUTC_early':   (UTCDateTime/None)
                                'timeUTC_late':    (UTCDateTime/None)
                              }, ...
                             ]
                }
        """
        if stat not in self.keys() or overwrite_stat:
            self[stat] = {}
        #
        if not overwrite_phase:
            for phasename in statdict.keys():
                for pp in statdict[phasename]:  # because it's a list
                    self.addPick(stat, phasename, **pp)

        elif overwrite_phase:
            for phasename in statdict.keys():
                if phasename in self[stat].keys():
                    del self[stat][phasename]
                for pp in statdict[phasename]:  # because it's a list
                    self.addPick(stat, phasename, **pp)
        else:
            logger.error("Something wrong with the overwrite_phase param.!")
            return False

        return True

    def addPick(self, stat, phsnm, **pckdct):
        """
        Method to automatically add a phasename
        or pick information

        :pckdct:

        RETURN (bool)
        """
        try:
            refpckdct = self.checkpickdict(pckdct)
        except TypeError:
            return False
        #
        if stat not in self.keys():  # station doesn't exist
            self[stat] = {}
            self[stat][phsnm] = [refpckdct]
        else:  # station exist
            if phsnm not in self[stat].keys():
                self[stat][phsnm] = [refpckdct]
            else:
                self[stat][phsnm].append(refpckdct)
        return True

    def checkpickdict(self, pickdict):
        """
        This `private` method double-checks that all keys defined
        are used and default to `None` eventually.

        *** NB: Maybe add a double check also for the type requested
        """
        if not isinstance(pickdict, dict):
            logger.write("Expected a dict! %s given instead" % type(pickdict))
            raise TypeError
        #
        outdict = {}
        in_keys = pickdict.keys()
        for kk in self.pickdict_keys:
            if kk not in in_keys:
                outdict[kk] = None
            else:
                outdict[kk] = pickdict[kk]
        return outdict

    def store2disk(self, pathfn):
        """
        This method should be able to store the object
        with pickle library.

        *** NB! it would override previously saved obj.
        """
        with open(pathfn, 'wb') as OUT:
            pickle.dump(self, OUT, pickle.HIGHEST_PROTOCOL)
        return True

    def merge_container(self,
                        pick_container,
                        check_eqid=True,
                        overwrite_stat=False,
                        overwrite_phase=False):
        """Merge the picks from two different PickContainer

            This method merge two instances of PickContainer.

            Args:
                pick_container (str, PickContainer): file path or
                    object to append/merge
                check_id (bool): compare the eqid to before merging

        """
        if isinstance(pick_container, str):
            pckcnt_add = QU.loadPickleObj(pick_container)
        else:
            pckcnt_add = pick_container
        #
        if check_eqid:
            if pckcnt_add.eqid.lower() != self.eqid.lower():
                logger.error("EQIDs don't match, abort merging!")
                return False
        #     def addStat(self, stat, statdict, overwrite_stat=False):
        for ss, ssdct in pckcnt_add.items():
            self.addStat(ss, ssdct,
                         overwrite_stat=overwrite_stat,
                         overwrite_phase=overwrite_phase)
        #
        return True

    def quick_stats(self):
        """Simply count occurence of phase-names and classes

        It should be handy for VELEST CNV objects

        """
        def __init_phase_stat():
            _dd = {}
            _dd['counts'] = 0
            for _nc in ('0', '1', '2', '3', '4', '5'):
                _dd[_nc] = 0
            return _dd

        # ---
        stats = {}
        #
        for ss, phdct in self.items():
            for phn, phnlist in self[ss].items():
                if phn not in stats.keys():
                    stats[phn] = __init_phase_stat()
                for xx in phnlist:
                    # Single-Obs finally
                    stats[phn]['counts'] += 1
                    stats[phn][str(xx['pickclass'])] += 1
        return stats

    def delete_pick(self, stat, phname, phidx):
        """ Remove precise obs """
        try:
            del self[stat][phname][phidx]
        except (KeyError, IndexError):
            logger.warning("No match found for: %s - %s - %d" %
                           (stat, phname, phidx))

    def delete_empty_station(self):
        """ Remove empty station entry """

        # --- Remove Empty phase if found
        remlst = []
        for _ss, _ppd in self.items():
            for _ppnm, _pplst in _ppd.items():
                if len(_pplst) == 0:
                    remlst.append((_ss, _ppnm))

        for _kill in remlst:
            del self[_kill[0]][_kill[1]]

        # --- Remove Empty station if left
        remlst = []
        for _ss, _ppd in self.items():
            if not _ppd:
                remlst.append(_ss)
        for _kill in remlst:
            del self[_kill]

    def sort_by_epidist(self, evla, evlo, statDict):
        """ Still to be implemented """
        pass

    @classmethod
    def from_dictionary(cls, dictobj):
        """Factory methods for creating new instance of a PickContainer.

        The dictionary (DD) can be created as follow:
            - DD['eqid'] => new instance's eqid
            - DD['eqtag'] => new instance's eqtag
            - DD['dicttag'] => new instance's dicttag
        Folowed up by all the stations-values pair.
        """
        if not isinstance(dictobj, dict):
            raise QE.InvalidType("I need to load a dictionary. Got a %r "
                                 "instead!" % type(dictobj))
        #
        logger.info("Creating PickContainer from dictionary")
        listofkeys = [kk.lower() for kk in dictobj.keys()]
        if "eqid" in listofkeys:
            eqid = dictobj.pop('eqid')
        else:
            eqid = "unknown"
        #
        if "eqtag" in listofkeys:
            eqtag = dictobj.pop('eqtag')
        else:
            eqtag = "unknown"
        #
        if "dicttag" in listofkeys:
            dicttag = dictobj.pop('dicttag')
        else:
            dicttag = "unknown"

        # Initialize
        _tmp = cls(eqid, eqtag, dicttag)
        workdict = copy.deepcopy(dictobj)
        for ss in workdict.keys():
            _tmp.addStat(ss, workdict[ss])
        return _tmp

    @classmethod
    def create_empty_container(cls,
                               evid="TESTEVENT",
                               evtag="nothing",
                               dicttag="EmptyBox"):
        """Handy function for creating empy pickdicts.

        It will come useful to create 'empty boxes' during tests

        """
        return cls(evid, evtag, dicttag)


class StatContainer(OrderedDict):
    """
    This module serves for contain only the
    necessary station metadata information for
    the adapt framework.

    This class serves as a container for an INVENTORY
    fast accessible and array related.
    """

    def __init__(self, source_id=None, contains="seismometer", tagstr="info-string"):
        super().__init__()
        self.source_id = source_id
        self.contains = contains.lower()
        self.tagstr = tagstr.upper()
        self.statdict_keys = ('fullname',        # (str)
                              'alias',           # (str)
                              'lat',             # (float)
                              'lon',             # (float)
                              'elev_m',          # (float)
                              'elev_km',         # (float)
                              'network',         # (str)
                              'general_infos'
                              )

    # The next Method has been added to let load/store with pickle
    # modules. Full answer at:
    # https://stackoverflow.com/questions/45860040/pickling-a-subclass-of-an-ordereddict
    # https://stackoverflow.com/questions/6190331/can-i-do-an-ordered-default-dict-in-python
    def __reduce__(self):
        """
        In order to be able to pickle object we need to override
        the __reduce__ method.

        OrderedDict.__reduce__ returns a 5 tuple
        the FIRST and last can be kept
        the FOURTH is None and needs to stay None
        the SECOND is a tuple containing UserInput argument in __init__

        *** NB the internal attributes are saved anyway, only custom
               ones needs to be stored properly
        """
        state = super().__reduce__()
        myargs = ()  # here add the list of user-input
        return (state[0], myargs, None, None, state[4])

    def _get_all_alias(self):
        """ DEDE """
        alarr = []
        for _k, _d in self.items():
            try:
                alarr.append(_d['alias'])
            except KeyError:
                alarr.append(None)
        return alarr

    def get_names(self):
        """Return object keys

        Class method to extract all the keys (station names) contained
        in the object.

        Returns:
            dict_keys (list): list of stations

        """
        return self.keys()

    def get_statname_from_alias(self, instr):
        """ Return FULLNAME from ALIAS """
        aa = [(ss, dd) for ss, dd in self.items() if dd['alias'] == str(instr)]
        if len(aa) > 1:
            raise QE.CheckError(
                "ERROR: [ %d ] stations have the same ALIAS !!!"
                " %s --> %s" % (len(aa), instr, [i[0] for i in aa]))
        #
        try:
            _ = aa[0][0]
        except IndexError:
            raise QE.CheckError("Missing Station ALIAS:  %s" % instr)
        #
        try:
            fn = aa[0][1]['fullname']
        except KeyError:
            raise QE.CheckError("Station  %s  has no fullname key !!!" % instr)
        #
        return fn

    def get_alias(self, instr):
        """ Return the ALIAS corresponding to input STATNAME """

        aa = [(ss, dd) for ss, dd in self.items()
              if dd['fullname'] == str(instr)]
        if len(aa) > 1:
            raise QE.CheckError(
                "ERROR: [ %d ] stations have the same FULLNAME !!!"
                " %s --> %s" % (len(aa), instr, [i[0] for i in aa]))
        #
        try:
            _ = aa[0][0]
        except IndexError:
            raise QE.CheckError("Missing Station FULLNAME:  %s" % instr)
        #
        try:
            al = aa[0][1]['alias']
        except KeyError:
            raise QE.CheckError("Station  %s  has no alias key !!!" % instr)
        #
        return al

    def set_alias(self, statkey, aliasname, force=False):
        """ Add Alias to station key """
        if len(aliasname) > 4:
            raise QE.CheckError("Alias must be MAX 4 chars:  %s has %d" %
                                (aliasname, len(aliasname)))
        #
        if aliasname in self._get_all_alias() and not force:
            _stnm = self.get_statname_from_alias(aliasname)
            raise QE.CheckError("Alias already present! [ %s --> %s ]" %
                                (aliasname, _stnm))
        elif aliasname in self._get_all_alias() and force:
            _stnm = self.get_statname_from_alias(aliasname)
            logger.warning("Alias already present [ %s --> %s ] !!! "
                           "But forced for station [%s]" % (
                                                aliasname, _stnm, statkey))
        try:
            self[statkey]['alias'] = aliasname
        except KeyError:
            raise QE.CheckError("Missing Station FULLNAME:  %s" % statkey)
        #
        return True

    def addStat(self, stat, statdict, override=False):
        """
        Class method to add the station names
        keys and relative phase picks information.

        At the moment the class is structured
        that to each station name corrispond name is a list of dict,
        containing all the possible picks with a
        certain tag.

        STATIONNAME={'fullname' (str),
                     'alias' (str),    MAX 4 CHAR !!!
                     'network' (str),
                     'lat' (float),
                     'lon' (float),
                     'elev_m' (float)
                }
        """
        if stat not in self.keys():
            self[stat] = {}
            self[stat] = statdict
            return True
        else:
            if override:
                # Station with same name already in ...
                logger.warning("Station %s already loaded, OVERRIDE" % stat)
                self[stat] = statdict
            else:
                # Station with same name already in ...
                logger.warning("Station %s already loaded, SKIPPING" % stat)
            return False

    def getStat(self, stat_name, is_alias=False):
        """Return object keys

        Class method to extract all the keys (station names) contained
        in the object.

        Returns:
            dict_keys (list): list of stations

        """
        if is_alias:
            return [ss for ss in self if ss['alias'] == stat_name]
        else:
            return self[stat_name]

    def store2disk(self, pathfn):
        """
        This method should be able to store the object
        with pickle library.

        *** NB! it would override previously saved obj.
        """
        with open(pathfn, 'wb') as OUT:
            pickle.dump(self, OUT, pickle.HIGHEST_PROTOCOL)
        return True

    @classmethod
    def from_dictionary(cls, dictobj):
        """Factory methods for creating new instance of a StatContainer.

        The dictionary (DD) can be created as follow:
            - DD['source_id'] => new instance's id
            - DD['contains'] => new instance's contains
            - DD['tagstr'] => new instance's tagstr
        Folowed up by all the stations-values pair.
        """
        if not isinstance(dictobj, dict):
            raise QE.InvalidType("I need to load a dictionary. Got a %r "
                                 "instead!" % type(dictobj))
        #
        logger.info("Creating StatContainer from dictionary")
        listofkeys = [kk.lower() for kk in dictobj.keys()]
        if "source_id" in listofkeys:
            source_id = dictobj.pop('source_id')
        else:
            source_id = "unknown"
        #
        if "contains" in listofkeys:
            contains = dictobj.pop('contains')
        else:
            contains = "unknown"
        #
        if "tagstr" in listofkeys:
            tagstr = dictobj.pop('tagstr')
        else:
            tagstr = "unknown"

        # Initialize
        _tmp = cls(source_id, contains, tagstr)
        workdict = copy.deepcopy(dictobj)
        for ss in workdict.keys():
            _tmp.addStat(ss, workdict[ss])
        return _tmp


class StatContainer_Event(OrderedDict):
    """
    This class is made as a container for station metadata
    referred to a specific event.
    """

    def __init__(self):
        super().__init__()
        self.statdict_key = ('isdownloaded',      # (bool)
                             'isautomatic',       # (bool)
                             'isreference',       # (bool)
                             'epidist',           # (float/None)
                             'missingchannel',    # (tuple)
                             'ispickable',        # (bool)
                             'isselected',        # (bool) -> if it falls inside the RADIUS
                             'automatic_picks',   # list of tuple/None
                             'reference_picks',   # list of tuple/None
                             'predicted_picks',   # list of tuple/None
                             'bait_picks',        # list of tuple/None
                             'sampling_rate',     # (float/None)
                             'p_delay',            # (float/None)
                             's_delay')            # (float/None)

    # BAIT_PICKS = [(time, bk_info), ]
    # PREDICTED_PICKS = [(info, time), ]

    # The next Method has been added to let load/store with pickle
    # modules. Full answer at:
    # https://stackoverflow.com/questions/45860040/pickling-a-subclass-of-an-ordereddict
    # https://stackoverflow.com/questions/6190331/can-i-do-an-ordered-default-dict-in-python
    def __reduce__(self):
        """
        In order to be able to pickle object we need to override
        the __reduce__ method.

        OrderedDict.__reduce__ returns a 5 tuple
        the FIRST and last can be kept
        the FOURTH is None and needs to stay None
        the SECOND is a tuple containing UserInput argument in __init__

        *** NB the internal attributes are saved anyway, only custom
               ones needs to be stored properly
        """
        state = super().__reduce__()
        myargs = ()  # here add the list of user-input data
        return (state[0], myargs, None, None, state[4])

    def _newStat(self, eqtag, stat):
        """
        Simple private method to initialize new stations in memory
        """
        self[eqtag][stat] = {}
        for kk in self.statdict_key:
            self[eqtag][stat][kk] = None

    def _checkstatdict(self, eqtag, stat, **indict):
        """
        This `private` method double-checks that all keys defined
        are used and default to `None` eventually and left untouched
        if already present. In any case, the keys in input will be
        overwrited

        *** NB: Maybe add a double check also for the type requested
        """
        if not isinstance(indict, dict):
            logger.write("Expected a dict! %s given instead" % type(indict))
            raise TypeError
        #
        for kk in indict:
            if kk not in self.statdict_key:
                raise QE.InvalidVariable("Invalid key: %s" % kk)
            self[eqtag][stat][kk] = indict[kk]

    def getStats4Event(self, eqtag, **kwargs):
        """
        Class method to extract the station names
        keys contained in the object.

        Return a string list
        """
        return self[eqtag].keys()

    def addEvent(self, eqtag):
        """
        This method just add a key to the object
        """
        self[eqtag] = {}

    def addStat(self, eqtag, stat, **statdict):
        """
        Class method to add the station names
        keys and relative phase picks information.

        The private method _checkstatdict will take care of adding
        default values for what is missin by user.
        """
        if eqtag not in self.keys():
            self.addEvent(eqtag)
        if stat not in self[eqtag].keys():
            self._newStat(eqtag, stat)
        #
        if statdict:
            self._checkstatdict(eqtag, stat, **statdict)
        return True

    def store2disk(self, pathfn):
        """
        This method should be able to store the object
        with pickle library.

        *** NB! it would override previously saved obj.
        """
        with open(pathfn, 'wb') as OUT:
            pickle.dump(self, OUT, pickle.HIGHEST_PROTOCOL)
        return True

    def change_metadata(self, eqtag, statlist="all", **kwargs):
        """
        This method will allow to change the metadata internally
        the class. PLEASE USE THIS METHOD instead of changing manually!
        """
        print("To be implemented, sorry")
        return False

    @classmethod
    def from_dictionary(cls, dictobj):
        """Factory methods for creating new instance of a
           StatContainer_Event class.
        """
        if not isinstance(dictobj, dict):
            raise QE.InvalidType("I need to load a dictionary. Got a %r "
                                 "instead!" % type(dictobj))
        #
        logger.info("Creating StatContainer_Event from dictionary")
        workdict = copy.deepcopy(dictobj)

        _tmp = cls()
        for evid in workdict.keys():
            _tmp[evid] = workdict[evid]
        return _tmp


# --------------------------------------------- Array Related


def createStationInventory(**kwargs):
    """
    This function is collecting all the *.xml
    files stored in adir and merge them into an
    ObsPy Inventory object.

    *** NB: The directory should contain only one
            single *.xml file for each station
    """

    inventorypath = (kwargs["GENERAL"]["workingrootdir"] +
                     os.sep + kwargs["ARRAY"]["inventoryxmlfile"])

    if kwargs["ARRAY"]["loadinventory"]:
        if os.path.exists(inventorypath) and os.path.getsize(
                inventorypath) > 0:
            logger.info("Loading Station Inventory found: %r" % inventorypath)
            inv = read_inventory(inventorypath, format='STATIONXML')
            logger.info(".... done!")
            return inv
        else:
            inv = buildInventory(**kwargs)
        return inv

    elif (not kwargs["ARRAY"]["loadinventory"] and
            kwargs["ARRAY"]["createinventory"]):
        inv = buildInventory(**kwargs)
        return inv
    else:
        return False


def buildInventory(**kwargs):
    """
    Central block that takes care of creating an
    ObsPy inventory object.

    This function will take care also to export
    it to hard-disk (if requested by user).

    *** NB: it will override without asking
    """

    inventorypath = kwargs["GENERAL"]["workingrootdir"] + \
        os.sep + kwargs["ARRAY"]["inventoryxmlfile"]

    logger.info("Building Station Inventory. This may take a while ...")
    statFileList = []
    for ext in (".xml", "*.xml"):
        statFileList.extend(
            glob.glob(
                kwargs["ARRAY"]["stationxmlfilepath"] +
                os.sep +
                ext))
    if not statFileList:
        logger.error(
            "No xml found in %r" %
            kwargs["ARRAY"]["stationxmlfilepath"])
        return None
    #
    networklist = []
    statFileList.sort()
    for xx, stat in enumerate(statFileList):
        if xx == 0:
            inv = read_inventory(stat, "STATIONXML")
            networklist.append(inv[0].code)
        else:
            tmpinv = read_inventory(stat, "STATIONXML")
            tmpnet = tmpinv[0].code
            # Check if Network already loaded
            if tmpnet in networklist:
                # find net index in the already loaded inventory
                tmpidx = next(_ii for _ii,
                              net in enumerate(networklist) if tmpnet == net)
                inv[tmpidx].stations.append(
                    tmpinv[0].stations[0])  # Only Append Station
            else:
                networklist.append(tmpinv[0].code)
                inv += tmpinv
            #
        QU.progressBar(
            xx + 1,
            len(statFileList),
            prefix='Import',
            decimals=1,
            barLength=15)
    # Exporting
    if kwargs["ARRAY"]["exportinventory"]:
        logger.info("Exporting the Station Inventory to %r" % inventorypath)
        inv.write(inventorypath, format='STATIONXML')
        logger.info(".... done!")
    return inv


def createStationContainer(**kwargs):
    """
    This function is collecting all the *.xml
    files stored in adir and merge them into an
    ADAPT StationContainer object.

    *** NB: The directory should contain only one
            single *.xml file for each station
    *** NB: it will always store on disk  the StationContainer


    """
    containerpath = kwargs["GENERAL"]["workingrootdir"] + \
        os.sep + kwargs["ARRAY"]["stationpicklepath"]
    qsc = StatContainer()
    if os.path.exists(containerpath) and os.path.getsize(containerpath) > 0:
        logger.info("Loading station container pickle: %s" % containerpath)
        qsc = QU.loadPickleObj(containerpath)
        logger.info(".... done!")
    else:  # create one and store it!
        logger.info("Creating Station Container. This may take a while ...")
        statFileList = []
        for ext in (".xml", "*.xml"):
            statFileList.extend(
                glob.glob(
                    kwargs["ARRAY"]["stationxmlfilepath"] +
                    os.sep +
                    ext))

        if not statFileList:
            logger.error(
                "No xml found in %r" %
                kwargs["ARRAY"]["stationxmlfilepath"])
            raise QE.MissingVariable()

        else:
            statFileList.sort()
            for xx, stat in enumerate(statFileList):
                tmpdict = {}
                try:
                    tmpinv = read_inventory(stat, kwargs["ARRAY"]['xmlfiletype'])
                except TypeError:
                    continue
                #
                statName = tmpinv[0].stations[0].code
                tmpdict["fullname"] = statName
                tmpdict["alias"] = None
                tmpdict["lat"] = tmpinv[0].stations[0].latitude
                tmpdict["lon"] = tmpinv[0].stations[0].longitude
                tmpdict["elev_m"] = tmpinv[0].stations[0].elevation
                tmpdict["network"] = tmpinv[0].code
                #
                qsc.addStat(statName, tmpdict)
                QU.progressBar(
                    xx + 1,
                    len(statFileList),
                    prefix='Import',
                    decimals=1,
                    barLength=15)
            qsc.store2disk(containerpath)
    return qsc


# --------------------------------------------- Miscellaneous


def checkChannels(instream, eqid, metastatdict, channels=("Z", "N", "E")):
    """
    Simple function that return a tuple of missing channel, or None.

    # v0.5.1 Now the function will append missing channels EVEN if DATA
             is missing from that channel.

    INPUT:
        obspy.Stream class
    OUTPUT:
        tuple class
    """

    # === v0.5.1
    statname = instream[0].stats.station
    inchann = [(tr.stats.channel[-1], tr.data.size) for tr in instream]
    misschann = []
    for ii in channels:
        chan_found = False
        for _cc, _ss in inchann:
            if _cc == ii:
                # chan match ==> check data
                chan_found = True
                if _ss != 0:
                    # all good, no missing ...
                    continue
                else:
                    # data missing ==> still count
                    misschann.append(ii)
                    logger.warning("Event: %s Stat: %s - MissChann: %s" %
                                   (eqid, statname, ii))
        #
        if not chan_found:
            # channel missing ==> still count
            misschann.append(ii)
            logger.warning("Event: %s Stat: %s - MissChann: %s" %
                           (eqid, statname, ii))
    #
    metastatdict.addStat(eqid, statname, missingchannel=tuple(misschann))
    return True
