import logging
import copy
#
from obspy.core.event import Catalog
from adapt.database import PickContainer
from adapt.database import StatContainer_Event
import adapt.picks.phaser_functions as FUNCTIONS
import adapt.errors as QE


logger = logging.getLogger(__name__)


class Spock(object):
    """ This class takes care of the phase identification problem,
        and outliers detection.
        Should be called at the end of an event, even though one
        instance could be initialized after a multipicking stage for
        outliers detection.

        NB: prob puoi rimuovere il catalogo

        NB: the shoot_outliers functions will return True if the
            OUTLIER CONDITION is found.

        NB: the phase key list is an helper to go straight at the
            needed phasetags. No need to specify those in the function.
            The function will receive all necessary from the `shoot*`
            methods.

        NB: `shoot_outliers` method should be called always before
            `shoot_phases` one.

    """

    def __init__(self,
                 proc_stream,
                 raw_stream,
                 final_pickdict,
                 station_metadata,
                 event_catalog,
                 channel="*Z",
                 stat_key_list=(),
                 eventid_key_list=(),
                 phases_key_list=(),
                 functions_dict_outliers={},
                 functions_dict_associator={}):
        if not isinstance(final_pickdict, PickContainer):
            logger.error("`final_pickdict` must be a PickContainer object!")
            raise QE.InvalidType()
        if not isinstance(station_metadata, StatContainer_Event):
            logger.error("`station_metadata` must be a StatContainer_Event " +
                         "object!")
            raise QE.InvalidType()
        if not isinstance(event_catalog, Catalog):
            logger.error("`event_catalog` must be a Catalog object!")
            raise QE.InvalidType()
        if not isinstance(stat_key_list, (list, tuple)):
            logger.error("`stat_key_list` must be a list or tuple object!")
            raise QE.InvalidType()
        if not isinstance(eventid_key_list, (list, tuple)):
            logger.error("`eventid_key_list` must be a list or tuple object!")
            raise QE.InvalidType()
        if not isinstance(phases_key_list, (list, tuple)):
            logger.error("`phases_key_list` must be a list or tuple object!")
            raise QE.InvalidType()
        if (not isinstance(functions_dict_outliers, dict) or
           not isinstance(functions_dict_associator, dict)):
            logger.error("`functions_dict` for outliers and associator " +
                         "must be a dict object!")
            raise QE.InvalidType()
        # Main
        self.pst = proc_stream.copy()
        self.ptr = self.pst.select(channel=channel)[0]
        self.rst = raw_stream.copy()
        self.rtr = self.rst.select(channel=channel)[0]
        self.picks_dict = copy.deepcopy(final_pickdict)
        self.stats_dict = station_metadata
        self.event_cata = event_catalog
        # Optional
        self.stats_keys = stat_key_list
        self.event_keys = eventid_key_list
        self.phase_keys = phases_key_list
        # Output
        self.funct_dict_outliers = functions_dict_outliers
        self.funct_dict_associator = functions_dict_associator
        self.result_dict_outliers = {}
        self.result_dict_associator = {}

    def _clear_out(self):
        self.result_dict_outliers = {}
        self.result_dict_associator = {}

    def _extract_event(self, idstr, deep_copy=False):
        if not isinstance(idstr, str):
            raise QE.InvalidType()
        # Search
        fe = None
        for _ee in self.event_cata:
            if idstr == _ee.resource_id.id:
                fe = _ee
                break
        # Out
        if fe:
            if deep_copy:
                return copy.deepcopy(fe)
            else:
                return fe

    def shoot_outliers(self):
        """ Running the given funtions in the dict """
        if not self.funct_dict_outliers:
            raise QE.MissingAttribute({"message":
                                       "outliers functions dict missing!"})
        if not self.event_keys:
            raise QE.MissingAttribute({"message":
                                       "event_keys list missing!"})
        if not self.stats_keys:
            raise QE.MissingAttribute({"message":
                                       "stats_keys list missing!"})

        # MB: sorting in alphabetical order the testfunctions
        sortedkeys = sorted(self.funct_dict_outliers, key=str.lower)

        # Looping through events ...
        for _ee in self.event_keys:
            if _ee not in self.result_dict_outliers.keys():
                self.result_dict_outliers[_ee] = {}
            # _ee_obj = self._extract_event(_ee, deep_copy=False)

            # ... stations ...
            for _ss in self.stats_keys:
                if _ss not in self.result_dict_outliers[_ee].keys():
                    self.result_dict_outliers[_ee][_ss] = {}

                # ... phases ...
                for _pp in self.phase_keys:
                    if _pp not in self.result_dict_outliers[_ee][_ss].keys():
                        self.result_dict_outliers[_ee][_ss][_pp] = {}
                    #
                    try:
                        _pick_time = self.picks_dict[_ss][_pp][0]['timeUTC_pick']
                    except KeyError:
                        # Missing ORIGINAL PHASE TAG, continue
                        continue

                    if not _pick_time:
                        # ADAPT missing pick --> Go to another phases
                        continue

                    # ... and functions!
                    for xx in sortedkeys:
                        logger.debug("%s - %r" % (
                                     xx, self.funct_dict_outliers[xx]))
                        _funct = getattr(FUNCTIONS, xx)
                        od = _funct(self.ptr, self.rtr, _pick_time,
                                    self.stats_dict[_ee][_ss],
                                    **self.funct_dict_outliers[xx])
                        #
                        self.result_dict_outliers[_ee][_ss][_pp][xx] = od

        # Print debug log
        logger.debug("DONE: %d functions over %d stations in %d events!" % (
                     len(self.funct_dict_outliers),
                     len(self.stats_keys),
                     len(self.event_keys)))

    def modify_outliers(self):
        """ This method should map the evaluation steps (outliers,
            phase association) in the input dict, modifying it
        """
        if not self.result_dict_outliers:
            logger.error("Missing RESULTS DICT, run `shoot_outliers` method first")
            raise QE.MissingAttribute
        # event ...
        for _ee in self.event_keys:
            # ... stations ...
            for _ss in self.stats_keys:
                # ... phases ...
                for _pp in self.phase_keys:
                    if not self.result_dict_outliers[_ee][_ss][_pp]:
                        # ADAPT missing pick --> Go to another phases
                        continue

                    # ... functions !
                    _results = []
                    for xx in self.funct_dict_outliers.keys():
                        _results.append(self.result_dict_outliers
                                        [_ee][_ss][_pp][xx]['result'])

                    # Define outlier:
                    # Will add a trailing '*' to the PHASE name
                    if _results and all(_results):
                        # Outlier detected if ALL tests returned TRUE
                        _pp_new = _pp.replace("~", "").replace("*", "")
                        self.picks_dict[_ss]["*"+_pp_new] = (
                                                 self.picks_dict[_ss].pop(_pp))
                        # Because a new phase has been introduced,
                        # we add that to the list for later use (PHASE)
                        _tmp = list(self.phase_keys)
                        _tmp.append("*"+_pp_new)
                        self.phase_keys = tuple(_tmp)

    def shoot_phases(self):
        """ Running the given funtions in the dict to define """
        pass
        if not self.funct_dict_associator:
            raise QE.MissingAttribute({"message":
                                       "associator functions dict missing!"})
        if not self.event_keys:
            raise QE.MissingAttribute({"message":
                                       "event_keys list missing!"})
        if not self.stats_keys:
            raise QE.MissingAttribute({"message":
                                       "stats_keys list missing!"})
        if not self.phase_keys:
            raise QE.MissingAttribute({"message":
                                       "phases_key list missing!"})

        # MB: sorting in alphabetical order the testfunctions
        sortedkeys = sorted(self.funct_dict_associator, key=str.lower)

        # Looping through events ...
        for _ee in self.event_keys:
            if _ee not in self.result_dict_associator.keys():
                self.result_dict_associator[_ee] = {}

            # ... stations ...
            for _ss in self.stats_keys:
                if _ss not in self.result_dict_associator[_ee].keys():
                    self.result_dict_associator[_ee][_ss] = {}

                # ... phases ...
                for _pp in self.phase_keys:
                    if _pp not in self.result_dict_associator[_ee][_ss].keys():
                        self.result_dict_associator[_ee][_ss][_pp] = {}
                    #

                    try:
                        _pick_time = self.picks_dict[_ss][_pp][0]['timeUTC_pick']
                    except KeyError:
                        # Missing ORIGINAL PHASE TAG, continue
                        continue

                    if not _pick_time:
                        # ADAPT missing pick --> Go to another phases
                        continue

                    # ... and functions!
                    for xx in sortedkeys:
                        logger.debug("%s - %r" % (
                                     xx, self.funct_dict_associator[xx]))
                        _funct = getattr(FUNCTIONS, xx)
                        od = _funct(self.ptr, self.rtr, _pick_time,
                                    self.stats_dict[_ee][_ss],
                                    **self.funct_dict_associator[xx])
                        #
                        self.result_dict_associator[_ee][_ss][_pp][xx] = od

        # Print debug log
        logger.debug("DONE: %d functions over %d stations in %d events!" % (
                     len(self.funct_dict_associator),
                     len(self.stats_keys),
                     len(self.event_keys)))

    def modify_phases(self):
        """ This method should map the evaluation steps (outliers,
            phase association) in the input dict, modifying it
        """
        if not self.result_dict_associator:
            logger.error(
                "Missing RESULTS DICT, run `shoot_phases` method first")
            raise QE.MissingAttribute
        # event ...
        for _ee in self.event_keys:
            # ... stations ...
            for _ss in self.stats_keys:
                # ... phases ...
                for _pp in self.phase_keys:
                    if not self.result_dict_associator[_ee][_ss][_pp]:
                        # ADAPT missing pick --> Go to another phases
                        continue

                    # ... functions !
                    _results = []
                    for xx in self.funct_dict_associator.keys():
                        _results.append(self.result_dict_associator
                                        [_ee][_ss][_pp][xx]['result'])
                    # Define phase
                    if _results and all(_results):
                        # Misphase detected if ALL tests returned TRUE
                        _pp_new = _pp.replace("~", "").replace("*", "")
                        self.picks_dict[_ss]["~"+_pp_new] = (
                                                 self.picks_dict[_ss].pop(_pp))
                        # Because a new phase has been introduced,
                        # we add that to the list for later use (OUTLIER)
                        _tmp = list(self.phase_keys)
                        _tmp.append("~"+_pp_new)
                        self.phase_keys = tuple(_tmp)

    def set_statkey_list(self, inlst):
        if not isinstance(inlst, (list, tuple)):
            logger.error("`stat_key_list` must be a list or tuple object!")
            raise QE.InvalidType()
        #
        self.stats_keys = inlst
        self._clear_out()

    def set_eventid_list(self, inlst):
        if not isinstance(inlst, (list, tuple)):
            logger.error("`eventid_key_list` must be a list or tuple object!")
            raise QE.InvalidType()
        #
        self.event_keys = inlst
        self._clear_out()

    def set_functions_dict_outliers(self, indct):
        if not isinstance(indct, dict):
            logger.error("`functions_dict_outliers` must be a dict object!")
            raise QE.InvalidType()
        #
        self.functions_dict_outliers = indct
        self._clear_out()

    def set_functions_dict_associator(self, indct):
        if not isinstance(indct, dict):
            logger.error("`funct_dict_associator` must be a dict object!")
            raise QE.InvalidType()
        #
        self.funct_dict_associator = indct
        self._clear_out()

    def get_result_dict_outliers(self):
        if not self.result_dict_outliers:
            logger.error("Missing RESULTS DICT, run `shoot_outliers` method first")
            raise QE.MissingAttribute
        #
        return self.result_dict_outliers

    def get_result_dict_associator(self):
        if not self.result_dict_associator:
            logger.error("Missing RESULTS DICT, run `shoot_phases` method first")
            raise QE.MissingAttribute
        #
        return self.result_dict_associator

    def get_picks_dict(self):
        """ Return the INPUT DICT, as it is modifyed directly """
        return self.picks_dict
