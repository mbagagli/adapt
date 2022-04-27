import logging
import numpy as np
from obspy import Stream
from obspy import UTCDateTime
from scipy import signal
#
from adapt.picks.weight import Weighter
from adapt.picks.polarity import Polarizer
from adapt.picks.evaluation import Gandalf
from adapt.scaffold.featuring import Miner
from adapt import processing as QPR
#
from adapt.database import PickContainer
import adapt.errors as QE

logger = logging.getLogger(__name__)

""" QUAKE pick analysis module.

In this module are stored all the classes and functions to analyse and
define a global pick from a `adapt.database.PickContainer` object.

"""

# --------------------------------------------- Classes DEF


class Judge(object):
    """ This class orchestrate the final decision of a pick definition
        over a set of observation.

        WORK PD is given by Multipicker Layer object (hive)
        FINAL PD is created HERE!

        Now the storage of upper and lower bounds is given by earliest
        and latest arrival of validobs

    """
    def __init__(self,
                 pick_dict,  # Retreived with get_workPickDict from HIVE
                 stat_list,  # List or Tuple containing the station keys
                 proc_stream,
                 raw_stream,
                 channel="*Z",
                 extract_picktime="median",
                 extract_pickerror="std",
                 triage_method="ekmb",  # jackknife
                 weigther_dict={},
                 polarizer_dict={},
                 evaluator_dict={},
                 features_dict={},    # Dictionary for Miner class
                 storing_pick_tag="JusticeForAll"):

        # Checks
        if not isinstance(pick_dict, PickContainer):
            logger.error("Invalid pick_dict object")
            raise QE.InvalidType()
        if (not isinstance(proc_stream, Stream) or
           not isinstance(raw_stream, Stream)):
            logger.error("Invalid stream object")
            raise QE.InvalidType()
        if not isinstance(stat_list, (list, tuple)):
            logger.error("Invalid pick_dict object")
            raise QE.InvalidType()
        if len(pick_dict.getStats()) < 1:
            raise QE.MissingPicks()

        # ------------------------------------ Attributes
        # General
        self.pick_dict = pick_dict
        self.statlst = stat_list  # MB: defined by `set_weighter_dict` method
        self.pst = proc_stream.copy()
        self.rst = raw_stream.copy()
        self.chan = channel
        self.wst = None  # MB: `defined in set_working_stream` method
        # Pick Analisy
        self.triage_method = triage_method
        self.eval_dict = evaluator_dict
        self.polar_dict = polarizer_dict
        self.weight_dict = weigther_dict
        self.feat_dict = features_dict
        # Finalizing
        self.extract_pt = extract_picktime
        self.extract_pe = extract_pickerror
        self.final_pick_tag = storing_pick_tag
        self.final_pd = PickContainer(self.pick_dict.eqid,
                                      self.pick_dict.type,
                                      "_".join(["JudgePickDict",
                                                self.final_pick_tag]))

    # ========================== Private

    def _reset_final_pd(self):
        self.final_pd = PickContainer(self.pick_dict.eqid,
                                      self.pick_dict.type,
                                      "JudgePickDict")
        logger.warning("The ouput PickContainer has been reset! " +
                       "Please run again the `self.deliberate` method " +
                       "to define the new pick's attributes")

    # ========================== Change Objects

    def set_extract_picktime(self, meth):
        """ Change the extract_picktime method.

            NB: The `final_pd` attribute will be resetted! Remember
                to run again the `deliberate` method to define the new
                pick's attributes.
        """
        if (not isinstance(meth, str) or
           meth.lower() not in ('mean', 'median')):
            logger.error("Invalid extraction method! ['mean', 'median']")
            raise QE.InvalidType
        #
        self.extract_pt = meth
        self._reset_final_pd()
        logger.warning("The PICKTIME METHOD changed and FINAL_PD resetted!" +
                       "Remember to manually call the deliberate method!")

    def set_extract_pickerror(self, meth):
        """ Change the extract_pickerror method.

            NB: The `final_pd` attribute will be resetted! Remember
                to run again the `deliberate` method to define the new
                pick's attributes.
        """
        if (not isinstance(meth, str) or
           meth.lower() not in ('std', 'var')):
            logger.error("Invalid extraction method! ['std', 'var']")
            raise QE.InvalidType
        #
        self.extract_pe = meth
        self._reset_final_pd()
        logger.warning("The PICKERROR METHOD changed and FINAL_PD resetted!" +
                       "Remember to manually call the deliberate method!")

    def set_station_list(self, inlist):
        """ Change the station keys to be analyzed.
            NB: The `final_pd` attribute will be resetted! Remember
                to run again the `deliberate` method to define the new
                pick's attributes.
        """
        if not isinstance(inlist, (tuple, list)):
            raise QE.InvalidType()
        self.statlst = inlist
        self._reset_final_pd()
        logger.warning("The STAT LIST changed and FINAL_PD resetted!" +
                       "Remember to manually call the deliberate method!")

    def set_weighter_dict(self, indict):
        """ It will automatically update the class WEIGHTER object.
            NB: The `final_pd` attribute will be resetted! Remember
                to run again the `deliberate` method to define the new
                pick's attributes.
        """
        if not isinstance(indict, dict):
            raise QE.InvalidType()
        #
        self.weight_dict = indict
        self._reset_final_pd()
        logger.warning("The WEIGHTER DICT changed and FINAL_PD resetted!" +
                       "Remember to manually call the deliberate method!")

    def set_evaluator_dict(self, indict):
        """ It will automatically update the class EVALUATOR object
            NB: The `final_pd` attribute will be resetted! Remember
                to run again the `deliberate` method to define the new
                pick's attributes.
        """
        if not isinstance(indict, dict):
            raise QE.InvalidType()
        #
        self.eval_dict = indict
        self._reset_final_pd()
        logger.warning("The EVALUATOR DICT changed and FINAL_PD resetted!" +
                       "Remember to manually call the deliberate method!")

    def set_polarizer_dict(self, indict):
        """ It will automatically update the class POLARIZER object
            NB: The `final_pd` attribute will be resetted! Remember
                to run again the `deliberate` method to define the new
                pick's attributes.
        """
        if not isinstance(indict, dict):
            raise QE.InvalidType()
        #
        self.polar_dict = indict
        self._reset_final_pd()
        logger.warning("The POLARIZER DICT changed and FINAL_PD resetted!" +
                       "Remember to manually call the deliberate method!")

    def set_features_dict(self, indict):
        """ It will automatically update the class MINER object
            NB: The `final_pd` attribute will be resetted! Remember
                to run again the `deliberate` method to define the new
                pick's attributes.
        """
        if not isinstance(indict, dict):
            raise QE.InvalidType()
        #
        self.feat_dict = indict
        self._reset_final_pd()
        logger.warning("The FEATURES DICT changed and FINAL_PD resetted!" +
                       "Remember to manually call the deliberate method!")

    # ========================== Main

    def _classify_evidence(self, pick, stat,
                           tw_sig=0.15, tw_noi=0.3, use_raw=False):
        """ This method will classify the possible outcome of a
            classification method, to be verified still ...
            For the moment il will release the ratio of S/N
        """
        if use_raw:
            tr = self.rst.select(station=stat, channel=self.chan)[0]
        else:
            tr = self.pst.select(station=stat, channel=self.chan)[0]
        #
        sig = tr.slice(pick, pick+tw_sig)
        noi = tr.slice(pick-tw_noi, pick)
        #
        max_sig2noi_ratio = np.max(np.abs(sig.data))/np.max(np.abs(noi.data))
        return max_sig2noi_ratio

    def _refine_pick_hilbert(intr, pick, tw=0.1):
        """ This method refines the pick in time, searching for a minima
            in the slope close
        """
        tr = intr.copy()
        tr = tr.trim(pick-tw, pick)
        dt = tr.stats.delta
        env = np.sqrt((tr**2) + (signal.hilbert(tr) ** 2))
        env = np.flip(env, axis=0)
        diff_env = np.diff(QPR.smooth(env))  # -1 sample
        smooth_env = QPR.smooth(diff_env)
        _tmp = [True if smooth_env[ii]-smooth_env[ii-1] >= 0 else False
                for ii in range(1, len(smooth_env))]
        idx = None
        for ii in range(1, len(_tmp)):
            if ((_tmp[ii-1] is False and _tmp[ii] is True) or  # local minima
               ((_tmp[ii-1] is True and _tmp[ii] is False))):  # local maxima
                idx = ii-2
                break
        #
        return (pick-(idx*dt))

    def deliberate(self):
        """ This method run everythin needed to extract the final dict
            and store it !!
        """
        logger.debug("Start to deliberate!")
        # ------------------ Work
        if not self.statlst:
            logger.error("Missing the station_keys. This court is adjourned!")
            raise QE.MissingVariable()

        # loop over all the give station
        for _stat in self.statlst:

            # Extract STREAM TRACES of given station, if not found skip.
            _pd = self.pick_dict[_stat]
            _pwst = self.pst.select(station=_stat)
            _rwst = self.rst.select(station=_stat)

            # Initialize objects
            _weighter_obj = Weighter(_pd, **self.weight_dict)
            _polarizer_obj = Polarizer(_pwst, _rwst,
                                       UTCDateTime(),
                                       channel=self.chan,
                                       **self.polar_dict)
            _evaluate_obj = Gandalf(_pwst, _rwst,
                                    UTCDateTime(),
                                    channel=self.chan,
                                    **self.eval_dict)

            # Extract pick-time and its error
            if self.triage_method.lower() in ("ek", "mb", "ekmb"):
                _weighter_obj.triage()
            elif self.triage_method.lower() in ("jk", "jackknife"):
                _weighter_obj.triage_jk()
            else:
                raise QE.InvalidType("TRIAGE METHOD must be 'ekmb' or 'jk'")

            pick_time = _weighter_obj.get_picktime(method=self.extract_pt)
            pick_error = _weighter_obj.get_uncertainty(method=self.extract_pe)
            bootstrap_obj = _weighter_obj.get_bootstrap_obj()

            if pick_time and pick_error:
                ####################################################
                # ---->  ADAPT has finally picked the final <----- #
                ####################################################

                # Calculate SIGNALtoNOISE ratio for CLASS estimation
                classification = self._classify_evidence(pick_time,
                                                         _stat,
                                                         tw_sig=0.5,
                                                         tw_noi=0.2,
                                                         use_raw=False)

                # Add pick-time to weighter and evaluator object
                _polarizer_obj.set_pick_time(pick_time)
                _evaluate_obj.set_pick_time(pick_time)

                # Extract polarity
                _polarizer_obj.work()
                pick_polarity = _polarizer_obj.get_polarity()

                # Evaluate pick arrival (accept/reject)
                try:
                    _evaluate_obj.work()
                    pick_accepted = _evaluate_obj.get_verdict()
                except QE.MissingAttribute:
                    logger.warning("No EVALUATION DICT specified ... " +
                                   "Pick accepted by default!")
                    pick_accepted = True

                # v0.5.8: Added feature extraction possibility on FINAL
                # We will pass a dummy PICKDICT because MINER wants it
                # as mandatory, but we pass the FINALPICK and it will
                # use that one instead of the pickdict (check MINER man)
                if self.feat_dict:
                    _features_obj = Miner(_rwst, _pwst,
                                          self.pick_dict,   # DUMMY PICKDICT
                                          channel=self.chan,
                                          phase_list=None,
                                          feature_dict=self.feat_dict,
                                          pick_time=pick_time)  # USED
                    _features_obj.digging()
                    _features_dict = _features_obj.get_gold()
                else:
                    _features_obj = None
                    _features_dict = None

                # Compile final dict with attributes
                #  --------------- Store it and go home
                self.final_pd.addPick(_stat, self.final_pick_tag,
                                      boot_obj=bootstrap_obj,
                                      polarity=_polarizer_obj,
                                      weight=_weighter_obj,
                                      evaluate_obj=_evaluate_obj,
                                      features=_features_dict,
                                      features_obj=_features_obj,
                                      pickclass=classification,
                                      pickerror=pick_error,
                                      pickpolar=pick_polarity,
                                      evaluate=pick_accepted,
                                      timeUTC_pick=pick_time,
                                      timeUTC_early=UTCDateTime(
                                        min(
                                          dict(_weighter_obj.triage_dict['valid_obs']).values())),
                                      timeUTC_late=UTCDateTime(
                                        max(
                                          dict(_weighter_obj.triage_dict['valid_obs']).values()))
                                      )
                                      # timeUTC_early=pick_time - (pick_error/2.0),
                                      # timeUTC_late=pick_time + (pick_error/2.0))
            else:
                ####################################################
                # ------->  ADAPT didn't picked the final <------- #
                ####################################################
                # MB: False/True should came AFTER Gandalf only!
                pick_accepted = None
                # Store a bunch of None and exit
                self.final_pd.addPick(_stat, self.final_pick_tag,
                                      polarity=_polarizer_obj,
                                      weight=_weighter_obj,
                                      evaluate_obj=_evaluate_obj,
                                      pickerror=None,
                                      pickpolar=None,
                                      evaluate=pick_accepted,
                                      timeUTC_pick=None,
                                      timeUTC_early=None,
                                      timeUTC_late=None)

    # ========================== Get final / intermidiate results

    def get_final_pd(self):
        if self.final_pd:
            return self.final_pd
        else:
            logger.warning("Final PickContainer not available ... " +
                           "did you try to run the `deliberate` method first?")
            raise QE.MissingVariable()
