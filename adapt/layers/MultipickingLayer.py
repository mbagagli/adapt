import sys
import logging
#
from adapt.picks.weight import Weighter
from adapt.picks.polarity import Polarizer
from adapt.picks.evaluation import Gandalf
from adapt.picks.analysis import Judge
#
import adapt.multipicker as QMP
import adapt.database as QD
import adapt.errors as QE

logger = logging.getLogger(__name__)

# 04072019
# MB: Possible implementation --> if phase identification class is
#     created it should return a string with the phasename and added
#     in the definePick method.
#     At the moment the final pick name is defined by the user
#     initializing the layer


# ----------------------------------------------------------- Classes
class Hive(object):
    """
    This Class take care of the multipicking stage.
    **Inputs:**
    - *procSt (`obspy.core.Stream`):* contains the processed stream
    - *rawSt (`obspy.core.Stream`):* contains the raw stream
    - *associated_time (`obspy.core.UTCDateTime`):* reference time
      for picking
    - *mp_dict (`dict`):* taken from configuration file, or use the
      same scheme and hierarchy for new parameters
    """
    def __init__(self,
                 event_id,
                 event_tag,
                 procSt,
                 rawSt,
                 associated_time_UTC=None,
                 associated_time_tag=None,
                 associated_time_attribute_dict=None,
                 multipicking_dict=None,
                 multipicking_tag="MP"):

        # MB: the Stream should contain only the data regarding ONE stat
        self.__stat = rawSt[0].stats.station
        self.eqid = event_id
        self.eqtag = event_tag
        self.pst = procSt.copy()
        self.rst = rawSt.copy()
        self.asstm = associated_time_UTC
        self.asstm_tag = associated_time_tag
        self.asstm_attr = associated_time_attribute_dict
        self.mpd = multipicking_dict
        self.mpt = multipicking_tag
        self.work_pd = QD.PickContainer(self.eqid, self.eqtag, "temporary")
        # self.final_pd = QD.PickContainer(self.eqid, self.eqtag, "final")
        self.final_pd = None
        # self.polarity_delta = polarity_sec_after_pick

    def _add_associated_pick(self):
        """
        Add to the working dir the ASSOCIATEDTIME
        (in this case BAIT).

        **NB**: in any case, the associated time / tag will be stored.
                no matter if differently specified by attribute_dict
        """
        if not self.asstm or not self.asstm_tag:
            logger.warning("No associated UTC time / tag specified ... " +
                           "nothing happened!")
        else:
            if self.asstm_attr:
                self.work_pd.addPick(self.__stat,
                                     self.asstm_tag,
                                     **{**self.asstm_attr,
                                        **{'timeUTC_pick': self.asstm}
                                        }
                                     )
            else:
                self.work_pd.addPick(self.__stat,
                                     self.asstm_tag,
                                     **{'timeUTC_pick': self.asstm})

    # MB: todo-list implement the check
    def _check_config(self):
        """ Double check all picking parameters are fine """
        if self.mpd and self.asstm:
            return True
        else:
            raise QE.MissingVariable()

    # MB: - make sure the next method is storing the pickDict in a
    #       temporary one (internal of the class)
    #     - at the moment the option `pickableStream` of the MP obj is
    #       always set to true
    def pick(self):
        """ Run all the pickers """
        if self._check_config():
            for _mptag in self.mpd.keys():

                # --- Initial steps before multipicking calls
                mp_procst = self.pst.copy()
                mp_rawst = self.rst.copy()
                QuakeMP = None             # resetting multipick class

                # --- Get Picker Name
                pickername = _mptag.split("_")[-1]

                # --- Checks
                if not self.mpd[_mptag]["doit"]:
                    logger.warning("%s is set to False ... skip!" % _mptag)
                    continue

                # ===============================
                # 0) initialize multipickerobject
                try:
                    QuakeMP = QMP.MultiPicker(self.eqid,
                                              self.eqtag,
                                              self.mpt,
                                              mp_procst,
                                              mp_rawst)
                except QE.BadInstance as e:
                    logger.error("FATAL: %s : %s" % (e.args[0]["message"] +
                                                     e.args[0]["wrong_type"]))
                    sys.exit()

                # ===============================
                # 1) Prepare PICKERS infos
                # *** NB if you want the evaluation process to be done
                #        add the correct pickEvalConfigDict to the
                #        multipicker object
                QuakeMP.pickers = self.mpd[_mptag]["pickers"]

                # MB: The next line make sure that all the multipicking
                #     picks are accepted. If a dictionary is provided,
                #     an internal check (inside `multipicker_new` class)
                #     is done. The evaluation over the median/mean pick
                #     is done by this class (`Hive`) with the Gandalf
                #     class object inside the `definePicks` method.
                try:
                    QuakeMP.pickEvalConfigDict = self.mpd[_mptag]['evaluate']
                except KeyError:
                    # MB: in case user didn't specify in config file
                    QuakeMP.pickEvalConfigDict = None

                # MB: The next line defines the features extraction dict
                #     that the user can provide. Inside the Multipicker
                #     class, the Miner class will take care of the rest.
                #     This info are stored in each pick observation, to
                #     be retrieved afterwards
                try:
                    QuakeMP.featuresExtractDict = self.mpd[_mptag]['features']
                except KeyError:
                    # MB: in case user didn't specify in config file
                    QuakeMP.featuresExtractDict = None

                # ===============================
                # 2) Create Slices
                try:
                    QuakeMP.associatedUTC = self.asstm
                    QuakeMP.associatedTAG = self.asstm_tag
                    QuakeMP.slices_delta = (self.mpd[_mptag]
                                                    ["slicesdelta"])
                    QuakeMP.sliceItUp()

                except QE.MissingVariable as e:
                    if e.args[0]["type"] == "MissingSlicesContainer":
                        logger.warning("Event: %s - Station: %s - %s skip!" % (
                                                    self.eqid, self.__stat,
                                                    e.args[0]["message"]))
                        continue
                    else:
                        # 09062019
                        # If ended up here the `self.checkStreamSlicing`
                        # returned FALSE, therefore some fishy start/end trace
                        # times are present in the stream, better raise error.
                        logger.error("%s: %s" % (e.args[0]["type"],
                                                 e.args[0]["message"]))
                        continue  # 09062019

                # ===============================
                # 3) Run PICKERS!
                try:
                    QuakeMP.runPickers()
                except QE.MissingVariable as e:
                    logger.error("%s: %s" % (e.args[0]["type"],
                                             e.args[0]["message"]))
                    continue

                # ===============================
                # 4) Extract pickdict
                _tmppd = QuakeMP.getPickContainer()

                # ========================== End of SINGLE multipicking
                # 5) populate layer class internal pickdict
                self.work_pd.addStat(self.__stat, {**_tmppd[self.__stat]})

        else:
            logger.error("Invalid MULTIPICKING parameter dictionary")
            sys.exit()

    def define_pick(self, judger_dict_config):
        """ This method will create an ad-hoc Judge class to extract
            the final pick from the multipicking collection.
        """

        # Checks
        if not isinstance(judger_dict_config, dict):
            logger.error("Judge config must be a dictionary!")
            raise QE.InvalidType()

        # ===================================== JudgerClass
        judger = Judge(self.work_pd,
                       (self.__stat,),
                       self.pst,
                       self.rst,
                       **judger_dict_config)
        judger.deliberate()
        self.final_pd = judger.get_final_pd()

        # ===================================== Phase identification
        # --- To be implemented: next obj. should contain all methods
        # phase_id_obj = adapt.picks.phaseid.Passport()
        # final_phasename = phase_id_obj.define()

    def get_work_pick_dict(self):
        """ Extract MultiPicking dict with error uncertainties and
            everything related.
        """
        if len(self.work_pd.getStats()) < 1:
            raise QE.MissingPicks()
        return self.work_pd

    def get_final_pick_dict(self):
        """
        Extract final pick with error uncertainties and everything related
        """
        if len(self.final_pd.getStats()) < 1:
            raise QE.MissingPicks()
        return self.final_pd
