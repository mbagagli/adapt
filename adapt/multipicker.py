import os
import logging
#
from adapt import plot as QPLT
from adapt.database import PickContainer
from adapt.picks.evaluation import Gandalf
import adapt.pickers as QP
from adapt.scaffold.featuring import Miner
#
import adapt.errors as QE
from obspy.core.stream import Stream
from obspy.core.trace import Trace

logger = logging.getLogger(__name__)

# -------------------------------------------------- Class + Methods


class MultiPicker(object):
    """
    This class will  take care of slicing and picking the input stream
    In order to be properly initialized the trace need:
    The stream is assumed to be from the same station NAME!

    *** NB

    :type st: ObsPy.Stream
    :param st: input stream object used by the class
    :type adaptstat: adapt.database.StatContainer
    :param adaptstat: container of all the needed stations informations
    :type adaptpick: adapt.database.PickContainer
    :param adaptpick: container of all the needed picks informations
    :type phases:  list, optional
    :type pickers: dict, optional
    :type slices_delta:  list, optional
    :type features_quality:  list, optional
    :type features_phases:  list, optional

    Moving towards the need of mandatory raw stream in input as
    positional argument. Because of evaluation tests, and flexibility
    between classes.
    """

    def __init__(self,
                 eqid,
                 eqtag,
                 dicttag,
                 st,
                 rawdatastream,
                 associatedUTC=None,
                 associatedTAG=None,
                 pickers=None,
                 slices_delta=None,
                 slices_container=None,
                 slices_container_raw=None,
                 pickEvalConfigDict=None,
                 featuresExtractDict=None):
        # Initial Checks
        if not isinstance(st, Stream):
            raise QE.BadInstance({"type": "BadInstance",
                                          "message": "wrong Stream PROC",
                                          "wrong_type": str(type(st))})
        if not isinstance(rawdatastream, Stream):
            raise QE.BadInstance({"type": "BadInstance",
                                          "message": "wrong Stream RAW",
                                          "wrong_type": str(type(st))})
        # Variables
        self.__stat = st[0].stats.station  # used as index for dicts
        self.__eqid = eqid
        self.__pdtag = dicttag
        self.st = st.copy()
        self.rawst = rawdatastream.copy()
        self.pd = PickContainer(self.__eqid, eqtag, dicttag)
        self.associatedUTC = associatedUTC
        self.associatedTAG = associatedTAG,
        self.pickers = pickers
        self.slices_delta = slices_delta
        self.slices_container = slices_container
        self.pickEvalConfigDict = pickEvalConfigDict
        # Initialize the slicing for phases
        self.slices_container = None
        self.slices_container_raw = None
        if self.slices_delta:
            # create a self.slices_container
            self.sliceItUp(self.phases, self.slices_delta)

    # ============================ PRIVATE Methods
    # ============================
    def _checkStreamSlicing(self, refTime, noisew, sigw):
        """
        This function is working to avoid empty stram slicing for
        a specific phase reference time.

        # v0.0.7
        This function will return also the upper and lower boundary
        for the selected  slice. (enlarge noise and signal window)

        # v0.5.1
        This function will now only examine the boundaries on the
        channels, onli IF DATA EXIST.

        return a boolean to slice or not and the boundaries in UTC.

        """
        lb = refTime - noisew  # It's UTCDateTime obj
        ub = refTime + sigw    # It's UTCDateTime obj
        # MB: it's going to check on all the traces contained in stream
        boolList_proc = [
            (tr.stats.starttime <= lb <= tr.stats.endtime and
             tr.stats.starttime <= ub <= tr.stats.endtime)
            for tr in self.st if tr.data.size != 0]
        boolList_raw = [
            (tr.stats.starttime <= lb <= tr.stats.endtime and
             tr.stats.starttime <= ub <= tr.stats.endtime)
            for tr in self.rawst if tr.data.size != 0]

        if False in boolList_proc + boolList_raw:
            return False, lb, ub
        return True, lb, ub

    # ============================ PUBLIC Methods
    # ============================
    def sliceItUp(self):
        """
        Create a container with the different stream slices.
        The cuts are based on the slices_delta parameter (in seconds).

        `self.slices_container` contains `obspy.Stream` object
        """
        if not self.slices_delta:
            raise QE.MissingVariable({"type": "MissingSlicesDelta",
                                      "message": "missing slices_delta"})
        if not self.associatedUTC or not self.associatedTAG:
            raise QE.MissingVariable(
                       {"type": "MissingAssociated",
                        "message": "missing associatedUTC or associatedTAG"})
        #
        logger.info(("Creating Slices on MultiPicker object - %s : %r") %
                    (self.associatedTAG, self.associatedUTC))
        self.slices_container = {}
        self.slices_container_raw = {}
        for _xx, timedelta in enumerate(self.slices_delta):
            # v0.6.32: lb and ub are UTCDateTime obj
            streamSliceCheck, lb, ub = self._checkStreamSlicing(
                                            self.associatedUTC,
                                            # NOISEWIN  -  SIGWIN
                                            timedelta[0], timedelta[1])
            if streamSliceCheck:
                logger.debug(("Slicing stream for AssociatedTime: %s - " +
                              "NOISEWIN: %4.2f - SIGWIN: %4.2f") %
                             (self.associatedTAG, timedelta[0], timedelta[1]))
                self.slices_container[str(_xx)] = (
                                                 self.st.slice(lb, ub))
                self.slices_container_raw[str(_xx)] = (
                                                self.rawst.slice(lb, ub))

                # ======== MB: to store the cuts FOR ANALYSIS --> make it an option
                # _picker, _picker_val = tuple(self.pickers.items())[0]
                # --- For Real CUTS (i.e FP, BK)
                # self.store_cuts(self.st.slice(self.associatedUTC + lb, self.associatedUTC + ub).select(channel="*Z"),
                #                 "/mnt/DATA/QUAKE_TUNING_March2020_waveforms_11events/MULTIPICKING_CUTS",
                #                 ".".join((self.__eqid, self.__stat, _picker, str(timedelta), "sac")),
                #                 stream_save_mode="compact", format="SAC")
                # # --- For Long CUTS (i.e AIC, KURT)
                # self.store_cuts(self.st.slice(self.associatedUTC - 6, self.associatedUTC + 6).select(channel="*Z"),
                #                 "/mnt/DATA/QUAKE_TUNING_March2020_waveforms_11events/MULTIPICKING_CUTS",
                #                 ".".join((self.__eqid, self.__stat, _picker, str(timedelta), "sac")),
                #                 stream_save_mode="compact", format="SAC")
                # ============================================================

            else:
                # If ended up here the `self.checkStreamSlicing`
                # returned FALSE, therefore some fishy start/end trace
                # times are present in the stream, better raise error.

                # NB: The check is made on ALL the trace in the stream,
                #     therefore the interested channel could still be ok
                #     while not the the others --> maybe improve bugfix
                # ... and continue picking :)

                # NB: v0.5.1 now the check is made on ALL channels only
                #            IF DATA EXIST for that channel!
                logger.warning((
                            "NO SLICE for AssociatedTime: %s -" +
                            "NOISEWIN: %4.2f - SIGWIN: %4.2f") %
                            (self.associatedTAG, timedelta[0], timedelta[1]))
                # *** IMPORTANT ***
                # Next raise d error is a temporary FIX:
                # Multipicker missing slice -> wrong indexing in PickDict class
                #         --> MAKE SURE ALL SLICES ARE CREATED <--
                raise QE.MissingVariable({
                            "type": "MissingSlice",
                            "message": ((
                              "NO SLICE for AssociatedTime: %s - " +
                              "NOISEWIN: %4.2f - SIGWIN: %4.2f") %
                              (self.associatedTAG, timedelta[0], timedelta[1]))
                            })
        # before passing to another phase, if no slices
        # could have been found for it --> delete it !
        # This action may arise QE.MissingVariable
        # in runPickers method.
        if not self.slices_container:
            raise QE.MissingVariable({
                                      "type": "MissingSlicesContainer",
                                      "message": "missing slices_container"})

    def runPickers(self):
        """
        Wrapper around the possible pickers in ADAPT
        """

        if not self.associatedUTC or not self.associatedTAG:
            raise QE.MissingVariable(
                       {"type": "MissingVariable",
                        "message": "missing associatedUTC or associatedTAG"})
        if not self.slices_container:
            raise QE.MissingVariable({"type": "MissingVariable",
                                      "message": "missing slices_container"})
        if not self.pickers:
            raise QE.MissingVariable({"type": "MissingVariable",
                                      "message": "missing list of pickers"})
        #
        logger.info("Running pickers on slices!" % self.pickers)
        logger.debug("%r" % self.pickers)

        # For each slice ... For each picker
        for ss in self.slices_container.keys():
            for pp in self.pickers.keys():
                # stream selection
                workst = self.slices_container[ss]
                workstraw = self.slices_container_raw[ss]

                # MB: [22022019]
                # better to remove the mean everytime we pick
                # *** This technique should be worthy for all pickers

                # minimal processing for the PROCSTREAM
                workst.detrend('demean')

                # minimal processing for the RAWSTREAM
                workstraw.detrend('demean')  # 22022019
                workstraw.detrend('linear')  # 11032019

                # MB: Next line is a default --> DO NOT DELETE !!!
                pick, pick_info = None, None

                if workst:
                    # Switch for pickers
                    if pp.lower() in ("bait", "iterativebk", "itbk"):
                        logger.info("BaIt picking on SLICE: %s" % ss)
                        pick, pick_info, bobj = QP.baitwrap(
                                                        workst,
                                                        stream_raw=workstraw,
                                                        channel="*Z",
                                                        **self.pickers[pp])
                        self.evaluateAndStore(workst, workstraw, pick,
                                              "BAIT_"+self.__pdtag,
                                              pick_info=pick_info,
                                              evaluation_dict=(
                                                self.pickEvalConfigDict),
                                              features_dict=(
                                                self.featuresExtractDict))

                    elif pp.lower() in ("bk", "baerkradolfer", "baer"):
                        logger.info("BaerKradolfer picking on SLICE: %s" % ss)

                        pick, pick_info, _CF, _bkix = QP.BaerKradolfer(
                                                    workst,
                                                    bkparam=self.pickers[pp])
                        self.evaluateAndStore(workst, workstraw, pick,
                                              "BK_"+self.__pdtag,
                                              pick_info=pick_info,
                                              evaluation_dict=(
                                                self.pickEvalConfigDict),
                                              features_dict=(
                                                self.featuresExtractDict))

                    elif pp.lower() in ("aic", "myaic", "myakaike"):
                        logger.info("AIC picking on SLICE: %s" % ss)
                        #
                        if self.pickers[pp]['useraw']:
                            pick, AIC, AICidx = QP.MyAIC(
                                    workstraw,
                                    **{_kk: _vv
                                       for _kk, _vv in self.pickers[pp].items()
                                       if _kk not in ('useraw')})
                        else:
                            pick, AIC, AICidx = QP.MyAIC(
                                    workst,
                                    **{_kk: _vv
                                       for _kk, _vv in self.pickers[pp].items()
                                       if _kk not in ('useraw')})

                        # --------------- If you want to have the window
                        #                 search as the picker sees it
                        if self.featuresExtractDict and pick is not None:
                            sigwintime = workst[0].stats.endtime-pick
                            noiwintime = pick-workst[0].stats.starttime
                            for _fe, _fep in self.featuresExtractDict.items():
                                if _fe in ('signal_over_threshold',):
                                    _fep['signal_window'] = sigwintime
                                elif _fe in ('noise_over_threshold',):
                                    _fep['noise_window'] = noiwintime
                                else:
                                    _fep['signal_window'] = sigwintime
                                    _fep['noise_window'] = noiwintime
                        # ------------------------------------------ MB

                        self.evaluateAndStore(workst, workstraw, pick,
                                              "AIC_"+self.__pdtag,
                                              pick_info=None,
                                              evaluation_dict=(
                                                self.pickEvalConfigDict),
                                              features_dict=(
                                                self.featuresExtractDict))

                    elif pp.lower() in ("kurt", "kurtosis", "hos_k",
                                        "hos", "host", "skew", "skewness"):
                        logger.info("HOST picking on SLICE: %s" % ss)

                        # --- MyHOS
                        if self.pickers[pp]['useraw']:
                            pick, kurt, kuAIC, kuidx = QP.HOS(
                                    workstraw,
                                    channel="*Z",
                                    **{_kk: _vv
                                       for _kk, _vv in self.pickers[pp].items()
                                       if _kk not in ('useraw')})
                        else:
                            pick, kurt, kuAIC, kuidx = QP.HOS(
                                    workst,
                                    channel="*Z",
                                    **{_kk: _vv
                                       for _kk, _vv in self.pickers[pp].items()
                                       if _kk not in ('useraw')})

                        # self.evaluateAndStore(workst, workstraw, pick,
                        #                       "KURT_"+self.__pdtag,
                        #                       pick_info=None,
                        #                       evaluation_dict=(
                        #                         self.pickEvalConfigDict),
                        #                       features_dict=(
                        #                         self.featuresExtractDict))
                        self.evaluateAndStore(workst, workstraw, pick,
                                              "HOS_"+self.__pdtag,
                                              pick_info=None,
                                              evaluation_dict=(
                                                self.pickEvalConfigDict),
                                              features_dict=(
                                                self.featuresExtractDict))

                    elif pp.lower() in ("filterpicker", "fp"):
                        logger.info("FILTERPICKER picking on SLICE: %s" % ss)
                        pick, pickUnc, pickBand = QP.fp_wrap(
                                                        workst,
                                                        fppar=self.pickers[pp],
                                                        channel="*Z")
                        self.evaluateAndStore(workst, workstraw, pick,
                                              "FP_"+self.__pdtag,
                                              pick_info=None,
                                              evaluation_dict=(
                                                self.pickEvalConfigDict),
                                              features_dict=(
                                                self.featuresExtractDict))

                    else:
                        logger.warning("No Valid Picker found for %s" % pp)
                        continue  # go to next picker
                else:
                    # Slice stream is NoneType (probably empty) --> skip
                    logger.warning("Empty SLICE - %s ... skipping" % ss)
                    continue

    def evaluateAndStore(self,
                         workst,
                         workstraw,
                         pickUTC,
                         storekey,
                         pick_info=None,
                         evaluation_dict=None,
                         features_dict=None):
        """
        This separate function take care of the switches and
        evaluation part of our resulting picks and the storing
        of the picks in a adapt.database.PickContainer object.

        INPUT: - UTCDateTime obj (pick)
               - pick_info (string)
        """

        # ================== EVALUATE and Extract FEATURES
        eval_obj, eval_result = self._evaluate_pick(
                                      storekey,
                                      workst,
                                      workstraw,
                                      pickUTC,
                                      evaluation_dict=evaluation_dict)
        feat_obj, feat_result = self._extract_features(
                                      storekey,
                                      workst,
                                      workstraw,
                                      pickUTC,
                                      features_dict=features_dict)

        # ================== Just STORE
        """ The next if-check, is done to keep the picker logic equal to
            previous version (<= v0.5.2).
            The _evaluate_pick method ALWAYS return TRUE or FALSE.

            If the EVALUATION dict is NOT SPECIFIED, then the PICK is
            accepted automatically without further `quality-check`.
                - For this case both the evaluation_result par and
                  evaluation_obj par are stored as NONE

            If the EVALUATION dict is SPECIFIED, the test are done.
            In this case, the boolean return of _evaluate_pick is taken
            into consideration and used to decide if the observation
            should be stored or not.

            Being the return of _evaluate_pick FALSE either if the
            EVALUATION dict is:
                1) provided + test FAILED
                2) not provided
            The classical check `if not X` return True for both
            boolean False or empty variables, a specific check on the
            actual boolean False is needed.
        """

        if eval_obj and eval_result is False:
            pt = None  # rejected
            ptb = pickUTC
        elif not eval_obj and eval_result is False:
            # Pick not evaluated ==> accepted by defaults
            pt = pickUTC
            eval_result = None
            eval_obj = None
            ptb = None
        else:
            pt = pickUTC
            ptb = None

        self._store_pick(storekey,
                         pickUTC=pt,
                         pick_info=pick_info,
                         evaluate=eval_result,
                         evaluate_obj=eval_obj,
                         features=feat_result,
                         features_obj=feat_obj,
                         # v0.6.14
                         backup_picktime=ptb)

        # Closing and keep the loogin equal to previus version
        if not pickUTC:
            logger.warning("NO pick found: %s" % storekey)

    def _evaluate_pick(self,
                       storekey,
                       workst,
                       workstraw,
                       pickUTC,
                       evaluation_dict=None):
        """ This method takes care of the single evaluation steps
            needed to verify the effective validity of the observation.

            !!! Evaluation results is ALWAYS True or False !!!
            !! Evaluation object is retruned None, ONLY if not given !!
        """
        evaluationResult = False
        if pickUTC:
            if evaluation_dict:
                eval_obj = Gandalf(workst,
                                   workstraw,
                                   pickUTC,
                                   channel="*Z",
                                   functions_dict=evaluation_dict)
                eval_obj.work()
                evaluationResult = eval_obj.get_verdict()
                if not evaluationResult:
                    logger.warning("NO valid pick: %s" % storekey)
            else:
                logger.warning("NO validation stage provided: %s" % storekey)
                eval_obj = None
        else:
            eval_obj = None
        #
        return eval_obj, evaluationResult

    def _extract_features(self,
                          storekey,
                          workst,
                          workstraw,
                          pickUTC,
                          features_dict=None):
        """ This method use the adapt.scaffold.Miner class to extract
            the needed features.

            We need the pickDict and the store key to check if the pick
            has been accepted, otherwise we skip.
        """
        if pickUTC and features_dict:
            extractor = Miner(workstraw,
                              workst,
                              self.pd,
                              feature_dict=features_dict,
                              pick_time=pickUTC,
                              channel="*Z")
            extractor.digging()
            extractor_resultdict = extractor.get_gold()
        else:
            logger.warning(("NO feature extraction stage provided, " +
                            "or MISSING pick: %s") % storekey)
            extractor = None
            extractor_resultdict = None
        #
        return extractor, extractor_resultdict

    def _store_pick(self,
                    storekey,
                    pickUTC=None,
                    pick_info=None,
                    evaluate=None,
                    evaluate_obj=None,
                    features=None,
                    features_obj=None,
                    backup_picktime=None):
        """ This method takes care of storing the given observation in the
            given class pickContainer. Make sure to have all the needed info
            before call this method because in the multipicker stage it's
            though, if not impossible, to implement the slicing count.
            This is a FEATURE rather than a PITFALL because it epand the
            multipicking possibility

                        !!! MUST BE CALLED FOR LAST !!!
        """
        if pick_info:
            self.pd.addPick(self.__stat, storekey,
                            timeUTC_pick=pickUTC,
                            onset=pick_info[0],
                            polarity=pick_info[2],
                            pickclass=pick_info[3],
                            evaluate=evaluate,
                            evaluate_obj=evaluate_obj,
                            features=features,
                            features_obj=features_obj,
                            backup_picktime=backup_picktime)
        else:
            self.pd.addPick(self.__stat, storekey,
                            timeUTC_pick=pickUTC,
                            evaluate=evaluate,
                            evaluate_obj=evaluate_obj,
                            features=features,
                            features_obj=features_obj,
                            backup_picktime=backup_picktime)

    def getPickContainer(self):
        """
        Method to extract the adapt pick container instance of
        the multipicker object
        """
        return self.pd

    def store_cuts(self,
                   st,
                   store_dir,
                   file_name,
                   format="MSEED",
                   stream_save_mode="compact"):
        """ This method will store the stream cuts passed to the single
            pickers. Can be useful for further analysis.

            - st could be either a stream or a trace: the obspy write
              method is the same for both
            - stream_mode defines the way to store it:
                - "compact" will store the Stram all at once
                -
        """
        if isinstance(st, Stream):
            if stream_save_mode.lower() == "compact":
                st.write(os.sep.join((store_dir, file_name)),
                         format=format.upper())
            elif stream_save_mode.lower() == "unpack":
                for _ii, tr in enumerate(st):
                    trace_name = "_".join((
                                    os.sep.join((store_dir, file_name)),
                                    tr.stats.channel, str(_ii+1)
                                    ))
                    tr.write(trace_name, format=format.upper())
            else:
                raise QE.InvalidParameter("Stream Save Mode must be "
                                          "either COMPACT or UNPACK!")
        elif isinstance(st, Trace):
            tr.write(os.sep.join((store_dir, file_name)),
                     format=format.upper())

        else:
            raise QE.InvalidType("Input object must be either "
                                 "obspy.Stream or obspy.Trace object!")
