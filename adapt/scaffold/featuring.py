import adapt.errors as QE
import adapt.scaffold.features_functions as FEATURES
#
import logging

logger = logging.getLogger(__name__)


class Miner():
    def __init__(self,
                 rawstream,
                 procstream,
                 pickdict,
                 phase_list=None,
                 feature_dict=None,
                 pick_time=None,    # UTCDateTime
                 channel="*Z"):
        """ This class will take care of extracting all the necessary
            features over the stream and returna  dictionary with results.

            USE ONLY ONE STREAM BELONGING TO THE SAME STATION

            The PHASE_LIST must be of the type:
                [(str,int),] --> [("PHASENAME", index),]

            This class will not modify directly the input pickContainer
            object. To do so, call the `store_in_place` method

            If the pick_time parameter is given, then it override the
            pick_time_search in the `digging` method. Neither the input
            pickContainer, nor the phaselist are useful anymore.
            Therefore, only the provided pick_time will be used for all
            the extraction steps.

        """
        self.__stat = procstream[0].stats.station
        self.pst = procstream
        self.rst = rawstream
        self.pd = pickdict
        self.fd = feature_dict
        self.phl = phase_list
        self.ch = channel
        self.pt = pick_time
        #
        self.res_dict = {}
        #
        self.pwt = None
        self.rwt = None
        self._setup()

    def _setup(self):
        """ WorkTrace channel selection """
        self.pwt = self.pst.select(channel=self.ch)[0]
        self.rwt = self.rst.select(channel=self.ch)[0]
        if self.pt:
            logger.warning("PICK_TIME parameter is set! " +
                           "Neither PickContainer, nor the phaselist are " +
                           "useful anymore.")

    def digging(self):
        """ Main method to extract features over the stream.
            It will also and append their results in self.res_dict
            attribute.
        """
        if not self.fd:
            raise QE.MissingAttribute({"message":
                                       "evaluation functions dict missing!"})
        if not self.phl and not self.pt:
            raise QE.MissingAttribute({"message":
                                       ("phase tag list missing! And no " +
                                        "pick_time parameter provided")})

        logger.debug("Extracting features on %s" % self.__stat)

        # MB: sorting in alphabetical order the testfunctions
        sortedkeys = sorted(self.fd, key=str.lower)

        # MB: Looping over given inputs (pick_time override phase list)
        if self.pt:
            pickt = self.pt
            self.res_dict["SINGLE_TIME"] = {}
            #
            for xx in sortedkeys:
                logger.debug("%s - %r" % (xx, self.fd[xx]))
                _funct = getattr(FEATURES, xx)
                try:
                    od = _funct(self.rwt, self.pwt, pickt, **self.fd[xx])
                    self.res_dict["SINGLE_TIME"][xx] = od
                except QE.InvalidVariable:
                    self.res_dict["SINGLE_TIME"][xx] = {}

                # Print results
                logger.debug("%s: %s" % (xx,
                                         str(self.res_dict["SINGLE_TIME"][xx]))
                             )
        else:
            for _pha in self.phl:
                self.res_dict[_pha[0]] = {}
                pickt = self.pd[self.__stat][_pha[0]][_pha[1]]['timeUTC_pick']
                for xx in sortedkeys:
                    logger.info("%s - %r" % (xx, self.fd[xx]))
                    _funct = getattr(FEATURES, xx)
                    try:
                        od = _funct(self.rwt, self.pwt, pickt, **self.fd[xx])
                        self.res_dict[_pha[0]][xx] = od
                    except QE.InvalidVariable:
                        self.res_dict[_pha[0]][xx] = {}

                    # Print results
                    logger.debug("%s: %s" % (xx, str(
                                                  self.res_dict[_pha[0]][xx])))

    def set_feature_dict(self, indict):
        if not isinstance(indict, dict):
            raise QE.InvalidType({"message":
                                  "wrong input type. A DICT needed!"})
        #
        self.fd = indict

    def set_channel(self, incha):
        if not isinstance(incha, str):
            raise QE.InvalidType({"message":
                                  "wrong input type. A DICT needed!"})
        #
        self.wc = incha
        self.res_dict = {}
        self._setup()
        logger.warning("Attention! RESULT DICT has been resetted, DIG again!")

    def get_gold(self):
        if self.res_dict:
            return self.res_dict
        else:
            logger.error("RESULT DICT missing! Run digging method first ...")
            raise QE.MissingAttribute({"message":
                                       ("RESULT DICT missing! Run digging " +
                                        "method first ...")})

    def store_in_place(self):
        """ Storing method of the Miner class.

        This method will store this object as-it-is and the pick results
        dict inside the inputs pickcontainer.

        !!! NB: It will override the features and features_obj
                fields of the given phaseslist tag and index

        !!! NB: if the pick_time parameter is given (UTCDateTime) when
                the class is initialized, then is not possible to
                store in place because of the lack of pointer.

        """
        if not self.res_dict:
            logger.error("RESULT DICT missing! Run digging method first ...")
            raise QE.MissingAttribute({"message":
                                       ("RESULT DICT missing! Run digging " +
                                        "method first ...")})
        if self.pt:
            logger.error("RESULT DICT can't be stored!")
            raise QE.MissingAttribute({"message":
                                       ("RESULT DICT can't be stored. The " +
                                        "pick_time parameter has been given!" +
                                        "Ambiguity on redirection")})
        #
        for _pha in self.phl:
            self.pd[self.__stat][_pha[0]][_pha[1]]['features'] = (
                                                        self.res_dict[_pha[0]])
            self.pd[self.__stat][_pha[0]][_pha[1]]['features_obj'] = self
