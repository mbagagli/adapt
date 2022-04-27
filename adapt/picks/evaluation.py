import logging
from obspy import Stream
from obspy import UTCDateTime
#
import adapt.picks.evaluation_functions as FUNCTIONS
import adapt.errors as QE

logger = logging.getLogger(__name__)


"""QUAKE evaluation module.

In this module are stored all the classes and functions to validate
a certain pick arrival time.

Note:
    The functions called by the Gandalf module class should be stored
    and named accordingly in the evaluation_functions.py module.

"""


class Gandalf(object):
    def __init__(self,
                 proc_stream,
                 raw_stream,
                 pick_time_UTC,
                 channel="*Z",
                 functions_dict=None):
        """This class decide who shall pass and who shall not

        As the name may suggest, this class orchestrate the given tests
        functions and store their results as a dictionary attribute.
        The getter class methods will help to retrieve the evaluation
        results and the resulting dictionary.

        Args:
            proc_stream (obspy.core.Stream): pre processed input obspy
                Stream object used in the evaluation test functions.
            raw_stream (obspy.core.Stream): raw input obspy Stream
                object used in the evaluation test functions.
            pick_time_UTC (obspy.core.UTCDateTime): associated time for
                the evaluation test functions trimming (i.e. the pick
                time to be analyzed)
            channel (:obj:`str`, optional): selected channel trace that
                will be passed to the test functions. Default to "*Z".
            functions_dict (:obj:`dict`, optional): a nested dictionary
                containing the names of test functions as main keys.
                Each of these keys should contain the parameters for the
                function itselfs. Default to None.

        Attributes:
            pst (obspy.core.Stream): pre processed input obspy
                Stream object used in the evaluation test functions.
            rst (obspy.core.Stream): raw input obspy Stream
                object used in the evaluation test functions.
            pickt (obspy.core.UTCDateTime): associated time for
                the evaluation test functions trimming (i.e. the pick
                time to be analyzed)
            wc (:obj:`str`, optional): selected channel trace that
                will be passed to the test functions. Default to "*Z".
            fd (:obj:`dict`, optional): a nested dictionary
                containing the names of test functions as main keys.
                Each of these keys should contain the parameters for the
                function itselfs. Default to None.

        Note:
            If the class is initialized without specifying the
            functions_dict parameter, the class will not automatically
            proceed with the investigation. Therefor the private method
            _work should be called separately.

        """
        # -- Checks
        if not isinstance(proc_stream, Stream):
            raise QE.InvalidType({'message':
                                  "proc_stream not a valid Stream object"})
        if not isinstance(raw_stream, Stream):
            raise QE.InvalidType({'message':
                                  "raw_stream not a valid Stream object"})
        # -- Storing
        self.pst = proc_stream.copy()
        self.rst = raw_stream.copy()
        self.pickt = pick_time_UTC
        self.wc = channel
        self.fd = functions_dict
        # -- Work
        self.res_dict = {}
        self.pwt = self.pst.select(channel=self.wc)[0]
        self.rwt = self.rst.select(channel=self.wc)[0]

    def work(self):
        """Private method that calls the evaluation functions

        Run all test, and append their results in self.res_dict
        attribute.

        """
        if not self.fd:
            raise QE.MissingAttribute({"message":
                                       "evaluation functions dict missing!"})

        logger.info("Evaluating ...")

        # MB: sorting in alphabetical order the testfunctions
        sortedkeys = sorted(self.fd, key=str.lower)
        for xx in sortedkeys:
            logger.debug("%s - %r" % (xx, self.fd[xx]))
            _funct = getattr(FUNCTIONS, xx)
            od = _funct(self.pwt, self.rwt, self.pickt, **self.fd[xx])
            #
            self.res_dict[xx] = od
        # Print results
        for xx in sortedkeys:
            logger.debug("%s: %s --> %s" % (xx,
                                            str(self.res_dict[xx]["result"]),
                                            str(self.res_dict[xx]["output"])))

    def who_failed(self):
        """Class method to extract failing functions

        Return the failed test functions name and relative output in a
        dictionary form.

        Returns:
            outdict (dict): a python dictionary containing the function
                name as key and relative results as parameter.

        """
        if not self.res_dict:
            raise QE.InvalidType({"message":
                                  "tests results dict missing!"})
        #
        outdict = {}
        for _kk in self.res_dict.keys():
            if not self.res_dict[_kk]["result"]:
                outdict[_kk] = self.res_dict[_kk]["output"]
        #
        return outdict

    def who_passed(self):
        """Class method to extract failing functions

        Return the positive test functions name and relative output in a
        dictionary form.

        Returns:
            outdict (dict): a python dictionary containing the function
                name as key and relative results as parameter.

        """
        if not self.res_dict:
            raise QE.InvalidType({"message":
                                  "tests results dict missing!"})
        #
        outdict = {}
        for _kk in self.res_dict.keys():
            if self.res_dict[_kk]["result"]:
                outdict[_kk] = self.res_dict[_kk]["output"]
        #
        return outdict

    def get_verdict(self):
        """Gandalf's final decision.

        Return the FINAL verdict by checking the boolean
        results dictionary.

        Returns:
            True if evaluation tests passed, False otherwise.

        """
        if not self.res_dict:
            raise QE.MissingAttribute({"message":
                                       "tests results dict missing!"})
        # Return results
        tests_res_bool = [self.res_dict[_kk]['result']
                          for _kk in self.res_dict.keys()]
        if False in tests_res_bool:
            logger.info("Evaluation picktests failed")
            return False
        else:
            logger.info("Evaluation picktests passed")
            return True

    def get_outdict(self):
        """Class method to extract the working dict attribute

        Extract the class dictionary containg all the test functions
        results.

        Returns:
            self.res_dict (dict): class attribute

        """
        if not self.res_dict:
            raise QE.InvalidType({"message":
                                  "tests results dict missing!"})
        return self.res_dict

    def set_pick_time(self, utc):
        if isinstance(utc, UTCDateTime):
            self.pickt = utc
        else:
            raise QE.InvalidType()
