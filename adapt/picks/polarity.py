import logging
import numpy as np
from obspy import Stream
from obspy import UTCDateTime
#
import adapt.errors as QE

logger = logging.getLogger(__name__)


"""ADAPT polarity module.

In this module are stored all the classes and functions to define the
polarity of a certain phase arrival time.

"""

# --------------------------------------------- Classes DEF


class Polarizer(object):
    def __init__(self,
                 proc_stream,
                 raw_stream,
                 pickUTC,
                 channel="*Z",
                 definition_method="conservative",
                 sec_after_pick=0.1,
                 use_raw=False):
        """This class defines the pick polarity

        As the name may suggest, this class orchestrate a simple class
        method to define the pick polarity.

        Args:
            instream (obspy.core.Stream): pre processed input obspy
                Stream object used in the evaluation test functions.
            pickUTC (obspy.core.UTCDateTime): associated time for
                the evaluation test functions slicing (i.e. the pick
                time to be analyzed)
            channel (:obj:`str`, optional): selected channel trace that
                will be passed to the test functions. Default to "*Z".
            sec_after_pick (:obj:`float`, optional): the seconds after
                the pickUTC that will be used to estimate the polarity.
                Default to 0.1.
            definition_method (:obj:`str`, optional): the identification
                method used to calculate the polarity of the arrival.
                The optional methods are 'conservative' if the user
                want a more confident estimation (at cost of more empty
                polarity results) or 'simple' where the estimation is
                carried by simple derivative and sign counts.
                The former method is suggested for regional seismicity,
                while the latter can be used in case of microseismicity.
                Deafult set to "conservative".

        Attributes:
            opstream (obspy.core.Stream): pre processed input obspy
                Stream object used in the evaluation test functions.
            picktime (obspy.core.UTCDateTime): associated time for
                the evaluation test functions trimming (i.e. the pick
                time to be analyzed)
            wtr (obspy.core.Trace) working trace extracted from the
                input Stream object used by the class. Default to None.
            wc (str): selected channel trace that will be passed to
                the test functions. Default to "*Z".
            sap (float): time window in seconds for the polarity eval.
            polarity (str): single character string with the subsequent
                definition. 'U' for up-polarity, 'D' for down-polarity,
                '_' for unknown polarity
            method (str): the identification method used to calculate
                the polarity of the arrival.

        Note:
            If the class is initialized without specifying the
            channel parameter, the class will not automatically
            select the working trace, and cannot perform the test.
            Therefore the private method _setup and _work should be
            called separately.

        """
        if (not isinstance(proc_stream, Stream) or
           not isinstance(raw_stream, Stream) or
           not isinstance(pickUTC, UTCDateTime)):
            logger.error("Wrong input variables type")
            raise QE.InvalidType()
        # Unwrap input parameters
        self.pst = proc_stream
        self.rst = raw_stream
        self.picktime = pickUTC
        self.sap = sec_after_pick
        self.method = definition_method
        # Define Class atttributes
        self.wtr = None       # (obspy.Trace/None)
        self.polarity = None  # (str/None)
        # Work
        self._setup(channel, use_raw)

    def _setup(self, chan, use_raw):
        """ Private class method for selecting the working trace.

            Args:
                trace_channel (str): selected channel trace that
                    will be passed to the test functions. ["*Z"]
                use_raw (bool): select the channel from the raw stream
                                [False]
        """
        if not isinstance(chan, str) or not isinstance(use_raw, bool):
            raise QE.InvalidType()
        #
        if use_raw:
            self.wtr = self.rst.select(channel=chan)[0]
        else:
            self.wtr = self.pst.select(channel=chan)[0]

    def _simple_method(self):
        """This method will apply a first derivative over the selected
        window after the pick. It will then count the number of positive
        signs (increasing value) and the number of negative signs ()

        """
        # MB: sec*freq -> nsample
        myslice = self.wtr.slice(self.picktime, self.picktime + self.sap)
        pos_sign = np.where(np.diff(myslice.data) > 0)[0].size
        neg_sign = np.where(np.diff(myslice.data) < 0)[0].size
        # Switch
        if pos_sign > neg_sign:
            self.set_polarity("U")
        elif pos_sign < neg_sign:
            self.set_polarity("D")
        elif pos_sign == neg_sign:
            self.set_polarity("-")
        else:
            logger.error(("Critical error! POS_SIGN: %f - NEG_SIGN: %f") %
                         (pos_sign, neg_sign))
            raise QE.MissingVariable()

    def _conservative_method(self):
        """This method try to imitate the eye of a seismologist
        to determine the polarity, checking that the gradient is
        continuos across the time window specified

        1.st check if value at time is + or -
        2.nd check if gradient respectively positive or negative

        """
        if not self.method:
            logger.error("Missing method specification")
            raise QE.MissingVariable()

        # Switch
        myslice = self.wtr.slice(self.picktime, self.picktime + self.sap)
        mygrad = np.diff(myslice.data)
        if (myslice.data[0] > 0 and (mygrad > 0).all()):
            self.set_polarity("U")
        elif (myslice.data[0] < 0 and (mygrad < 0).all()):
            self.set_polarity("D")
        elif myslice.data[0] == 0:
            if (mygrad > 0).all():
                self.set_polarity("U")
            elif (mygrad < 0).all():
                self.set_polarity("D")
            else:
                self.set_polarity("-")
        else:
            self.set_polarity("-")

    def work(self):
        """Pursue the effective calculation with a derivative

        To calculate the polarity, this method calculate the absolute
        derivative of the selected trace data and then count the
        respective amount of increase or decrease values.
        The resulting polarity is then stored as a class attribute.

        Args:
            delta_time (float): the seconds after the pickUTC that will
                be used to estimate the polarity. Default to 0.1.

        """
        if not self.method:
            logger.error("Missing method specification")
            raise QE.MissingVariable()

        # Switch
        if self.method.lower() == "conservative":
            self._conservative_method()
        elif self.method.lower() == "simple":
            self._simple_method()
        else:
            raise QE.InvalidParameter()

    def set_polarity(self, pol):
        """ Class setter method to return the object polarity attribute

            Returns:
                polarity (str): 'U' for up-polarity, 'D' for down-polarity,
                    '_' for unknown polarity.

        """
        if not isinstance(pol, str):
            logger.error("Invalid polarity")
            raise QE.InvalidType()
        #
        self.polarity = pol


    def set_pick_time(self, utc):
        """Class setter method to return the object polarity attribute

        Returns:
            polarity (str): 'U' for up-polarity, 'D' for down-polarity,
                '_' for unknown polarity.

        """
        if isinstance(utc, UTCDateTime):
            self.picktime = utc
        else:
            logger.error("Input object must be an obspy.UTCDateTime instance!")
            raise QE.InvalidType()


    def get_polarity(self):
        """Class getter method to return the object polarity attribute

        Returns:
            polarity (str): 'U' for up-polarity, 'D' for down-polarity,
                '_' for unknown polarity.

        """
        if self.polarity:
            return self.polarity
        else:
            logger.warning("Polarity is empty ... " +
                           "(HINT) did you run the Polarizer.work() method?")
