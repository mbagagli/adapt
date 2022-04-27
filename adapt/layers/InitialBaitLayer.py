import logging
import os
from obspy import Stream, Trace
#
from bait.bait import BaIt
import bait.bait_errors as BE
#
import adapt.errors as QE

logger = logging.getLogger(__name__)

# 05072019
# MB: This layer should take care of the necessary for a BaIt
#     pick refining step.
#     Possible implementation: make it work with multiple station
#     stream
#
# MB: Remember to STORE BAIT OBJECT in the MAIN SCRIPT
# # Store bait object +save pick
#  QuakeBaitEval[statname] = bobj
#  # As the pickinfo for ObsPyBk is much stronger bound to result.
#  # If it's there, also the time will be.
#  if MP_one_pickInfo:
#      QuakePick_ALL.addPick(statname, "BAIT_EVAL",
#                            timeUTC_pick=MP_one_assUTC,
#                            onset=MP_one_pickInfo[0],
#                            polarity=MP_one_pickInfo[2],
#                            pickclass=MP_one_pickInfo[3])


class BaitBeautify(object):
    """
    This layer class will pick and host the methods for BaIt picking
    algorithm. The class need an initial pick to trim the stream.
    """
    def __init__(self,
                 eqtag,  # str
                 proc_stream,
                 raw_stream,
                 associated_time_UTC,
                 working_dict,
                 working_channel="*Z"):
        # Private
        self.__stat = proc_stream[0].stats.station
        self.eqtag = eqtag
        self.itbk = None
        # Mandatory
        self.pst = proc_stream.copy()
        self.rst = raw_stream.copy()
        self.assUTC = associated_time_UTC
        self.wd = working_dict
        # Optional
        self.wch = working_channel

    def _baitwrap_layer(self, **kwargs):
        """
        Class BaIt wrapper build to keep things neat on the code.
        This method will call bait picking algorithm, and store the
        BaIt class object already picked as a layer attribute.
        The picks be extracted with the extract_pick method.
        """
        # Initializeclass
        self.itbk = BaIt(self.pst,
                         stream_raw=self.rst,
                         channel=self.wch,
                         **kwargs)
        # MB: to see that a new instance is created everytime and class
        #     is resetted, otherwise check why there's already some pick
        if self.itbk.baitdict:
            logger.error(
                "BaIt istance is not correctly resetted, double check!")
            # ---------  MB: debug
            import pdb
            pdb.set_trace()
            # --------------------
            logger.error(self.itbk.baitdict)
            raise QE.BadInstance()

        # Run the picker
        try:
            self.itbk.CatchEmAll()
        except BE.MissingVariable:
            raise QE.NoBaitValidPick

    def pickMe(self, store_input_cuts=False, store_dir=None):
        """ Run BaIt algorithm and store all picks
            store_input_cuts:
                    will store the cuts AFTER the trimming
                    if some are missing is because trim went wrong
        """

        # =====================================================
        # 0) trim the stream around the prediction + removemean
        for _bst in (self.pst, self.rst):
            _bst.trim(self.assUTC -
                      self.wd['slicedel'] -
                      self.wd['enlargenoisewin'],

                      self.assUTC +
                      self.wd['slicedel'] +
                      self.wd['enlargesignalwin'])
            _bst.detrend('demean')

        # --- Check that stream EXIST after TRIM
        # v0.5.1 - 20112019 (update - bug patch)
        # additional check that the desired channel exist
        if (not self.pst or
           not self.rst or
           len(self.pst.select(channel=self.wch)) == 0):
            logger.error("%s - Empty stream after BAIT EVAL TRIM" %
                         self.__stat)
            raise QE.MissingVariable({'message': "Empty stream"})

        # =====================================================
        # 1) store_cuts in input to BAIT (optional)
        #    At this stage self.pst/rst are TRIMMED.
        if store_input_cuts:
            if not store_dir:
                raise QE.MissingVariable(
                                "Please provide a STORAGE DIR for cuts!")
            #
            file_name = "_".join([self.eqtag, self.__stat, "BAIT_PROC.SAC"])
            self.store_cuts(self.pst,
                            store_dir,
                            file_name,
                            format="SAC",
                            stream_save_mode="compact")
            #
            file_name = "_".join([self.eqtag, self.__stat, "BAIT_RAW.SAC"])
            self.store_cuts(self.rst,
                            store_dir,
                            file_name,
                            format="SAC",
                            stream_save_mode="compact")

        # =====================================================
        # 2) Run BaIt
        self._baitwrap_layer(**self.wd['pickerpar'])

    # MB: 08072019: at the moment, if indexed pick not present in BaIt
    #               object --> None, None is returned.
    def extract_pick(self, pick_idx, picker_method, compact=True):
        """
        This method extract the selected index of true pick from
        the object pick list (if present).
        User can specify if the method should return the BK or AIC pick
        of the True pick attributes.

        *** NB: indexing start from 0
        """
        picklist = []
        if self.itbk:
            # --- Extract the pick_idx TruePick
            # MB: IndexError is catched internally by BAIT
            #     If it happen --> return None, None
            picklist = self.itbk.extract_true_pick(idx=pick_idx,
                                                   picker=picker_method,
                                                   compact_format=compact)
        else:
            logger.warning("BaIt instance not defined yet ..." +
                           "run first the 'pickMe' method and retry")
            return None, None
        #
        return picklist

    def get_bait(self):
        """ Return the reference to private attribute BaIt object """
        if self.itbk:
            return self.itbk

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
