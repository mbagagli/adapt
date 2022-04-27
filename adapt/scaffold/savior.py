import logging
import yaml
import collections

# Quake
from adapt import __version__
from adapt import __author__
import adapt.database as QD

# ObsPy Pick class
from obspy.core.event.origin import Pick
from obspy.core.event.base import WaveformStreamID
from obspy.core.event.base import CreationInfo
from obspy.core.event.base import QuantityError

# ObsPy Inventory class
from obspy.core.inventory import Inventory
from obspy.core.inventory import Network
from obspy.core.inventory import Station
from obspy.core.inventory import Channel
from obspy.core.inventory import Site


logger = logging.getLogger(__name__)


class PickContainer2ObsPyPick(object):
    def __init__(self, pick_dict, store_debug=True):
        """ This class will store properly a adapt.database.PickContainer
            class into a list of obspy.core.event.origin.Pick ones.
        """
        self.pd = pick_dict
        self.pd_attr = pick_dict.__dict__
        self.out_pick_list = []
        self.debug_dict = collections.defaultdict(dict)
        self.store_debug = store_debug

    def _create_waveform_id(self, **kwargs):
        """
        network_code (str) Network code.
        station_code (str) Station code.
        location_code (str, optional) Location code.
        channel_code (str, optional) Channel code.
        resource_uri (ResourceIdentifier, optional) Resource identifier
                                                    for the waveform
                                                    stream.
        seed_string (str, optional) Provides an alternative
                                    initialization way by passing a SEED
                                    waveform string in the form
                                    network.station.location.channel,
                                    (e.g. BW.FUR..EHZ, which will be
                                    used to populate the
                                    WaveformStreamID’s attributes.
                                    It will only be used if the network,
                                    station, location and channel
                                    keyword argument are ALL None.
        """
        pass

    def store(self):
        """ Main method to convert and append the converted picks """
        for _sidx, (_stat, _phasedict) in enumerate(self.pd.items()):
            for _phasename, _phaselist in _phasedict.items():
                for _pidx, _singlepickdict in enumerate(_phaselist):

                    # ==================================== Object Attr
                    # resource_id (ResourceIdentifier)
                    # force_resource_id (bool, optional)
                    # time (UTCDateTime)
                    # time_errors (QuantityError)
                    # waveform_id (WaveformStreamID)
                    # method_id (ResourceIdentifier, optional)
                    # slowness_method_id (ResourceIdentifier, optional)
                    # onset (str, optional)
                    # phase_hint (str, optional)
                    # polarity (str, optional)
                    # evaluation_mode (str, optional)
                    # evaluation_status (str, optional)
                    # comments (list of Comment, optional)
                    # creation_info (CreationInfo, optional)
                    # filter_id (ResourceIdentifier, optional)
                    # horizontal_slowness (float, optional)
                    # horizontal_slowness_errors (QuantityError)
                    # backazimuth (float, optional)
                    # backazimuth_errors (QuantityError)

                    # 'polarity',       # (adapt.picks.polarity.Polarizer/None)
                    # 'onset',          # (str/None)
                    # 'weight',         # (adapt.picks.weight.Weighter/None)
                    # 'pickerror',      # (float/None)
                    # 'pickclass',      # (int/None)
                    # 'pickpolar',      # (str/None)
                    # 'evaluate',       # (bool/None)
                    # 'evaluate_obj',   # (adapt.picks.evaluation.Gandalf/None)
                    # 'timeUTC_pick',   # (UTCDateTime/None)
                    # 'timeUTC_early',  # (UTCDateTime/None)
                    # 'timeUTC_late',   # (UTCDateTime/None)

                    # ==================================== Create risky
                    try:
                        _lowunc = (_singlepickdict["timeUTC_early"] -
                                   _singlepickdict["timeUTC_pick"])
                    except TypeError:
                        _lowunc = None

                    try:
                        _uppunc = (_singlepickdict["timeUTC_late"] -
                                   _singlepickdict["timeUTC_pick"])
                    except TypeError:
                        _uppunc = None
                    #
                    if _singlepickdict["pickpolar"]:
                        if _singlepickdict["pickpolar"] == "D":
                            _pol = "negative"
                        elif _singlepickdict["pickpolar"] == "U":
                            _pol = "positive"
                        else:
                            _pol = "undecidable"
                    else:
                        _pol = None

                    # ==================================== Populate PICK
                    _tmp_pick = Pick(
                                  time=_singlepickdict["timeUTC_pick"],
                                  time_errors=QuantityError(
                                      # confidence_level (float),
                                      uncertainty=_singlepickdict["pickerror"],
                                      lower_uncertainty=_lowunc,
                                      upper_uncertainty=_uppunc,
                                      ),
                                  phase_hint="_".join([_phasename, str(_pidx)]),
                                  polarity=_pol,
                                  waveform_id=WaveformStreamID(
                                      station_code=_stat
                                      # network_code (str)
                                      # location_code (str, optional)
                                      # channel_code (str, optional)
                                      ),
                                  creation_info=CreationInfo(
                                      author="QUAKE",
                                      version=__version__)
                                     )

                    self.out_pick_list.append(_tmp_pick)

            ### MISSING THE DEBUG ATTRIBUTES STORING ###

        # Creating the debug dict

        # with open('data.yml', 'w') as outfile:
        #     yaml.dump(data, outfile, default_flow_style=False)


    def get_output(self):
        if self.out_pick_list:
            return self.out_pick_list
        else:
            logger.warning("The output PICK LIST is not created yet ...")



class StationContainer2StationXML(object):
    """ This class should store properly a collection of QuakePick """
    def __init__(self, stat_container, store_debug=True):
        self.sc = stat_container
        self.sc_attr = stat_container.__dict__
        self.out_inventory = None
        self.store_debug = store_debug

    def store(self):
        # ================  Create Tricky Field
        pass

    def get_output(self):
        if self.out_inventory:
            return self.out_inventory
        else:
            logger.warning("The output INVENTORY is not created yet ...")


# ========================================= TIPS and HELPS
# =========================================================

# import obspy
# from obspy.core.inventory import Inventory, Network, Station, Channel, Site
# from obspy.clients.nrl import NRL


# # We'll first create all the various objects. These strongly follow the
# # hierarchy of StationXML files.
# inv = Inventory(
#     # We'll add networks later.
#     networks=[],
#     # The source should be the id whoever create the file.
#     source="ObsPy-Tutorial")

# net = Network(
#     # This is the network code according to the SEED standard.
#     code="XX",
#     # A list of stations. We'll add one later.
#     stations=[],
#     description="A test stations.",
#     # Start-and end dates are optional.
#     start_date=obspy.UTCDateTime(2016, 1, 2))

# sta = Station(
#     # This is the station code according to the SEED standard.
#     code="ABC",
#     latitude=1.0,
#     longitude=2.0,
#     elevation=345.0,
#     creation_date=obspy.UTCDateTime(2016, 1, 2),
#     site=Site(name="First station"))

# cha = Channel(
#     # This is the channel code according to the SEED standard.
#     code="HHZ",
#     # This is the location code according to the SEED standard.
#     location_code="",
#     # Note that these coordinates can differ from the station coordinates.
#     latitude=1.0,
#     longitude=2.0,
#     elevation=345.0,
#     depth=10.0,
#     azimuth=0.0,
#     dip=-90.0,
#     sample_rate=200)

# # By default this accesses the NRL online. Offline copies of the NRL can
# # also be used instead
# nrl = NRL()
# # The contents of the NRL can be explored interactively in a Python prompt,
# # see API documentation of NRL submodule:
# # http://docs.obspy.org/packages/obspy.clients.nrl.html
# # Here we assume that the end point of data logger and sensor are already
# # known:
# response = nrl.get_response( # doctest: +SKIP
#     sensor_keys=['Streckeisen', 'STS-1', '360 seconds'],
#     datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])


# # Now tie it all together.
# cha.response = response
# sta.channels.append(cha)
# net.stations.append(sta)
# inv.networks.append(net)

# # And finally write it to a StationXML file. We also force a validation against
# # the StationXML schema to ensure it produces a valid StationXML file.
# #
# # Note that it is also possible to serialize to any of the other inventory
# # output formats ObsPy supports.
# inv.write("station.xml", format="stationxml", validate=True)





