import logging
#
from obspy import Stream
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
from obspy.clients.fdsn.mass_downloader import RectangularDomain
from obspy.clients.fdsn.mass_downloader import CircularDomain
from obspy.clients.fdsn.mass_downloader import Restrictions
from obspy.clients.fdsn.mass_downloader import MassDownloader
#
from obspy.io.mseed import InternalMSEEDError
from obspy.clients.filesystem.sds import Client as SDSCLIENT
#
import adapt.errors as QE


logger = logging.getLogger(__name__)

"""ADAPT downloader module

In this module are contained all the classes and functions related to
data download and their relative storing.

"""


class TheGathering(object):
    """Take care of downloading and storing the data.

    Wrapper over the `MassDownloader` ObsPy function
    """
    def __init__(self,
                 domain_type='circular',
                 domain_dict={},
                 restrictions_dict={},
                 providers_list=[]):
        # MB: defining attributes
        self.mdl = MassDownloader(providers=providers_list)
        # Domain
        if domain_type.lower() in ('circular', 'circ', 'c'):
            self.domain = CircularDomain(**domain_dict)
        elif domain_type.lower() in ('rectangular', 'rect', 'r'):
            self.domain = RectangularDomain(**domain_dict)
        else:
            raise QE.InvalidParameter()
        # Restrictions
        self.restrictions = Restrictions(**restrictions_dict)

    def collect(self,
                mseed_storage_path="waveforms",
                stationxml_storage_path="stations",
                convert_to=None):
        """Download data from provided fdsn client

        User can specify the destination folder for both miniseed and
        stations.

        """
        self.mdl.download(self.domain,
                          self.restrictions,
                          mseed_storage=mseed_storage_path,
                          stationxml_storage=stationxml_storage_path)
        if (convert_to and
           isinstance(convert_to, 'str') and
           convert_to in ('SAC', 'GSE2')):
            #
            logging.info('Converting Downloaded data into %s format' %
                         convert_to)
            # MB: Not yet implemented


class TheGatheringLocal(object):
    """ Take care of downloading and storing the data with a local
        SDS databates

    Args:
        server_path (str)
        stations_list (list/tuple)
        timecut_start (UTCDateTime obj.)
        timecut_end (UTCDateTime obj.)
    """
    def __init__(self,
                 server_path,
                 stations_list=None,
                 timecut_start=None,
                 timecut_end=None):
        self.storage = {}
        if not server_path:
            raise TypeError("I need a local or online serverpath!")
        else:
            self.server = server_path

    def collect(self,
                mseed_storage_path="waveforms",
                stationxml_storage_path="stations",
                convert_to=None):
        """Download data from provided fdsn client

        User can specify the destination folder for both miniseed and
        stations.

        """
        self.mdl.download(self.domain,
                          self.restrictions,
                          mseed_storage=mseed_storage_path,
                          stationxml_storage=stationxml_storage_path)
        if (convert_to and
           isinstance(convert_to, 'str') and
           convert_to in ('SAC', 'GSE2')):
            #
            logging.info('Converting Downloaded data into %s format' %
                         convert_to)
            # MB: Not yet implemented


# ======
def adapt_catalog_download(client_name, store=False, store_dir=".",
                           store_date=False, **kwargs):
    """ ADAPT utility for downloading catalogs from major agencies
    Retrieve event list from given client and given query

    Args:
        client_name (str): name of client

    Other Parameters:
        store (bool): if true, the downloaded catalog is also store as
            QUAKEML format
        **kwargs: _kwargs_ are used to specify the query parameter. For
            more informations check
            http://www.fdsn.org/webservices/fdsnws-event-1.2.pdf

    Returns:
        cat (obspy.Catalog): obspy catalog object
    """
    client = Client(client_name)
    try:
        print("... downloading (limit = %d)" % kwargs['limit'])
    except KeyError:
        print("... downloading")
    cat = client.get_events(**kwargs)

    # Fix SavingDate
    SavingDate = UTCDateTime()
    SavingString = ("%d%02d%02d_%02d%02d" % (
                            SavingDate.year, SavingDate.month, SavingDate.day,
                            SavingDate.hour, SavingDate.minute
                            )
                    )
    print("... storing  QUAKEML")
    if store and store_dir:
        if store_date:
            cat.write(store_dir+"/"+".".join(["ADAPT.Catalog", client_name,
                                              "Download",
                                              SavingString,
                                              "xml"]),
                      format="QUAKEML")
        else:
            cat.write(store_dir+"/"+".".join(["ADAPT.Catalog", client_name,
                                              "xml"]),
                      format="QUAKEML")

    #
    print("DONE")
    return cat


def adapt_inventory_download(client_name, store=False, store_dir=".",
                             store_date=False, **kwargs):
    """ ADAPT utility for downloading catalogs from major agencies
    Retrieve event list from given client and given query

    Args:
        client_name (str): name of client

    Other Parameters:
        store (bool): if true, the downloaded inventory is also store as
            QUAKEML format
        **kwargs: _kwargs_ are used to specify the query parameter. For
            more informations check
            http://www.fdsn.org/webservices/fdsnws-station-1.1.pdf

    Returns:
        cat (obspy.Catalog): obspy catalog object
    """
    client = Client(client_name)
    print("... downloading")
    inv = client.get_stations(**kwargs)

    # Fix SavingDate
    SavingDate = UTCDateTime()
    SavingString = ("%d%02d%02d_%02d%02d" % (
                            SavingDate.year, SavingDate.month, SavingDate.day,
                            SavingDate.hour, SavingDate.minute
                            )
                    )
    print("... storing  STATIONXML")
    if store and store_dir:
        if store_date:
            inv.write(store_dir+"/"+".".join(["ADAPT.Inventory", client_name,
                                              "Download",
                                              SavingString,
                                              "xml"]),
                      format="STATIONXML")
        else:
            inv.write(store_dir+"/"+".".join(["ADAPT.Inventory", client_name,
                                              "xml"]),
                      format="STATIONXML")
    #
    print("DONE")
    return inv


def adapt_waveforms_download(client_name,
                             start_time,
                             end_time,
                             networks_list=[],
                             stations_list=[],
                             locations_list=[],
                             channel_list=[],
                             store=True,
                             store_dir="waveforms",
                             store_date=False,
                             store_format="SEED"):
    """ ADAPT utility for downloading waveforms from single client
    Retrieve all the waveforms for the given

    Args:
        client_name (str): name of client

    Other Parameters:
        store (bool): if true, the downloaded waveforms is also store as
            QUAKEML format
        **kwargs: _kwargs_ are used to specify the query parameter. For
            more informations check
            http://www.fdsn.org/webservices/fdsnws-station-1.1.pdf

    Returns:
        cat (obspy.Catalog): obspy catalog object
    """

    # === prepare parameters

    if isinstance(networks_list, str):
        networks_list = [networks_list, ]
    if isinstance(stations_list, str):
        stations_list = [stations_list, ]
    if isinstance(locations_list, str):
        locations_list = [locations_list, ]
    if isinstance(channel_list, str):
        channel_list = [channel_list, ]
    # ===========================

    client = Client(client_name)
    st = Stream()
    print("... downloading:")
    for nn in networks_list:
        for ss in stations_list:
            for lc in locations_list:
                for ch in channel_list:
                    print("NET: %s - STA: %s - LOC: %s - CHA: %s " %
                          (nn, ss, lc, ch))

                    try:
                        _st = client.get_waveforms(nn, ss, lc, ch,
                                                   start_time, end_time)
                        print("... got it ...")
                        st += _st
                    except:  # FDSNNoDataException
                        print("... no available data ...")

    # Fix SavingDate
    SavingDate = UTCDateTime()
    SavingString = ("%d%02d%02d_%02d%02d" % (
                            SavingDate.year, SavingDate.month, SavingDate.day,
                            SavingDate.hour, SavingDate.minute
                            )
                    )

    print("... storing  STREAM-SEED")
    if store and store_dir:
        if store_date:
            st.write(store_dir+"/"+".".join(["ADAPT.Stream", client_name,
                                             "Download",
                                             SavingString,
                                             "mseed"]),
                     format="MSEED")
        else:
            st.write(store_dir+"/"+".".join(["ADAPT.Stream", client_name,
                                             "mseed"]),
                     format="MSEED")
    #
    print("DONE")
    return st
