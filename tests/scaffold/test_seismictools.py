import pprint
import obspy
from adapt import forward as FWD
from adapt.database import PickContainer
from adapt.scaffold import seismictools as ST

TESTARR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]
RESPFILT = {'pre_filt': None,
            'water_level': 100,
             # Defaults from obspy
            'zero_mean': False,
            'taper': False,
            'taper_fraction': False}

def test_mad():
    """ Check stats MAD """
    errors = []
    if ST._simple_mad(TESTARR) != 3.0:
        errors.append("MAD statistics went wrong")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


# def test_station_mag():
#     """ Simple calculation tests """
#     st = obspy.read()
#     inv = obspy.read_inventory()
#     # eve = obspy.read_events()
#     # pd = PickContainer.create_empty_container()
#     # pd.addPick("RJOB", phsnm, **pckdct)
#     #
#     for tr in st:
#         tr.stats.network = "BW"
#         tr.stats.location = ""
#     import pdb; pdb.set_trace()
#     smg, amp, ampt, chan = ST._calc_station_Mlv(
#                                 st.copy(),
#                                 inv,
#                                 st[0].stats.starttime, st[0].stats.endtime,
#                                 5, #km
#                                 method="minmax",
#                                 output="VEL",
#                                 spaceunit="mm",
#                                 response_filt_parameters=RESPFILT)

