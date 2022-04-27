""" Storing all the functions applied to the ADAPT CLEANER class!!

    Every function mus return a RESULTS dictionary containing the
    pair value of OUTNAME - OUTVALUE
"""

import os
import logging
import adapt.download as QD
from adapt.utils import calcEpiDist
import numpy as np

logger = logging.getLogger(__name__)


# ============================================ Private


def _calculate_epicentral_distance(ela, elo, sla, slo, outdist="km"):
    """ return the epicentral distance among event and station """
    return calcEpiDist(ela, elo, sla, slo, outdist=outdist)


# ============================================ Public

def ratio_SC3_picked_availability(event, pickdict, inventory,
                                  radius=80.0,
                                  ot_buffer_start=-10,
                                  ot_buffer_end=30,
                                  channel="*Z",
                                  epi_measure="km",  # 'km'/'m'
                                  debug_file=False):
    """ Define the ratio among the Picked by SC3 against the available
        SC3 dataset. It will calculate 2 ratio:
            - The one among epicenter and radius
            - The one from radius to the farthest picked SC3 station.

        Improved version that seeks prior for the MAX_EPI_SC3 and then
        checks the availability on SC3 dataset. ==> faster!
    """
    res_dict = {}

    # Event INFO
    evla = event.origins[0].latitude
    evlo = event.origins[0].longitude

    # Searching windows
    starttime = event.origins[0].time + ot_buffer_start  # sec
    endtime = event.origins[0].time + ot_buffer_end    # sec

    # Epicenter SC3
    sc3list = list(pickdict.keys())
    smaller_radius_sc3 = []
    larger_radius_sc3 = []
    max_epi_sc3 = 0.0
    for _sta in sc3list:
        epidist = _calculate_epicentral_distance(evla, evlo,
                                                 inventory[_sta]['lat'],
                                                 inventory[_sta]['lon'],
                                                 outdist=epi_measure)
        if epidist <= radius:
            smaller_radius_sc3.append((_sta, epidist))
        else:
            larger_radius_sc3.append((_sta, epidist))
        #
        if epidist > max_epi_sc3:
            max_epi_sc3 = epidist

    new_inventory = {_kk: _meta for _kk, _meta in inventory.items() if
                     _calculate_epicentral_distance(
                        evla, evlo, _meta['lat'], _meta['lon']) <= max_epi_sc3}

    # Avail. INFO
    if debug_file:
        eqid = event.resource_id.id
        dataDict, _ = QD.database_availability(
                                starttime, endtime, new_inventory,
                                channel=channel,
                                extracttofile=eqid+"_StationsAvailability.txt",
                                save_stream=False)
    else:
        dataDict, _ = QD.database_availability(
                                starttime, endtime, new_inventory,
                                channel=channel,
                                extracttofile=False,
                                save_stream=False)

    # Epicenter DB
    smaller_radius_db = []
    larger_radius_db = []
    for _sta, _meta in dataDict.items():
        epidist = _calculate_epicentral_distance(evla, evlo,
                                                 _meta['lat'], _meta['lon'])
        if epidist <= radius:
            smaller_radius_db.append((_sta, epidist))
        elif radius < epidist <= max_epi_sc3:
            larger_radius_db.append((_sta, epidist))
        else:
            # Station too far from MAX_RADIUS
            continue

    # Output DICT
    res_dict["stat_close_sc3"] = len(smaller_radius_sc3)
    res_dict["stat_close_db"] = len(smaller_radius_db)
    try:
        res_dict["close_ratio"] = len(smaller_radius_sc3) / len(smaller_radius_db)
    except ZeroDivisionError:
        res_dict["close_ratio"] = 9999.9

    #
    res_dict["stat_far_sc3"] = len(larger_radius_sc3)
    res_dict["stat_far_db"] = len(larger_radius_db)
    try:
        res_dict["far_ratio"] = len(larger_radius_sc3) / len(larger_radius_db)
    except ZeroDivisionError:
        res_dict["far_ratio"] = 9999.9

    #
    res_dict["radius"] = radius
    res_dict["max_epi_dist_sc3"] = max_epi_sc3

    # Output DEBUG
    if debug_file:
        with open(event.resource_id.id+".availability", "w") as IN:
            IN.write(("%s, %5.4f, %5.4f" + os.linesep) % (
                                        event.resource_id.id,
                                        res_dict["close_ratio"],
                                        res_dict["far_ratio"]))
            IN.write(("%s - close, %d, %d" + os.linesep) % (
                                          event.resource_id.id,
                                          len(smaller_radius_sc3),
                                          len(smaller_radius_db)))
            IN.write(("%s - far, %d, %d" + os.linesep) % (
                                        event.resource_id.id,
                                        len(larger_radius_sc3),
                                        len(larger_radius_db)))
            IN.write(("%5.2f - %5.2f" + os.linesep) % (radius, max_epi_sc3))
    return res_dict


def epicentral_distance_statistics(event, pickdict, inventory,
                                   percentile=75.0,
                                   epi_measure="km"):
    """ Calculate and analyse the statistical distribution of the
        epicentral distances. Will return:
            - mean
            - median
            - percentile (user defined)
            - n.stats >= percentile
    """
    res_dict = {}

    # Event INFO
    evla = event.origins[0].latitude
    evlo = event.origins[0].longitude

    # Epicentral Dist SC3
    epilist = [_calculate_epicentral_distance(evla, evlo,
                                              inventory[_sta]['lat'],
                                              inventory[_sta]['lon'],
                                              outdist=epi_measure)
               for _sta in pickdict.keys()]

    # Statistics
    epiarr = np.array(epilist)
    percentile_val = np.percentile(epiarr, percentile)
    res_dict['epidist_mean'] = np.mean(epiarr)
    res_dict['epidist_median'] = np.median(epiarr)
    res_dict['epidist_' + str(percentile) + '_percentile'] = percentile_val
    res_dict['epidist_statcount_over_percentile'] = np.sum(
                                                      epiarr >= percentile_val)
    return res_dict
