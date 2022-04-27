import os
import numpy as np
from obspy import read, read_inventory, Stream, Inventory, Catalog
from obspy.geodetics.base import gps2dist_azimuth
#
from adapt.processing import processStream
import adapt.utils as QU
import adapt.errors as QE
import adapt.plot as QP
import logging
import matplotlib.pyplot as plt
from adapt.scaffold.statistics import AdaptDatabase
# For Magnitudes
from obspy.signal.invsim import simulate_seismometer


logger = logging.getLogger(__name__)

MT = 1000.0
KM = 0.001


def _simple_mad(inlist):
    """Median Absolute Deviation"""
    if isinstance(inlist, (list, tuple, np.ndarray)):
        return np.median(np.abs(inlist - np.median(inlist)))
    else:
        raise QE.InvalidType()


def _is_phase_valid(pickdict):
    """Input only the single dictionary defining a pick """

    if pickdict["weight"]:
        # If the key exists
        _triage = pickdict["weight"].get_triage_dict()
    else:
        raise QE.MissingAttribute("WEIGHT object is missing in PickDict !!!")

    if (len(_triage['valid_obs']) >= 6 or
       (len(_triage['valid_obs']) == 4 and len(_triage['pickers_involved']) == 4) or
       (len(_triage['valid_obs']) == 5 and len(_triage['pickers_involved']) == 4)):
        return True
    else:
        return False


def _sorting_station_clockwise(evlon, evlat, coord_tuples, debug_plot=True):
    """ Sorting based on North ax passing throught event an NorthPole

        Args:
            evlon (float): earthquake's longitude
            evlat (float): earthquake's latitude
            coord_tuples (list): it's a list of tuples! (lon, lat)

        Returns
            (list): input list counter-clock ordered

    """
    worklst = []
    for ss in coord_tuples:
        slon, slat = ss[0], ss[1]
        az, _ = _calculate_interstat_angle_dist(evlon, evlat,
                                                evlon, 90,    # NorthPole
                                                slon, slat)
        worklst.append((az, ss))
    # sorti dal piu' piccolo al piu' grande [(az, (lon,lat)), ...]
    worklst.sort(key=lambda x: x[0])
    return worklst


def _calculate_interstat_angle_dist(evlon, evlat,
                                    s1lon, s1lat,
                                    s2lon, s2lat,
                                    debug_plot=False):
    """Calculate azimutal gap of 2 stations realted to an event

        Args:
            evlon (float)
            evlat (float)
            s1lon (float)
            s1lat (float)
            s2lon (float)
            s2lat (float)
        Returns:
            azGap (float)

    """
    # gps2dist_azimuth(lat1, lon1, lat2, lon2, a=WGS84_A, f=WGS84_F)
    _, s1az, _ = gps2dist_azimuth(evlat, evlon, s1lat, s1lon)
    _, s2az, _ = gps2dist_azimuth(evlat, evlon, s2lat, s2lon)

    a, b = np.max((s1az, s2az)), np.min((s1az, s2az))
    delaz = a - b
    bakdelaz = 360.0 - delaz

    # # GRACIELA
    # if delaz > 180.0:
    #     azGap = 360.0 - delaz
    # else:
    #     azGap = delaz
    # bazGap = 360.0 - delaz

    if debug_plot:
        mpax, _ = QP.plot_map()
        mpax.set_title("azGap: %6.2f - bazGap: %6.2f" % (delaz, bakdelaz))
        QP.plot_feature_xy([s1lon, s2lon], [s1lat, s2lat], inax=mpax,
                           marker="^", color="cyan",
                           ms=5, mew=0.7, mec="k",
                           linestyle='None')
        QP.plot_feature_xy([evlon, ], [evlat, ], inax=mpax,
                           marker="*", color="red",
                           ms=9, mew=0.7, mec="k",
                           linestyle='None')
        plt.show()

    return delaz, bakdelaz


def event_azimuthal_coverage_gap(event,
                                 statcontainer,
                                 pickcontainer,
                                 phase_check="P1",
                                 indexnum=0,
                                 debug_plot=False):
    """Calculate first and second order azimuthal GAP of a picked event

    This function keep it simple, pick dict (containing interested phase
    to check). It will internally evaluate the phase if it's a valid
    phase (4 picker picked or )

    Args:
        ev (obspy.Event): the class containing informations about
            the earthquakes
        pick_dict (database.PickContainer): the pick dict related to the
            given event containing the interested final picks
        stat_dict (database.StatContainerontainer)

    Returns:
        asd

    Note:
        It will work only for _P1_ phase picks.
    """
    check_stat_list = []
    evlon = event.origins[0].longitude
    evlat = event.origins[0].latitude
    #
    if isinstance(statcontainer, str):
        statcont = QU.loadPickleObj(statcontainer)
    else:
        statcont = statcontainer

    # Query Station on pick dict
    for _ss in pickcontainer.keys():
        matchList = pickcontainer.getMatchingPick(_ss,
                                                  phase_check,
                                                  indexnum=indexnum)
        checkRes = _is_phase_valid(matchList[0][1])
        if checkRes:
            check_stat_list.append((statcont[_ss]['lon'],
                                    statcont[_ss]['lat']))

    # Check before going on
    if len(check_stat_list) == 0:
        raise QE.MissingVariable("No valid phases found!")

    # sort counter-clock
    coordListCC = _sorting_station_clockwise(evlon, evlat, check_stat_list)

    # loop over and calculate GAP:
    azdist = []
    for _xx in range(0, len(coordListCC)):
        if _xx == len(coordListCC)-1:
            northaz1, (s1lon, s1lat) = coordListCC[_xx]  # --> closing the circle
            northaz2, (s2lon, s2lat) = coordListCC[0]
            az = 360 - northaz1 + northaz2
        else:
            northaz1, (s1lon, s1lat) = coordListCC[_xx]
            northaz2, (s2lon, s2lat) = coordListCC[_xx+1]
            #
            az, _ = _calculate_interstat_angle_dist(evlon, evlat,
                                                    s1lon, s1lat,
                                                    s2lon, s2lat,
                                                    debug_plot=False)
        azdist.append((_xx, az, (s1lon, s1lat, s2lon, s2lat)))

    azdist.sort(key=lambda x: x[1], reverse=True)
    gapFirstAZ = azdist[0]

    ''' @develop
    To develop the 2*nd order GAP, try x-1 e x+1   x e x+1
    '''

    # ... eventually plot
    if debug_plot:
        mpax, _ = QP.plot_map()
        mpax.set_title("azGap: %6.2f" % gapFirstAZ[1])
        QP.plot_feature_xy([s[0] for s in check_stat_list],
                           [s[1] for s in check_stat_list],
                           inax=mpax,
                           marker="^", color="cyan",
                           ms=5, mew=0.7, mec="k",
                           linestyle='None')
        QP.plot_feature_xy([evlon, ], [evlat, ], inax=mpax,
                           marker="*", color="red",
                           ms=9, mew=0.7, mec="k",
                           linestyle='None')
        QP.plot_feature_xy([gapFirstAZ[2][0], gapFirstAZ[2][2]],
                           [gapFirstAZ[2][1], gapFirstAZ[2][3]], inax=mpax,
                           marker="^", color="yellow",
                           ms=5, mew=0.7, mec="k",
                           linestyle='None')
        plt.show()
    # (_xx, az, (s1lon, s1lat, s2lon, s2lat))
    return gapFirstAZ, len(coordListCC)


def event_local_magnitude(event,
                          inventory,
                          pickcontainer,
                          path_to_waveforms,
                          phase_check="P1",
                          phase_pick_idx=0,
                          magnitude_scale="Mlv",
                          station_amp_evaluation="p2p",
                          station_correction="DISP",
                          station_correction_unit="mm",
                          percentile_reject=95,
                          epicentral_threshold=400,  # km
                          max_time_window=150.0,
                          winfunctiondict={},
                          response_filter_dict={},
                          debug_plot=False):
    """Calculate local magnitude Mlv as referred in SeisComp3

    This function keep it simple, pick dict (containing interested phase
    to check). It will internally evaluate the phase if it's a valid
    phase (4 picker picked or ). Input station names must correspond to
    the ones present in the inventory in input!

    Args:
        event (obspy.Event): the class containing informations about
            the earthquakes
        statcontainer (database.StatContainer)
        pickcontainer (database.PickContainer): the pick dict related to the
            given event containing the interested final picks
        path_to_waveforms (str): a dir path that contains the waveforms
            cuts. At least the station and the channel must be specified
            like i.e *.MYSTAT.HHZ.*

    Returns:
        event (obspy.Event): return the input events with new magnitude
            appended to existent mag-list.

    Note:
        It will work only for _P1_ phase picks.

    """
    evid = event.resource_id.id
    evlo = event.origins[0].longitude
    evla = event.origins[0].latitude
    evtm = event.origins[0].time
    #
    pd_stat = pickcontainer.getStats()
    result_dict = {}

    # Loop
    stations_mag = []
    statmag_db = AdaptDatabase(tag=("Adapt_StationMagnitudes.%s" % evid))

    # Logging
    logger.debug(
        os.linesep.join(["EventMagnitude calculation for: %s -Scale: %s",
                         "Path to waveforms: %s -",
                         "PhaseName: %s - PhaseIndex: %d",
                         "Stations Correction: %s - Unit: %s - Method: %s"]) %
                       (evid, magnitude_scale, path_to_waveforms, phase_check,
                        phase_pick_idx, station_correction,
                        station_correction_unit, station_amp_evaluation)
                       )

    for ss in pd_stat:
        ssdict = {}

        try:
            VALIDPHASE = _is_phase_valid(
                        pickcontainer[ss][phase_check][phase_pick_idx])
        except QE.MissingAttribute:
            # In case THe WEIGHT object (from ADAPT) is missing, we take
            # for granted that the phase is VALID! (i.e. if loaded from CNV)
            VALIDPHASE = True

        if (phase_check in pickcontainer.getStatPick(ss) and
           pickcontainer[ss][phase_check][phase_pick_idx]['timeUTC_pick'] and
           VALIDPHASE):

            # 1. Get station + Calculate EPIDIST
            try:
                netobj = inventory.select(station=ss)[0]
            except IndexError:
                logger.error("!!! Station %s Missing in Inventory !!!" % ss)
                continue

            # 2. ======================================   CheckNetwork
            # if netobj._code.upper() == "CR":  # ... in ("SL", "CR"):
            #     logger.error("!!! Station belonging to CROATIA network, "
            #                  "skipping !!!")
            #     continue
            # ==========================================================

            # Calculate EPIDIST
            statobj = netobj[0]
            epidist_km = QU.calcEpiDist(
                                evla, evlo,
                                statobj.latitude,
                                statobj.longitude,
                                outdist='km')

            logger.info("EVENT:  %s  -  STAT:  %s.%s (%6.2f) --> %s found --> calculate Mlv" %
                        (evid, netobj.code, ss, epidist_km, phase_check))

            # 2. Read Trace
            st = read(path_to_waveforms + "/*." + ss + ".*")

            # 3. Extract pick and define window for station-mag
            picktime = (pickcontainer[ss][phase_check]
                                     [phase_pick_idx]["timeUTC_pick"])
            startt_adapt, endt_adapt = __time_window_sc3(
                                            evtm, epidist_km, picktime,
                                            max_time_window=max_time_window)

            # 4. Calc Mag
            if magnitude_scale.lower() == "mlv":
                # 4.b Calc Stat Mag -- ADAPT
                try:
                    smg, amp, ampt, chan = _calc_station_Mlv(
                                                st.copy(),
                                                inventory,
                                                startt_adapt, endt_adapt,
                                                epidist_km,
                                                method=station_amp_evaluation,
                                                output=station_correction,
                                                spaceunit=station_correction_unit,
                                                response_filt_parameters=response_filter_dict)

                except QE.MissingAttribute:
                    smg, amp, ampt, chan = None, None, None, None

            else:
                raise ValueError("At the moment only Mlv is supported!")

            # ========== Populate + Append to Station Dict
            ssdict['EVID'] = evid
            ssdict["NETWORK"] = netobj.code
            ssdict["STAT"] = ss
            ssdict["CHANNEL"] = chan
            ssdict["EPIDIST"] = epidist_km
            ssdict["EPITHR"] = epicentral_threshold
            #
            if smg:
                stations_mag.append((ss, epidist_km, smg))
                ssdict["ADAPT_MAG"] = smg
                ssdict["ADAPT_AMP"] = amp
                ssdict["ADAPT_AMPT"] = ("%4d-%02d-%02d %02d:%02d:%02d.%06d" % (
                                         ampt.year, ampt.month, ampt.day,
                                         ampt.hour, ampt.minute, ampt.second,
                                         ampt.microsecond))
            #
            statmag_db._append_record(ssdict)

    # ====================  Store StationMag Database
    if not statmag_db.is_empty():
        statmag_db.sort_dataframe(['EVID', 'EPIDIST'], inplace=True)
        statmag_db.export_dataframe(outfile=evid+"_stations_magnitude.csv",
                                    column_caps=True, floatformat="%.3f")
    else:
        logger.error("!!! Event %s has NO VALID STATIONS !!!" % evid)
        # Store anyway and exit function with All None/Empty
        # Populate result dict
        result_dict["EVID"] = evid
        result_dict["LON"] = evlo
        result_dict["LAT"] = evla
        result_dict["OT"] = evtm
        result_dict["TOT_OBS"] = 0
        result_dict["WORK_OBS"] = 0
        result_dict["EPI_THR"] = epicentral_threshold
        result_dict["PERCENTILE"] = percentile_reject
        result_dict["PERCENTILE_VAL"] = None
        result_dict["MEAN"] = None
        result_dict["MEDIAN"] = None
        result_dict["STD"] = None
        result_dict["MAD"] = None
        #
        return None, None, None, None, result_dict

    # ====================  Network Magnitude
    if len(stations_mag) <= 3:
        logger.warning("The event has 3 or less station magnitudes -> %d" %
                       len(stations_mag))
    mn, medn, std, mad, nobs_tot, nobs_work, used_sta, perc_val = _calc_network_magnitude(
                                        stations_mag,
                                        epi_thr=epicentral_threshold,
                                        reject_bound=percentile_reject)

    # "RESID_MEDIAN": []}
    # "EVMAG_MEDIAN": [],
    isused_col = {'STAT': [],
                  "ISUSED": [],
                  "EVMAG_MEAN": [],
                  "EVMAG_MEDIAN": []}

    for xxx in stations_mag:
        # stations_mag = [(name, epi, magval),]
        isused_col['STAT'].append(xxx[0])
        isused_col['EVMAG_MEAN'].append(mn)
        isused_col['EVMAG_MEDIAN'].append(medn)
        if xxx[0] in used_sta:
            isused_col['ISUSED'].append(1)
        else:
            isused_col['ISUSED'].append(0)

    # -- Overriting the previously stored file after col-update, super ugly!!!
    statmag_db.expand_columns(isused_col, "STAT")
    statmag_db.export_dataframe(outfile=evid+"_stations_magnitude.csv",
                                column_caps=True, floatformat="%.3f")

    # Populate result dict
    result_dict["EVID"] = evid
    result_dict["LON"] = evlo
    result_dict["LAT"] = evla
    result_dict["OT"] = evtm
    result_dict["TOT_OBS"] = nobs_tot
    result_dict["WORK_OBS"] = nobs_work
    result_dict["EPI_THR"] = epicentral_threshold
    result_dict["PERCENTILE"] = percentile_reject
    result_dict["PERCENTILE_VAL"] = perc_val
    result_dict["MEAN"] = mn
    result_dict["MEDIAN"] = medn
    result_dict["STD"] = std
    result_dict["MAD"] = mad
    #
    return mn, medn, std, mad, result_dict


def _calc_station_Mlv(opst, opinv, searchwin_start, searchwin_end, epidist,
                      output="DISP", spaceunit='m', method='p2p',
                      response_filt_parameters={}, debug_plot=False):
    """Calculate the Mlv over a Vertical-Components - SeisComp3 like

    In SeisComP3 a modified local magnitude MLv is determined by
    simulation of a Wood-Anderson instrument and then measuring the
    amplitude in a 150 s time window on the vertical component of
    station with distances smaller than 8°.

    mag = \log10(A) - (-2.9) = \log10(A) + 3

    which is according to the original Richter (1935) formula if
    the amplitude is measured in millimeters.

    Inputs:
        wave (obspy.core.Trace): object of a vertical component (*Z)
        inv (obspy.core.Inventory): object containing stat-channel
            instrument response.
        displ (str): could either be 'm' (meter), 'cm' (centimeter),
            'mm' (millimeter). Not cas-sensitive

    Output:
        Mlv: magnitude value

    NB: Trace must contains Z-channel and that all of them are processed

    """

    st = opst.select(channel="*Z")
    if not st:
        raise QE.MissingAttribute("Input stream must have a Z channel. "
                                  "Given --> %r" %
                                  [tt.stats.channel for tt in opst])

    tr = st[0]
    traceChannel = tr.stats.channel
    # Deafult AlpArray STS-2 response
    AAHHR = opinv.select(station="A291A", channel="HHZ")[0][0][0].response

    # ----------------------  Pre-Process
    tr.detrend('demean')
    tr.detrend('simple')
    tr.taper(max_percentage=0.05, type='cosine')  # 'hann'
    # ====== BP
    tr.filter("bandpass",
              freqmin=1,
              freqmax=50,    # originally 30
              corners=4,     # originally 2
              zerophase=True)
    # ====== HP
    # tr.filter("highpass", freq=1, corners=4, zerophase=True)

    # ----------------------  Attach Response
    try:
        tr.attach_response(opinv)
    except ValueError:
        raise QE.MissingAttribute("No instrument response in INVENTORY for: "
                                  "%s.%s.%s" % (tr.stats.station,
                                                tr.stats.channel,
                                                tr.stats.location))

    # Additional check for location code in AlpArray stations (or if broadband)
    if tr.stats.network == "Z3" and tr.stats.location == "00":
        if tr.stats.channel[0:2].upper() == "HH":
            # ---- Use DEFAULT broadband response
            tr.stats.response = AAHHR
        else:
            raise QE.MissingAttribute("Errors for Z3.00 location codes!")

    # ----------------------  Remove response

    # ObsPy: Displacement (m) Velocity (m/s) Acceleration (m/s^2)
    logger.debug(("Stat: %s - RemoveResponse: %s - Unit: %s"
                  "FilterParam: %r" + os.linesep) %
                 (tr.stats.station, output.upper(), spaceunit,
                  response_filt_parameters))
    logger.debug("Stat: %s - RemoveResponse: %s - Unit: %s" %
                 (tr.stats.station, output.upper(), spaceunit))

    tr.remove_response(
        output=output.upper(),
        **response_filt_parameters)

    # === NEW --> The next switch is though for instruments reconding in M/s
    # ===         If some instruments record in nm/s we to first convert them into
    # ===         M/s, then, one could do the switch
    if tr.stats.network == "SL":
        tr.data = tr.data * 1e-9
    #
    if spaceunit.lower() == "nm":
        tr.data = tr.data * 1000000000.0
    elif spaceunit.lower() == "mm":
        tr.data = tr.data * 1000.0
    elif spaceunit.lower() == "m":
        pass
    else:
        logger.warning("Unit: %s not supported !! Back to METERs !!")

    # ----------------------  Calculate
    if searchwin_end >= tr.stats.endtime:
        logger.warning("Station %s has reached the end of time!" %
                       tr.stats.station)

    tr = _simulate_wood_anderson(tr)  # simulateWA
    a0corr = __calculate_correction_factor(epidist)

    if method.lower() == 'p2p':
        try:
            amp, tamp = _calculate_peak2peak_amplitude(
                                            tr, searchwin_start,
                                            searchwin_end,
                                            debug_plot=debug_plot)
        except ValueError:
            logger.error("No amplitudes for %s !!!" % tr.stats.station)
            return None, None, None, traceChannel

        except QE.InvalidVariable:
            logger.error("Empty Data-Array %s !!!" % tr.stats.station)
            return None, None, None, traceChannel

    elif method.lower() == 'maxabs':
        try:
            amp, tamp = _calculate_peak_amplitude(
                                            tr, searchwin_start,
                                            searchwin_end)
        except ValueError:
            logger.error("No amplitudes for %s !!!" % tr.stats.station)
            return None, None, None, traceChannel

        except QE.InvalidVariable:
            logger.error("Empty Data-Array %s !!!" % tr.stats.station)
            return None, None, None, traceChannel

    elif method.lower() == 'minmax':
        try:
            amp, tamp = _calculate_half_min_max_amplitude(
                                            tr, searchwin_start,
                                            searchwin_end)
        except ValueError:
            logger.error("No amplitudes for %s !!!" % tr.stats.station)
            return None, None, None, traceChannel

        except QE.InvalidVariable:
            logger.error("Empty Data-Array %s !!!" % tr.stats.station)
            return None, None, None, traceChannel

    else:
        raise QE.InvalidParameter("Method time could be either 'p2p' or 'maxabs' or 'minmax'")

    # 19.04.2021  (Some Z3 stations have tooo high sensitivity --> AMP == 0.0) #MB
    if np.isclose(0.0, amp):
        logger.warning("Final AMP equal to zero! Discarded")
        return None, 0.0, tamp, traceChannel
    else:
        # Go calculate
        amp *= 2     # compensate the lack of horizontal-measurement
        smg = __calc_Mlv(amp, a0corr)
        return smg, amp, tamp, traceChannel


def __calc_Mlv(amp, ampcorr):
    """ Simply calculate the local-magnitude """
    return np.log10(amp) - ampcorr


def __evaluate_noise(tr, startt, endt):
    """ Module needed to determine the noise level of the trace. (STD)
        That will be then removed from the trace
    """
    tr.slice(startt, endt)
    #
    sd = np.std(tr.data)
    minv = np.min(tr.data)
    maxv = np.max(tr.data)
    return sd, minv, maxv


def __time_window_sc3(origintime, epidist, picktime, max_time_window=150.0,
                      tpre_pick=5, epifun_m=0.335, epifun_k=15):
    """ simply define the time window search for amplitudes

    *** From SC3 analysis: **
    - It uses P-pick time as REFERENCE
    - It cuts 5 seconds befor pick --> (we use it also to analyze noise)
    - Window's length is related to epidist ina LINEAR way
    - MaxWindowTime_post_Ppick == 150 sec.

    FINAL RESULT of analysis (linear): 0.334 +14.9 --> maybe round 0.335 +15
    startt = (Ptime 5 )
    endt = Ptime + (epidist*0.335) + 15

    """
    # The OFFICIAL SC3 win-time function are:
    #   - tpre_pick = 5 sec
    #   - epifun_m = 0.335
    #   - epifun_k = 15

    startt = picktime - tpre_pick
    tpostpick = epidist*epifun_m + epifun_k
    if tpostpick >= max_time_window:
        endt = picktime + max_time_window
    else:
        endt = picktime + tpostpick
    #
    return startt, endt


def __calculate_correction_factor(epidist, magpoints=None):
    """  Calculate the amplitude corrections for Richter magnitude.

    The corrections are made only fot the log10(A0) value.
    No site effect accounted for

    - Default SC3
    module.trunk.global.MLv.logA0 = "0 -1.3;60 -2.8;400 -4.5;1000 -5.85"

    The logA0 configuration string consists of an arbitrary number of
    distance-value pairs separated by semicolons.
    The distance is in km and the value corresponds to the log10(A0)
    term above.

    Within each interval the values are computed by linear interpolation.
    E.g. for the above default specification, at a distance of 100 km the
        logA0 value would be ((-4.5)-(-2.8))*(100-60)/(400-60)-2.8 = -3.0 --
        in other words, at 100 km distance the magnitude would be -3)

        mag = \log10(A) - (-3) = \log10(A) + 3

        which is according to the original Richter (1935) formula if the
        amplitude is measured in millimeters. Note that the baseline
        for logA0 is millimeters for historical reasons, while internally
        in SeisComP 3 the Wood-Anderson amplitudes are measured and stored
        micrometers.

    Link(https://www.seiscomp.de/seiscomp3/doc/seattle/2013.149/apps/global_mlv.html)

    """
    A0CORRPOINTS = ((0.0, -1.3), (60.0, -2.8), (400.0, -4.5), (1000.0, -5.85))
    if not magpoints:
        magpoints = A0CORRPOINTS
    #
    magdists = [xx[0] for xx in magpoints]
    magvalues = [xx[1] for xx in magpoints]

    if epidist in magdists:
        # if epidist is equal to a point, return the related value
        return magvalues[magdists.index(epidist)]
    else:
        # ... else interpolate, find upper tuple and lower tuple
        upidx = [ii for ii, xx in enumerate(magdists) if epidist < xx][0]
        lpidx = upidx - 1
        #
        up, lp = magpoints[upidx], magpoints[lpidx]

        # (-4.5  -  (-2.8)  ) * (X-60)/(400-60) -2.8
        corrfact = (up[1] - lp[1]) * (epidist - lp[0]) / (up[0] - lp[0]) + lp[1]
    return corrfact


def _calc_network_magnitude(stamags, reject_bound=None, epi_thr=None):
    """ Calculating network magnitude.

    This function takes in inputs the list of stations magnitude.
    It will output (based on `method` parameter) the Network Magnitude.

    By default, the trimmed mean is calculated from the station
    magnitudes to form the network magnitude.

    Outliers below the

    Inputs:
        stamags (list): list ot tuples containing
            (statname, epidist, statmag)
        reject_bound (int, float): confidence interval (multiplier of std)
    Output:
        netMag (float): network/array magnitudes,
        nsta (int): number of stations used to calculate the magnitude

    """

    # -- Select by Epicenter Distance
    if epi_thr:
        # vals = [sm[2] for sm in stamags if sm[1] <= epi_thr]
        vals = [(sm[0], sm[2]) for sm in stamags if sm[1] <= epi_thr]
    else:
        # vals = [sm[2] for sm in stamags]
        vals = [(sm[0], sm[2]) for sm in stamags]
    logger.info("Calculating EventMagnitude: %d (epi.thr: %5.1f) --> %d" %
                (len(stamags), epi_thr, len(vals)))

    # -- Define Confidence Interval
    _only_val = [ii[1] for ii in vals]
    _only_sta = [ii[0] for ii in vals]
    if reject_bound and isinstance(reject_bound, (int, float)):
        # interval-of-confidence. Calculate threshold, query array
        testmean = np.mean(_only_val)
        teststd = np.std(_only_val)
        #
        high_thresh = testmean + (reject_bound * teststd)
        low_thresh = testmean - (reject_bound * teststd)
        newvals = [ii for ii in vals if (ii[1] >= low_thresh and ii[1] <= high_thresh)]
        #
        _only_newval = [ii[1] for ii in newvals]
        _only_newsta = [ii[0] for ii in newvals]
        mn = np.mean(_only_newval)
        medn = np.median(_only_newval)
        std = np.std(_only_newval)
        mad = _simple_mad(_only_newval)
        #
        return mn, medn, std, mad, len(vals), len(newvals), _only_newsta, high_thresh
    else:
        # No interval-of-confidence. Use all array
        mn = np.mean(_only_val)
        medn = np.median(_only_val)
        std = np.std(_only_val)
        mad = _simple_mad(_only_val)
        return mn, medn, std, mad, len(vals), len(vals), _only_sta, None


def _simulate_wood_anderson(optr, water_level=10):
    """ Simple as it it. We want to simulate a WoodAnderson """
    # Sensitivity is 2080 according to:
    # P. Bormann: New Manual of Seismological Observatory Practice
    # IASPEI Chapter 3, page 24
    #
    # (PITSA has 2800)
    #

    # Seiscomp3 (puoi pure invertire la parte immaginaria)
    PAZ_WA_SC3 = {
         'poles': [(-6.283185-4.712389j),
                   (-6.283185+4.712389j)],
         'zeros': [0+0j],
         'gain': 1.0,
         'sensitivity': 2800}

    # # ----- ObsPy
    # # Sensitivity is 2080 according to:
    # # P. Bormann: New Manual of Seismological Observatory Practice
    # # IASPEI Chapter 3, page 24
    # # (PITSA has 2800)
    # PAZ_WA_OBSPY = {
    #      'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],
    #      'zeros': [0+0j],
    #      'gain': 1.0,
    #      'sensitivity': 2080}

    # # Graciela Rojo (Bormann and Dewey, 2014)
    # PAZ_WA_GRACIELA = {
    #       'poles': [-5.49779 - 5.60886j,
    #                 -5.49779 + 5.60886j],
    #       'zeros': [0.0 + 0.0j, 0.0 + 0.0j],
    #       'gain': 1.0028,
    #       'sensitivity': 2080}

    optr.simulate(paz_simulate=PAZ_WA_SC3,
                  paz_remove=None,
                  water_level=10)
    return optr


def _calculate_peak2peak_amplitude(
                        intr, start_win, end_win, debug_plot=False):
    """ Trace calculation of peaktopeak. For explanation see graciela scripts

    Use trace.slice() method as you need to search and not to process

    # ============  Calculate Peak2Peak

    # Localmag can perform two different searches for the "peak" magnitude of
    # a given Wood-Anderson trace.
    # First, (provided the trace has no gaps and is not clipped), a search is
    # made for the largest absolute value within the search window.
    # This will be the largest zero-to-peak value, since the Wood-Anderson
    # trace always has zero mean in the program.

    # The second search is a "sliding window" search, in which the largest
    # swing, positive-to-negative or negative-to-positive, is measured within
    # a sliding window of N seconds duration.
    # The window slides along the full length of the search window.
    # The result is the largest peak-to-peak value for the trace.
    # Since the local magnitude formula is based on a zero-to-peak value,
    # one half of the peak-to-peak value is used for the magnitude calculation.
    # The intent of this sliding window search is to reduce the chances of
    # picking a one-sided glitch that could happen with the zero-to-peak search.

    """
    original_start = intr.stats.starttime
    original_end = intr.stats.endtime
    wt = intr.slice(start_win, end_win)

    # ==== Additional Check for EMPTY ARRAY
    if (np.max(np.abs(wt.data)) == 0.0 and np.min(np.abs(wt.data)) == 0.0
       or np.isnan(wt.data).all()):
        raise QE.InvalidVariable("Empty Data-Array ... bailing out")

    # Search A1
    A1 = np.max(np.abs(wt.data))
    A1_timeidx = np.where(np.logical_or(A1 == wt.data, -A1 == wt.data))[0][0]  #time index on the data_ampW
    A1_sign = np.sign(wt.data[A1_timeidx])
    if not A1_timeidx:
        raise ValueError("Missing time for Amplitude 1 ... bailing out")
    else:
        A1_timeutc = wt.stats.starttime + A1_timeidx*wt.stats.delta

    # Search A2 AFTER
    wt = intr.slice(A1_timeutc, original_end)
    try:
        after_zerocross = np.where(np.diff(np.sign(wt.data)))[0][2]  # 2nd zero-crossing vector (3rd element)
    except IndexError:
        # MB: mostlikely end-of-trace reached (higher amplitude really in the back)
        pass
    tmpvec = wt.data[0:after_zerocross]

    if A1_sign == -1:
        after_max_amp = np.max([ii for ii in tmpvec if ii >= 0])
    else:
        after_max_amp = np.min([ii for ii in tmpvec if ii <= 0])
    #
    A2_after_timeidx = np.where(after_max_amp == wt.data)[0][0]
    if not A2_after_timeidx:
        import pdb; pdb.set_trace()
        raise ValueError("Missing time for Amplitude 2 AFTER ... bailing out")
    else:
        A2_after = wt.data[A2_after_timeidx]
        A2_after_timeutc = wt.stats.starttime + A2_after_timeidx*wt.stats.delta

    # Search A2 BEFORE
    wt = intr.slice(original_start, A1_timeutc)
    wt.data = np.flipud(wt.data)   # giralo e lavoraci
    before_zerocross = np.where(np.diff(np.sign(wt.data)))[0][2]  # 2nd zero-crossing vector (3rd element)
    tmpvec = wt.data[0:before_zerocross]

    if A1_sign == -1:
        before_max_amp = np.max([ii for ii in tmpvec if ii >= 0])
    else:
        before_max_amp = np.min([ii for ii in tmpvec if ii <= 0])
    #
    A2_before_timeidx = np.where(before_max_amp == wt.data)[0][0]
    if not A2_before_timeidx:
        import pdb; pdb.set_trace()
        raise ValueError("Missing time for Amplitude 2 BEFORE ... bailing out")
    else:
        A2_before = wt.data[A2_before_timeidx]
        A2_before_timeutc = wt.stats.endtime - A2_before_timeidx*wt.stats.delta
    # go back to right order
    wt.data = np.flipud(wt.data)

    # Define final A2
    A2 = np.max([np.abs(A2_before), np.abs(A2_after)])

    # Define half peak-to-peak amplitude
    A = (A1+A2)/2.0

    # Return the earliest arrival of the involved amplitudes
    if np.abs(A2_after_timeutc) > np.abs(A2_before_timeutc):
        At = A1_timeutc
    else:
        At = A2_before_timeutc

    if debug_plot:
        intr.stats.timemarks = [
                (A1_timeutc,
                    {'color': 'g', 'markeredgewidth': 5, 'markersize': 10}),
                (A2_before_timeutc,
                    {'color': 'b', 'markeredgewidth': 5, 'markersize': 10}),
                (A2_after_timeutc,
                    {'color': 'r', 'markeredgewidth': 5, 'markersize': 10}),
                ]
        intr.plot(plot_time_marks=True)

    return A, At


def _calculate_peak_amplitude(wt, start_win, end_win, debug_plot=False):
    """ Trace calculation of peaktopeak. For explanation see graciela scripts

    Use trace.slice() method as you need to search and not to process

    """
    # ==== Additional Check for EMPTY ARRAY
    if (np.max(np.abs(wt.data)) == 0.0 and np.min(np.abs(wt.data)) == 0.0
       or np.isnan(wt.data).all()):
        raise QE.InvalidVariable("Empty Data-Array ... bailing out")
    #
    wt.slice(start_win, end_win)
    # Search A1
    A1 = np.max(np.abs(wt.data))
    A1_timeidx = np.where(np.logical_or(A1 == wt.data, -A1 == wt.data))[0][0]

    if not A1_timeidx:
        raise ValueError("Missing time for Amplitude 1 ... bailing out")
    else:
        A1_timeutc = wt.stats.starttime + A1_timeidx*wt.stats.delta

    if debug_plot:
        wt.stats.timemarks = [
                (A1_timeutc,
                    {'color': 'r', 'markeredgewidth': 1, 'markersize': 50}),
                ]
        wt.plot(plot_time_marks=True)

    return A1, A1_timeutc


def _calculate_half_min_max_amplitude(wt, start_win, end_win):
    """ Trace's amplitude calculation method.

    This method will decifer the half/peak amplitude among the
    maximum and minimum amplitude in Trace.data array.

    The returned UTCDateTime represent the time of the highest absolute
    value among the 2.

    Use trace.slice() method as you need to search and not to process

    """
    # ==== Additional Check for EMPTY ARRAY
    if (np.max(np.abs(wt.data)) == 0.0 and np.min(np.abs(wt.data)) == 0.0
       or np.isnan(wt.data).all()):
        raise QE.InvalidVariable("Empty Data-Array ... bailing out")
    #
    wt.slice(start_win, end_win)
    # Search A1
    A1max = np.max(wt.data)
    A1min = np.min(wt.data)

    # Define Amp
    Awa = (np.abs(A1max) + np.abs(A1min)) / 2.0

    # Find Time
    if np.abs(A1max) >= np.abs(A1min):
        # The max amplitude is positive
        Awa_timeidx = np.where(A1max == wt.data)[0][0]
    else:
        # The max amplitude is negative
        Awa_timeidx = np.where(A1min == wt.data)[0][0]

    if not Awa:
        raise ValueError("Missing time for Amplitude 1 ... bailing out")
    else:
        Awa_timeutc = wt.stats.starttime + Awa_timeidx*wt.stats.delta

    return Awa, Awa_timeutc


class StationMagnitude(object):
    """ Initial attempt to create a magnitude object for stations """
    def __init__(self, optr, opinv):
        """ To start I need the trace (single-trace only) + inventory obj """
        self.__dict__.update(**optr.stats.__dict__)  # expand trace stats

    def __convert_to_obspy_station_magnitude(self):
        """ This method will simply convert to a standard
            ObsPy Station magnitude class, so they can be transferred
        """
        pass

# ========== TIPS

# The constant ε is known as the water level and is chosen to be larger
# than the noise in Y(f ) and G(f). When |G(f)|>ε  large the effect of
# the constant is to reduce the amplitude of the deconvolved signal very slightly.
# When |G(f)| is small the constant stabilizes the deconvolution.

# https://www.seiscomp.de/seiscomp3/doc/jakarta/current/apps/scolv.html#fig-scolv-magnitudes

# ==============================
# ========== EQcorrscan
# ==============================
# https://eqcorrscan.readthedocs.io/en/latest/_modules/eqcorrscan/utils/mag_calc.html#_sim_WA

# --- They do-give::
#  -  unprocessed trace
#  -  detrend data
#  -  remove response
#  -  simulate trace with PAZWA

# --- Helpers for local magnitude estimation
# --- Note Wood anderson sensitivity is 2080 as per Uhrhammer & Collins 1990
# PAZ_WA = {'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],
#           'zeros': [0 + 0j], 'gain': 1.0, 'sensitivity': 2080}

# trace.data = seis_sim(trace.data, trace.stats.sampling_rate,
#                       paz_remove=None, paz_simulate=paz_wa,
#                       water_level=water_level)
