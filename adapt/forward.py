import logging
import numpy as np
#
from adapt.utils import calcEpiDist
import adapt.errors as QE
#
from pyrocko import cake
from pyrocko.cake import CakeError

logger = logging.getLogger(__name__)

KM = 1000.0
MT = 0.001

# ---------------------------------------- Functions


def PredictPhaseArrival_PYROCKOCAKE(
        opev,
        pickDict,
        statDict,
        statLst,
        phaselist=None,
        vel1d_model=None,
        limit_quake_depth=False,  # km
        avoid_missing_firstarrival=0.5,  # km
        use_stations_corrections=False):
    """
    In this method the predicted first arrival phase is calculated with
    the CAKE module from the PyRocko libraries. It can diffrentiate
    between phases, more info here:
    https://pyrocko.org/docs/current/apps/cake/manual.html
    https://pyrocko.org/docs/current/library/reference/cake.html

    return phaseDict with the upgrade of PREDICTED:{"Ppre":(UTCDateTime)}

    *** NB: The rec_distance array must be a  numpy array!
    *** NB: If use_stat_correction param. is True, 'p_delay', 's_delay'
            are mandatory keys in the statDict!
    *** NB: the model *nd should be 6 fields long:
            `depth pVel sVel Density Qp Qs`

    avoid_missing_firstarrival: this param. is added to solve a CAKE bug.
                                Sometimes there are no arrivals predicted
                                and this is impossible. The value (in km)
                                will define the perturbation of the event
                                source depth. Usually, it's ok just 100 mt

    """
    if not vel1d_model:
        logger.error("Missing 1D Velocity Model, please provide correct " +
                     "path in ConfigFile")
        return False

    logger.info("Loading VEL-1D: %s" % vel1d_model)
    if use_stations_corrections:
        logger.info("Using Stations Corrections!")

    # *** NB default measurement in cake are in meters #headonly=True
    if phaselist:
        logger.info("Predicting phase-arrival times of phases: %s" %
                    str(phaselist))
        eqid = opev.origins[0].resource_id
        model = cake.load_model(vel1d_model)
        source_depth = opev.origins[0].depth   # MUST BE in METERS !!!

        # Convert standard names to PyRocko ones
        phase2calc = []
        for ii in phaselist:
            phase2calc.extend(cake.PhaseDef.classic(ii))
            # # @develop: quick hack for issue on CAKE
            # # @develop: (https://git.pyrocko.org/pyrocko/pyrocko/issues/114)
            # if ii == 'Pg':
            #     phase2calc.append(cake.PhaseDef('p<(moho)'))
            # else:
            #     phase2calc.extend(cake.PhaseDef.classic(ii))

        # v0.7.17: adjust quake depths based on a matrix
        # i.e. [ [lim_inf, fixdepth_km], [lim_sup, fixeddepth_km] ]
        if limit_quake_depth:
            logger.info("The `limit_quake_depth` option is active: %s" %
                        limit_quake_depth)
            if source_depth <= limit_quake_depth[0][0] * KM:
                logger.info("Adjusting DEPTH for prediction: %7.2f km --> %7.2f km" %
                            (source_depth * MT, limit_quake_depth[0][1]))
                source_depth = limit_quake_depth[0][1] * KM  # km -> m

            elif source_depth >= limit_quake_depth[1][0] * KM:
                logger.info("Adjusting DEPTH for prediction: %7.2f km --> %7.2f km" %
                            (source_depth * MT, limit_quake_depth[1][1]))
                source_depth = limit_quake_depth[1][1] * KM  # km -> m

        # For subsequent reset (at each station loop)
        original_source_depth = source_depth

        # Loop over the stations
        rottenstat = []  # station with erroneus altidute, or create problem
        for ii in statLst:
            # Reset source_depth to avoid constant shift only PmP
            # calculations adjustment
            source_depth = original_source_depth

            if isinstance(ii, (list, tuple)):
                stat = statDict[ii[0]]
            elif isinstance(ii, str):
                stat = statDict[ii]
            else:
                logger.error("Wrong station index type")
                raise QE.InvalidParameter("Wrong station index type")

            distStaEpi = calcEpiDist(opev.origins[0].latitude,
                                     opev.origins[0].longitude,
                                     stat['lat'], stat['lon'], outdist="m")
            # need to be a numpyarray list
            rec_distance = np.asarray([distStaEpi]) * cake.m2d

            logger.debug("ID: %s - NAME: %s - NET: %s - EQDm: %7.1f - "
                         "STEm: %7.2f - SHOOT: %6.1f" % (
                          eqid, stat["fullname"], stat["network"],
                          opev.origins[0].depth, stat['elev_m'], source_depth))

            # =================================== Calculate TravelTimes
            perturbate_source = False
            try:
                cnt, pkd = shoot_rays(opev,
                                      model,
                                      stat,
                                      rec_distance,
                                      phaselist,
                                      phase2calc,
                                      source_depth,
                                      use_stat_corr=use_stations_corrections)

                if ((cnt['PmP'] == 1 and cnt['Pg'] == 0 and cnt['Pn'] == 0) or
                   (cnt['PmP'] == 0 and cnt['Pg'] == 0 and cnt['Pn'] == 0)):
                    # NO predictions found (Only PmP or 0 count), therefore
                    # perturbate source (deeper)
                    perturbate_source = True
                    source_depth = (
                                source_depth +
                                (avoid_missing_firstarrival * KM))  # meters
                    #
                    logger.warning("ID: %s - STAT: %s - EPIdist: %6.2f  --> "
                                   "No first-arrivals found!! New SHOOT: %9.3f" % (
                                    eqid, stat['fullname'],
                                    distStaEpi * MT, source_depth * MT))
                    #
                    cnt, pkd = shoot_rays(opev,
                                          model,
                                          stat,
                                          rec_distance,
                                          phaselist,
                                          phase2calc,
                                          source_depth,
                                          use_stat_corr=(
                                                use_stations_corrections))

            except CakeError as e:
                logger.error(e)
                logger.error("Problem with Station: %s.%s ^^^ (prev.line)" % (
                             stat["network"], stat["fullname"])
                             )
                rottenstat.append((
                            stat["network"],
                            stat["fullname"],
                            stat['elev_m'],
                            e
                            ))
                continue

            finally:
                # If still no first arrival, BUMPS out!!!
                if ((cnt['PmP'] == 1 and cnt['Pg'] == 0 and cnt['Pn'] == 0) or
                   (cnt['PmP'] == 0 and cnt['Pg'] == 0 and cnt['Pn'] == 0)):
                    logger.error("ID: %s - STAT: %s - SOURCE: %9.3f - EPIdist: %6.2f  --> "
                                 "NO ARRIVALS, EXIT !!" % (
                                    eqid, stat['fullname'],
                                    source_depth * MT, distStaEpi * MT))
                    # # --- Original QUAKE --> throw an error
                    # raise QE.MissingVariable()
                    # # --- ADAPT v0.8 --> note and go on
                    # rottenstat.append((
                    #             stat["network"],
                    #             stat["fullname"],
                    #             stat['elev_m'],
                    #             "No first arrival prediction"
                    #             ))
                    raise QE.MissingVariable()
                else:
                    if perturbate_source:
                        # 2nd try was done, advise you're safe now!
                        logger.info(
                          "ID: %s - STAT: %s - SOURCE: %9.3f - EPIdist: %6.2f  --> "
                          "FOUND ARRIVALS !!" % (
                                    eqid, stat['fullname'],
                                    source_depth * MT, distStaEpi * MT))

            # =================================== Sorting
            for ii in phaselist:
                if pkd[ii]:
                    tmppd = {'polarity': None,
                             'onset': None,
                             'weight': None,
                             'pickclass': None,
                             'timeUTC_early': None,
                             'timeUTC_late': None,
                             'timeUTC_pick': sorted(pkd[ii])[0]
                             }
                    namephs = "Predicted_" + ii
                    logger.debug("%s - %s" % (namephs, tmppd["timeUTC_pick"]))
                    pickDict.addPick(stat["network"]+"."+stat["fullname"],
                                     namephs, **tmppd)
        #
    return pickDict, rottenstat


def shoot_rays(opev,
               model,
               statinfo,
               rec_distance,
               phaselist,
               phase2calc,
               source_depth,
               use_stat_corr=False):
    """ Small routine that calculates the traveltimes for Pyrocko CAKE
        model.
    """

    cnt, pkd = {}, {}
    for ii in phaselist:
        cnt[ii], pkd[ii] = 0, []

    for ttimes in model.arrivals(
            rec_distance,
            phases=phase2calc,
            zstart=source_depth,
            zstop=-statinfo['elev_m']):
        namphskey = ttimes.given_phase().given_name()

        # @develop: quick hack for issue on CAKE
        # @develop: (https://git.pyrocko.org/pyrocko/pyrocko/issues/114)
        # phsdef = ttimes.given_phase()
        # if phsdef == "p<(moho)":
        #     namphskey = "Pg"

        if use_stat_corr:
            corr = statinfo[namphskey[0].lower() + "_delay"]
            pkd[namphskey].append(opev.origins[0].time +
                                  ttimes.t + corr)
        else:
            pkd[namphskey].append(opev.origins[0].time + ttimes.t)
        cnt[namphskey] += 1
    #
    return cnt, pkd
