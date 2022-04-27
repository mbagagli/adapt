import os
import shutil
import glob
import logging
#
from obspy import read
from obspy import UTCDateTime
from obspy.io.mseed import InternalMSEEDError
#
import adapt.errors as QE
from adapt.utils import calcEpiDist
from obspy import read as obsread
from adapt.utils import runtimeFormat
from adapt.utils import progressBar
#
from obspy.clients.filesystem.sds import Client as SDSCLIENT
from obspy import Stream

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------


def createEventArchive(opev, eqtag, statDict, metaStatDict, **kwargs):
    """
    This function will download data in a project subdir 'waveforms'.
    The station selection as well asll other info will be stored in the
    main EventDir.

    # 07122018
    Selstat epicentral distance is added on metadatastatdict

    Args:
        EVENTID, adaptStationDict, ObsPy event obj.

    Returns:
        return a list o tuple (statname, rpidist)
    """
    # 0) Select radius from epicenter
    # selStat is a list of tuple (sta,epidist)

    # 1) 02062020 --> now EpiDist added for all the stations!
    selStat, radius, mag, metaStatDict = stationSelection(
                                            opev, eqtag, statDict,
                                            metaStatDict, **kwargs)
    logger.debug("%r %d" % (selStat, len(selStat)))

    if not selStat:
        logger.error(
            "Epicentral distance, station selection gone wrong! (empty)")
        raise QE.MissingVariable("Station selection gone wrong! (empty)")

    # # 1) 07122018 --> add metadata to dict (old)
    # for stat in selStat:
    #     metaStatDict.addStat(eqtag, stat[0], epidist=stat[1])

    # 2) downloadData & Trim MSEED
    if kwargs["DOWNLOAD_DATA"]["doit"]:

        # 2a) Create subdir waveform
        wavedir = (os.sep).join(
            [kwargs["GENERAL"]["workingrootdir"],
             str(opev.resource_id), "waveforms"])

        if os.path.isdir(wavedir) and len(os.listdir(wavedir)) > 0:
            logger.warning("[%s] Waveforms dir NOT EMPTY ..." % eqtag)
            if kwargs["DOWNLOAD_DATA"]["overwritewaveforms"]:
                logger.warning("[%s] Deleting waveforms dir!" % eqtag)
                shutil.rmtree(wavedir)
                logger.info("[%s] Downloading waveforms ..." % eqtag)
                os.makedirs(wavedir, exist_ok=True)
                downloadWaveforms(opev, statDict, selStat, **kwargs)
            else:
                logger.warning("[%s] No waveform owerwriting!" % eqtag)
        elif os.path.isdir(wavedir) and len(os.listdir(wavedir)) == 0:
            logger.warning("[%s] Waveforms dir EMPTY!" % eqtag)
            shutil.rmtree(wavedir)
            logger.info("[%s] Downloading waveforms ..." % eqtag)
            os.makedirs(wavedir, exist_ok=True)
            downloadWaveforms(opev, statDict, selStat, **kwargs)
        else:
            logger.info("[%s] Downloading waveforms ..." % eqtag)
            os.makedirs(wavedir, exist_ok=True)
            downloadWaveforms(opev, statDict, selStat, **kwargs)
    #
    return selStat, radius, mag, metaStatDict


def stationSelection(opev, eqtag, statDict, metaStatDict, **kwargs):
    """
    Given a radius based on the Event Magnitude,
    a selection of station based on epicentral
    distance is made.

    """
    logger.info("Selecting Stations")
    #
    outList = []
    mag = opev.magnitudes[0].mag
    ela = opev.origins[0].latitude
    elo = opev.origins[0].longitude
    # Find index of range of magnitudes
    for xx, ra in enumerate(kwargs["DOWNLOAD_DATA"]["radiusSelection"]):
        if float(ra[0]) < mag <= float(ra[1]):
            radius = ra[2]
    # Calculate EpicentralDistance
    for stat in statDict.keys():
        sla = statDict[stat]["lat"]
        slo = statDict[stat]["lon"]
        # Calculate DISTANCE
        distance = calcEpiDist(ela, elo, sla, slo, outdist="km")

        # 02082020
        metaStatDict.addStat(eqtag, stat, epidist=distance)

        # v0.7.16: adding anyway to all station the P/S correction vals
        try:
            metaStatDict.addStat(eqtag, stat,
                                 p_delay=statDict[stat]["p_delay"],
                                 s_delay=statDict[stat]["s_delay"])
        except KeyError:
            # Missing Station Corrections in the Inventory
            pass

        if distance <= radius:
            outList.append((stat, distance))
            metaStatDict.addStat(eqtag, stat, isselected=True)
        else:
            metaStatDict.addStat(eqtag, stat, isselected=False)

    #
    logger.info("... done")
    return outList, radius, mag, metaStatDict


def downloadWaveforms(opev, statDict, statList, **kwargs):
    """
    This function take care of downloading the
    requested portion of data.

    statList=list of tuple (stationName,distanceEpi )

    """

    # Define General variables
    saveDir = (os.sep).join(
        [kwargs["GENERAL"]["workingrootdir"],
         str(opev.resource_id), "waveforms"])
    logpath = (os.sep).join(
        [kwargs["GENERAL"]["workingrootdir"],
         str(opev.resource_id), "waveforms", "DownloadLog.txt"])
    downloadedStat = []
    channel_names = kwargs["DOWNLOAD_DATA"]["channellist"]
    # channel_names = [ ("HHZ","HHN","HHE"),
    #                   ("BHZ","BHN","BHE"),
    #                   ("EHZ","EHN","EHE"),
    #                   ("HNZ","HNN","HNE") ]

    # Define TimeCuts + OriginTimeString
    odate = (str(opev.origins[0].time.year) +
             str(opev.origins[0].time.month) +
             str(opev.origins[0].time.day) +
             str(opev.origins[0].time.hour) +
             str(opev.origins[0].time.minute) +
             str(opev.origins[0].time.second)
             )
    cutstart = opev.origins[0].time + \
        kwargs["DOWNLOAD_DATA"]["cutsinterval"][0]
    cutend = opev.origins[0].time + kwargs["DOWNLOAD_DATA"]["cutsinterval"][1]
    # cutstart_web = cutstart.format_iris_web_service()
    # cutend_web = cutend.format_iris_web_service()

    # Request+Download+Trimming
    myclient = SDSCLIENT("/alparray",
                         sds_type='D',
                         format='MSEED',
                         fileborder_seconds=400)

    with open(logpath, "wt") as DOWNLOG:
        _start_down_time = UTCDateTime()
        DOWNLOG.write(
                ("  Download START: %s" + os.linesep) %
                _start_down_time.strftime("%Y-%m-%d %H:%M:%S"))
        #
        misscnt = 0
        lessthanthree = []
        for statTuple in statList:
            stat = statTuple[0]
            epidist = statTuple[1]
            baseName = (odate + "." +
                        str(opev.resource_id) + "." +
                        statDict[stat]["network"] + "." +
                        statDict[stat]["fullname"])
            for xx in range(len(channel_names)):            # CHANNEL
                for ii in range(len(channel_names[xx])):    # COMPONENTS
                    # First, check if the component is already taken by
                    # previous tuple!
                    # it seeks indistinctively for mseed and SAC (safer
                    # for trimseed removeOriginal==True)
                    gotcha = glob.glob((os.sep).join(
                                [saveDir, baseName + ".*" +
                                 channel_names[xx][ii][-1] + ".*"]))
                    if not gotcha:
                        mseed_outfile = (os.sep).join(
                            [saveDir, baseName + "." +
                             channel_names[xx][ii] + ".mseed"])

                        # ------ OLDMODE  => FDSN on client
                        # web_request = "wget -O %s 'http://halp9000.ethz.ch:8080/fdsnws/dataselect/1/query?station=%s&location=*&network=%s&channel=%s&start=%s&end=%s' " % (
                        #     mseed_outfile, statDict[stat]["fullname"], statDict[stat]["network"], channel_names[xx][ii], cutstart_web, cutend_web)
                        # DOWNLOG.write(web_request + os.linesep)
                        # os.system(web_request)
                        # if os.stat(mseed_outfile).st_size == 0:
                        #     remove_file = "rm %s" % (mseed_outfile)
                        #     os.system(remove_file)
                        #     DOWNLOG.write(
                        #         ("-- TRIED %s ---" + os.linesep) %
                        #          channel_names[xx][ii])

                        # ------ NEWMODE  => directly on SeisComp
                        # network, station, location, channel, starttime, endtime

                        try:
                            st = myclient.get_waveforms(
                                statDict[stat]["network"],
                                statDict[stat]["fullname"], "*",
                                channel_names[xx][ii],
                                cutstart, cutend)
                            DOWNLOG.write(("ClientCall: %s - %s - %s - %s - %s" +
                                           os.linesep) % (
                                        statDict[stat]["network"],
                                        statDict[stat]["fullname"],
                                        channel_names[xx][ii],
                                        cutstart, cutend)
                                    )
                        except InternalMSEEDError:
                            st = None
                            DOWNLOG.write(
                                 ("-- MSEED ERROR %s ---" + os.linesep) %
                                 channel_names[xx][ii])

                        if not st:
                            DOWNLOG.write(
                                 ("-- TRIED %s ---" + os.linesep) %
                                 channel_names[xx][ii])
                        else:
                            st.write(mseed_outfile, format="MSEED")

                        # MB Precise trimming for the cut (OT-10:Spre+30)
                        if os.path.isfile(mseed_outfile):
                            startt = opev.origins[0].time - 10
                            # traveltime S1 +30 sec
                            endt = opev.origins[0].time + (epidist / 2.2) + 30

                            try:
                                trimMSEED(
                                    mseed_outfile,
                                    startt,
                                    endt,
                                    exportFormat="SAC",
                                    removeOriginal=True)
                            except Exception:
                                # Raised by ObsPy if something go wrong
                                # in merging/cutting MSEED
                                DOWNLOG.write("!! ERROR in merging/cutting" +
                                              os.linesep)
                                os.remove(mseed_outfile)

            # Check if station was downloaded
            # (any sensor/component is valid to append)
            if len(glob.glob((os.sep).join([saveDir, baseName + ".*"]))) > 0:
                downloadedStat.append(statTuple)
                if len(glob.glob((os.sep).join(
                        [saveDir, baseName + ".*"]))) < 3:
                    lttlst = glob.glob((os.sep).join(
                        [saveDir, baseName + ".*"]))
                    lessthanthree.append((stat, lttlst))
            else:
                misscnt += 1
                DOWNLOG.write(
                    ("[WARNING] Station %s.%s was selected for %s " +
                        "... no data available" + os.linesep) %
                    (statDict[stat]["network"], statDict[stat]["fullname"],
                        str(opev.resource_id)))
        # Logging
        DOWNLOG.write(os.linesep * 2)
        _end_down_time = UTCDateTime()
        DOWNLOG.write(
                ("  Download END: %s" + os.linesep) %
                _end_down_time.strftime("%Y-%m-%d %H:%M:%S"))
        totTime = runtimeFormat(_end_down_time - _start_down_time)
        DOWNLOG.write(("Total DOWNLOAD time: %02d:%02d:%05.2f " + os.linesep) %
                      (totTime[0], totTime[1], totTime[2]))
        #
        DOWNLOG.write(os.linesep * 2)
        DOWNLOG.write(
            ("TOTAL SEL. STATIONS: %4d" + os.linesep) %
            (len(statList)))
        DOWNLOG.write(
            ("DOWNLOADED STATIONS: %4d" + os.linesep) %
            (len(downloadedStat)))
        DOWNLOG.write(
            ("MISSED     STATIONS: %4d" + os.linesep * 2) %
            (misscnt))
        if lessthanthree:
            DOWNLOG.write(
                "These stations has LESS THAN 3 components:" +
                os.linesep)
            for ltt in lessthanthree:
                DOWNLOG.write((" " * 4 + "%s" + os.linesep) % ltt[0])
                for fn in range(len(ltt[1])):
                    DOWNLOG.write((" " * 8 + "%s" + os.linesep) %
                                  ltt[1][fn].split(os.sep)[-1])
    #
    return downloadedStat


def trimMSEED(filein, starttime, endtime, exportFormat="SAC",
              removeOriginal=False):
    """
    Finer trimming after webrequest on alp9000.
    The merging of MSEED file will be done using a zero-fill method

    *** NB: filein must be a msedd file
    *** NB: starttime/endtime must be a UTCDateTime obj.
    """
    tr = obsread(filein, format="MSEED")
    #
    if len(tr) > 1:                            # multi - miniseed file
        tr.sort(["starttime"])
        tr.merge(method=1, fill_value=0)
        obspyTrace = tr[0]
    else:
        obspyTrace = tr[0]
    #
    obspyTrace.trim(starttime, endtime)
    obspyTrace.write(".".join(filein.split(".")[:-1]) + "." + exportFormat,
                     format=exportFormat)
    #
    if removeOriginal:
        os.remove(filein)
    return True


def sanity_check(wave_dir,
                 stat_list,
                 event_obj=None,
                 station_obj=None,
                 max_epi_dist=1000.0,
                 wave_format="SAC",
                 out_log=None):
    """ This function will double check that the downloaded waveforms
        are all present for the given iterative list.

        NB: Make sure that the miniseed files contains the same
            station information (grouped by station)

    """
    logger.info("Performing a sanity check ...")
    critical_stats = []
    all_station = []
    #
    wave_list = glob.glob(os.sep.join([wave_dir, "*"+wave_format]))
    for _trn in wave_list:
        _trw = read(_trn)
        all_station.append(_trw[0].stats.station)
    #
    missing_station = tuple(
                        set(stat_list) - set(all_station)
                    )
    #
    if not missing_station:
        sanity_check_result = True
    else:
        sanity_check_result = False

    if event_obj and station_obj:
        mag = event_obj.magnitudes[0].mag
        ela = event_obj.origins[0].latitude
        elo = event_obj.origins[0].longitude
        eqid = event_obj.resource_id
        #
        missing_station = [
            (_sta, calcEpiDist(ela, elo,
                               station_obj[_sta]['lat'],
                               station_obj[_sta]['lon']))
            for _sta in missing_station
        ]
        #
        critical_stats = [
            _xx for _xx in missing_station if _xx[1] <= max_epi_dist
        ]

    #
    if out_log:
        with open(out_log, 'w') as OUT:
            OUT.write(("@@@ Sanity Check Result:   %s"+os.linesep) %
                      sanity_check_result)
            OUT.write(os.linesep+"### Station List:"+os.linesep)
            for _ii in missing_station:
                if event_obj and station_obj:
                    OUT.write(("%s - %6.2f"+os.linesep) % (_ii[0], _ii[1]))
                else:
                    OUT.write(("%s"+os.linesep) % _ii)
            #
            if event_obj and station_obj:
                OUT.write(os.linesep)
                OUT.write("!!! CRITICAL Station List:"+os.linesep)
                for _ii in critical_stats:
                    OUT.write(("%s - %6.2f"+os.linesep) % (_ii[0], _ii[1]))
                OUT.write(os.linesep)
                OUT.write(("@@@ %s - %4.2f - %f"+os.linesep) % (eqid, mag, max_epi_dist))
    #
    return sanity_check_result, missing_station, critical_stats


def database_availability(startt, endt, statdict, channel="*Z",
                          extracttofile="StationsAvailability.txt",
                          sc3db="/alparray", save_stream=False):
    """ This function query the SDS database and return a list
        of all the available data between the given start and end time
    """

    sdsclnt = SDSCLIENT(sc3db,
                        sds_type='D',
                        format='MSEED',
                        fileborder_seconds=400)
    #
    if save_stream:
        st = Stream()
    availDict = {}
    totkey = len(statdict.keys())
    if extracttofile:
        with open(extracttofile, "w") as IN:
            for xx, (_kk, _ss) in enumerate(statdict.items()):
                net = _ss["network"]
                sta = _ss["fullname"]
                lat = _ss["lat"]
                lon = _ss["lon"]
                ele = _ss["elev_m"]
                #
                try:
                    _tmp = sdsclnt.get_waveforms(net, sta, "*", channel, startt, endt)
                except InternalMSEEDError:
                    _tmp = None
                #
                if _tmp:
                    availDict[sta] = {
                                    'lat': lat, 'lon': lon, 'net': net, 'ele': ele
                    }
                    if save_stream:
                        st += _tmp
                    #
                    IN.write(("%s %s %9.5f %8.5f %.0f"+os.linesep) % (
                            net, sta, lon, lat, ele
                        ))
                #
                progressBar(xx, totkey-1, prefix="Checking ", suffix=" DONE",
                            decimals=2, barLength=15)
    else:
        for xx, (_kk, _ss) in enumerate(statdict.items()):
            net = _ss["network"]
            sta = _ss["fullname"]
            lat = _ss["lat"]
            lon = _ss["lon"]
            ele = _ss["elev_m"]
            #
            try:
                _tmp = sdsclnt.get_waveforms(net, sta, "*", channel, startt, endt)
            except InternalMSEEDError:
                _tmp = None
            #
            if _tmp:
                availDict[sta] = {
                                'lat': lat, 'lon': lon, 'net': net, 'ele': ele
                }
                if save_stream:
                    st += _tmp
            #
            progressBar(xx, totkey-1, prefix="Checking ", suffix=" DONE",
                        decimals=2, barLength=15)
    #
    if save_stream:
        return availDict, st
    else:
        return availDict, None
