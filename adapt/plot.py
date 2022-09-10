import os
import logging
import numpy as np
# Cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# Picks
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from adapt.processing import normalizeTrace
# Error and Checks
import adapt.errors as QE
from adapt.utils import calcEpiDist, get_pick_slice
from obspy.core import UTCDateTime
from obspy.core.trace import Trace
from obspy.core.stream import Stream

logger = logging.getLogger(__name__)
#mpl.use('Agg')  # fix for display issue n bigstar (Keep commented!)

# -------------------------------------------- COSTANTS

# STANDARD_CARTOPY_PROJ = ccrs.Mercator(0, 45)

STANDARD_CARTOPY_PROJ = ccrs.TransverseMercator(0, 45)
STANDARD_MAP_EXTENT = [1, 21, 41, 51]

MT = 0.001
KM = 1000.0

# -------------------------------------------- MyStyle

plt.style.use("seaborn-colorblind")
csfont = {'fontname': 'cmss10'}
hfont = {'fontname': 'cmss10', 'weight': 'bold', 'size': 15}  # cmb10
tfont = {'fontname': 'cmss10'}
lfont = {'fname': '/home/matteo/miniconda3/envs/quake_v061_p376/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/cmss10.ttf', 'size': 12}  # 'fontname': 'cmss10'


# -------------------------------------------- PRIVATE  Functions
def __miniproc__(st):
    prs = st.copy()
    prs.detrend('demean')
    prs.detrend('simple')
    prs.taper(max_percentage=0.05, type='cosine')
    prs.filter("bandpass",
               freqmin=1,
               freqmax=30,
               corners=2,
               zerophase=True)
    return prs

# -------------------------------------------- PUBLIC  Functions


def plot_ADAPT_CF(st,
                  CF,
                  chan="*Z",
                  picks=None,
                  inax=None,
                  normalize=False,
                  axtitle='adapt_cf',
                  shift_cf=None,
                  show=False):
    """
    Input is a obspy stream object
    Method to plot all the necessary CF of ADAPT pickers.

    CF is a dict containing the name of CF as key and np.array as data
    The key name will be used for plotting legend

    picks is a dict containing the name as key and idx/UTCDateTime as
    value. The key name will be used for plotting legend

    """
    if not isinstance(st, Stream):
        raise QE.InvalidVariable({"message":
                                  "Please input a valid obspy.stream object"})
    if not isinstance(CF, dict):
        raise QE.InvalidVariable({"message":
                                  "CF must be a dict containing the name of" +
                                  " CF as key and np.array as data"})
    if picks and not isinstance(picks, dict):
        raise QE.InvalidVariable({"message": "picks must be a dict"})
    # deep copy to avoid changes on original stream
    wst = st.copy()
    #
    tr = wst.select(channel=chan)[0]
    orig = tr.data
    df = tr.stats.sampling_rate
    # just transform in seconds the x ax (sample --> time)
    tv = np.array([ii / df for ii in range(len(orig))])
    #
    colorlst = ("r",
                "b",
                "g",
                "y",
                "c")

    colorlst = (
            '#0072B2',
            '#009E73',
            '#D55E00',
            '#CC79A7',
            '#F0E442',
            '#56B4E9'
            )

    # CFs
    if normalize:
        orig = normalizeTrace(orig, rangeVal=[-1, 1])

    for _ii, (_kk, _aa) in enumerate(CF.items()):
        if normalize:
            _aa = normalizeTrace(_aa, rangeVal=[0, 1])
        zeropad = len(orig) - len(_aa)
        if not inax:
            if shift_cf:
                plt.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                    constant_values=(np.nan,)) +
                             (_ii+1)*shift_cf,
                         colorlst[_ii], label=_kk)
            else:
                plt.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                    constant_values=(np.nan,)),
                         colorlst[_ii], label=_kk)
            if picks:
                for _pp, _tt in picks.items():
                    plt.axvline(picks[_pp]/df,
                                color=colorlst[_ii],
                                linewidth=2,
                                linestyle='dashed',
                                label=_pp)

        else:
            if shift_cf:
                inax.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                     constant_values=(np.nan,))  +
                             (_ii+1)*shift_cf,
                          colorlst[_ii], label=_kk)
            else:
                inax.plot(tv, np.pad(_aa, (zeropad, 0), mode='constant',
                                     constant_values=(np.nan,)),
                          colorlst[_ii], label=_kk)

            if picks:
                for _pp, _tt in picks.items():
                    inax.axvline(picks[_pp]/df,
                                 color=colorlst[_ii],
                                 linewidth=2,
                                 linestyle='dashed',
                                 label=_pp)

    # StreamTrace + labels
    if not inax:
        plt.plot(tv, orig, "k", label="trace")
        plt.xlabel("time (s)", **csfont)
        plt.ylabel("counts", **csfont)
        plt.legend(loc='lower left', prop=lfont)
        plt.title(axtitle, **hfont, loc='center')
    else:
        inax.plot(tv, orig, "k", label="trace")
        inax.set_xlabel("time (s)", **csfont)
        inax.set_ylabel("counts", **csfont)
        plt.legend(loc='lower left', prop=lfont)
        inax.set_title(axtitle, **hfont, loc='center')  #  {'fontsize': 16, 'fontweight': 'bold'})

    if show:
        plt.tight_layout()
        plt.show()
    return True


def plot_AIC_CF(aic, inax=None, evidence=None, show=False):
    """
    Plot MyAIC characteristic function

    AIC => numpy series
    evidence = > numpy.ndarray / list
    """
    if not inax:
        tv = np.arange(len(aic))
        plt.plot(tv, aic)
    else:
        inax.plot(aic)
    #
    if isinstance(evidence, (np.ndarray, list)):
        plt.axhline(y=aic.mean(), color="gray", linestyle="--")
        plt.plot(evidence, aic[evidence], "x")
    #
    if show:
        plt.show()
    return True


def plotMap_ADAPT(opev, statDict, statLst, inax=None, **kwargs):
    """
    This function plots an Eruopean Map with
    localized eq (redstar) and stations (could be input a list/tuple or
    by fullname as index). This function adopts the newer libraries for
    mapping in python "Cartopy".

    *** NB: if inax is specified, it MUST contain the Cartopy projection
            class ALREADY!

    RETURNS: a Matplotlib Axes
    """
    if not inax:
        mappax = plt.axes(projection=STANDARD_CARTOPY_PROJ)
    else:
        mappax = inax

    mappax.set_extent(STANDARD_MAP_EXTENT)

    # draw coastlines and borders
    mappax.add_feature(cfeature.NaturalEarthFeature("physical",
                                                    "coastline",
                                                    "50m",
                                                    edgecolor='black',
                                                    facecolor='none',
                                                    linestyle="-", lw=0.7))

    mappax.add_feature(cfeature.NaturalEarthFeature("cultural",
                                                "admin_0_boundary_lines_land",
                                                    "10m",
                                                    edgecolor='black',
                                                    facecolor='none',
                                                    linestyle="--", lw=0.5))

    # draw meridians and parallels
    gl = mappax.gridlines(color='k', linestyle=(0, (1, 1)),
                          xlocs=range(-5, 35, 5),
                          ylocs=range(35, 60, 5))

    # *** NB:
    # at this stage is not possible to have coordinate label so easily
    # in cartopy map. deal with it!

    # # latlon label
    # mappax.set_xticks([0, 5, 10, 15, 20], crs=STANDARD_CARTOPY_PROJ)
    # mappax.set_yticks([41, 42, 43, 44, 45], crs=STANDARD_CARTOPY_PROJ)
    # # mappax.set_yticks([41, 42, 43, 44, 45], crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(number_format='.1f')
    # lat_formatter = LatitudeFormatter(number_format='.1f')
    # mappax.xaxis.set_major_formatter(lon_formatter)
    # mappax.yaxis.set_major_formatter(lat_formatter)

    # # latlon label
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # gl.xlabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

    # Draw eq + station
    mappax.plot(opev.origins[0].longitude, opev.origins[0].latitude,
                "r*", ms=10, mew=1.5, mec="k", transform=ccrs.Geodetic())

    if isinstance(statLst, (list, tuple)):
        for ii in statLst:
            stat = statDict[ii]
            mappax.plot(stat['lon'], stat['lat'], "g^",
                        ms=7, mew=0.7, mec="k", transform=ccrs.Geodetic())
    elif isinstance(statLst, str):
        stat = statDict[statLst]
        mappax.plot(stat['lon'], stat['lat'], "g^",
                    ms=7, mew=0.7, mec="k", transform=ccrs.Geodetic())
    else:
        logger.error("Wrong station index")

    # Plot
    try:
        if kwargs["show"]:
            plt.show()
    except KeyError:
        pass
    #
    return mappax


def plotPick_ADAPT(worktrace,
                   pickcont,
                   inax=None,
                   plotPickPhase=None,
                   pickidx=None,
                   plotbackup=False,
                   show=False):
    """
    This module serves for plotting the ADAPT pick container.
    RETURNS:
        - fig // The handle of the plotted fig

    *** NB: the station NAME (as idx) for the relative picks are
            taken directly in the obspy waveform stat

    *** NB: if pickIDX is specified, the exact pick of the list
            for each phase is choosen. The idx cannot be greater than
            len(MPO.slicesdelta)

    If plotbackup is set to true, the function will try to detect also
    the evaluated FALSE multipicks. This function perform best for the
    multipicking stage, rather than the plotPick_ADAPT_uncertainty

    """
    if not isinstance(worktrace, Trace):
        logger.error("Not a valid ObsPy trace ...")
        return False

    # ---------------------------------- Colorlist for picks
    ''' Next list is the old ColorList
         NEW: ColorList[phase[0:3]],
    '''
    ColorList = {"Ref": "#08f000",  # green
                 "Pre": "teal",  # golden yellow
                 "Sei": "darkgrey",
                 "Ass": "#fec615",  # golden yellow
                 "BK_": "#4f4f56",  # darkslategray
                 "AIC": "#d66460",  # indiared
                 "BAI": "#af928b",  # rosybrown
                 "HOS": "#dacfc0",  # lightgray
                 # "HOS": "#029386",  # teal
                 "FP_": "orange",
                 "P1": "#800020",
                 "VEL": "blue",   # burgundry
                 #
                 "AAR": "#800020",
                 "KIT": "blue"   # burgundry
                 }
    LineSpecList = ("-", "--", ":")  # Used for multiple istance of Pick

    # "BK_": "#800020",  # burgundry
    # "MyA": "#029386",  # teal
    # "HOS": "#C0C0C0",  # silver
    # "BAI": "#C0C0C0"   # silver
    # }

    # ---------------------------------- Create Time+Data Vector
    stat = worktrace.stats.station
    timeVector = worktrace.times("matplotlib")
    dataVector = worktrace.data

    # ---------------------------------- Plot Waves
    if not inax:
        ax1 = plt.axes()
    else:
        ax1 = inax
    ax1.plot(timeVector, dataVector, color='black')
    # ---------------------------------- Plot Picks and Decorator
    for col, phase in enumerate(pickcont[stat]):
        ''' 05122018
        Colors cannot be the same from one figure to another, because
        if they are not in picklist,the idx is also skipped, depends
        on how many picktag are in the dictionary for a given station.

        TODO: maybe create a dictionary of colors based on tag!
        '''
        # logger.debug(("col: %d - phase: %s") % (col, phase))
        if phase in plotPickPhase:
            if isinstance(pickidx, int) and pickidx >= 0:
                # selected pick list
                try:
                    picktime = pickcont[stat][phase][pickidx]["timeUTC_pick"]
                    picktime_bkp = pickcont[stat][phase][pickidx]["backup_picktime"]
                except IndexError:
                    # probably these are the CAKE predicted --> onlyone
                    logger.warning(("Index %d not found for: %s - %s . " +
                                    "taking first value instead [0]!") %
                                   (pickidx, stat, phase))
                    picktime = pickcont[stat][phase][0]["timeUTC_pick"]
                    picktime_bkp = pickcont[stat][phase][0]["backup_picktime"]

                if picktime:
                    if phase in ("P1", "~P1", "*P1"):
                        _color = "#800020"
                    else:
                        _color = ColorList[phase[0:3]]
                    ax1.axvline(picktime.datetime,
                                color=_color,
                                linewidth=2,
                                linestyle='solid',
                                label=phase)
                elif plotbackup:
                    if picktime_bkp:
                        if phase in ("P1", "~P1", "*P1"):
                            _color = "#800020"
                        else:
                            _color = ColorList[phase[0:3]]
                        ax1.axvline(picktime_bkp.datetime,
                                    color=_color,
                                    linewidth=1.5,
                                    linestyle='solid',
                                    label=phase+"_backup")
            else:
                # all pick in list
                for idx in range(len(pickcont[stat][phase])):
                    picktime = pickcont[stat][phase][idx]["timeUTC_pick"]
                    picktime_bkp = pickcont[stat][phase][idx]["backup_picktime"]
                    if picktime:
                        if phase in ("P1", "~P1", "*P1"):
                            _color = "#800020"
                        else:
                            _color = ColorList[phase[0:3]]
                        ax1.axvline(picktime.datetime,
                                    color=_color,
                                    linewidth=1.5,
                                    linestyle=LineSpecList[idx],
                                    label=phase+"_"+str(idx))
                    elif plotbackup:
                        if picktime_bkp:
                            if phase in ("P1", "~P1", "*P1"):
                                _color = "#800020"
                            else:
                                _color = ColorList[phase[0:3]]
                            ax1.axvline(picktime_bkp.datetime,
                                        color=_color,
                                        linewidth=1.5,
                                        linestyle=LineSpecList[idx],
                                        label=phase+"_"+str(idx)+"_backup")

    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='lower right')
    # ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

    # NEW: the next call should take care of the X-axis formatting on zoom
    AutoDateFormatter(AutoDateLocator())

    # Plot
    if show:
        plt.show()
    #
    return ax1


def plotPick_ADAPT_uncertainty(worktrace,
                               quake_pkct,
                               refer_pkct,
                               inax=None,
                               pickidx=None,
                               phaseplot=None,
                               plotwhat=('mean', 'median'),
                               debug_mode=False,
                               show=False):
    """
    This module serves for plotting the ADAPT FINAL pick container.
    RETURNS:
        - fig // The handle of the plotted fig

    *** NB: the station NAME (as idx) for the relative picks are
            taken directly in the obspy waveform stat

    *** NB: if pickIDX is specified, the exact pick of the list
            for each phase is choosen. The idx cannot be greater than
            len(MPO.slicesdelta)

    MB: changed in the  0.4.1


    """

    if not isinstance(worktrace, Trace):
        logger.error("Not a valid ObsPy trace ...")
        return False

    if not phaseplot:
        logger.error("No phaseplot specified ...")
        return False

    # ---------------------------------- Create Time+Data Vector
    stat = worktrace.stats.station
    timeVector = worktrace.times("matplotlib")
    dataVector = worktrace.data

    # ---------------------------------- Plot Waves
    if not inax:
        ax1 = plt.axes()
    else:
        ax1 = inax
    ax1.plot(timeVector, dataVector, color='black')

    # ---------------------------------- Plot ADAPT Picks and Decorator
    for phase in quake_pkct[stat]:
        if phase in phaseplot:
            if isinstance(pickidx, int) and pickidx >= 0:
                # selected pick list
                try:
                    weightobj = quake_pkct[stat][phase][pickidx]["weight"]
                except IndexError:
                    logger.error("Wrong index specified ...")
                    return False

                if weightobj:
                    stdall = weightobj.get_uncertainty(method="std")
                    meanpick = weightobj.get_picktime(method="mean")
                    medianpick = weightobj.get_picktime(method="median")
                    # # mean
                    # ax1.axvspan((meanpick-stdall).datetime,
                    #             (meanpick+stdall).datetime,
                    #             alpha=0.5, color='lightgray')
                    # # median
                    # ax1.axvspan((medianpick-stdall).datetime,
                    #             (medianpick+stdall).datetime,
                    #             alpha=0.4, color='lightgray')
                    # tearly, tlatest
                    timeUTC_early = UTCDateTime(min(dict(
                        weightobj.triage_dict['valid_obs']).values()))
                    timeUTC_late = UTCDateTime(max(dict(
                        weightobj.triage_dict['valid_obs']).values()))
                    ax1.axvspan(timeUTC_early.datetime,
                                timeUTC_late.datetime,
                                alpha=0.5, color='darkgray')
                    if 'mean' in plotwhat:
                        ax1.axvline(
                            meanpick.datetime,
                            color="#fec615",
                            linewidth=2,
                            linestyle='solid',
                            label=phase+'_mean')
                    if 'median' in plotwhat:
                        ax1.axvline(
                            medianpick.datetime,
                            color="#900020",
                            linewidth=2,
                            linestyle='solid',
                            label=phase+'_median')

            else:
                # all pick in list
                for idx in range(len(quake_pkct[stat][phase])):
                    weightobj = quake_pkct[stat][phase][pickidx]["weight"]
                    if weightobj:
                        stdall = weightobj.get_uncertainty(method="all")
                        meanpick = weightobj.get_mean(out_type="date")
                        medianpick = weightobj.get_median(out_type="date")
                        # # mean
                        # ax1.axvspan(meanpick-stdall,
                        #             meanpick+stdall,
                        #             alpha=0.5, color='lightgray')
                        # median
                        ax1.axvspan(medianpick-stdall,
                                    medianpick+stdall,
                                    alpha=0.4, color='lightgray')
                        if 'mean' in plotwhat:
                            ax1.axvline(
                                meanpick.datetime,
                                color="#fec615",
                                linewidth=2,
                                linestyle='solid',
                                label=phase+'_mean')
                        if 'median' in plotwhat:
                            ax1.axvline(
                                medianpick.datetime,
                                color="#900020",
                                linewidth=2,
                                linestyle='solid',
                                label=phase+'_median')

    # ---------------------------------- Plot REFERERENCE Picks
    try:
        for _phase in refer_pkct[stat]:
            if _phase[0:3] == "Ref":
                picktime = refer_pkct[stat][_phase][0]["timeUTC_pick"]
                ax1.axvline(
                    picktime.datetime,
                    color="#08f000",
                    linewidth=2,
                    linestyle='solid',
                    label=_phase)
    except KeyError:
        logger.warning("No REFERENCE pick for %s station!" % stat)

    # ---------------------------------- Plot DEBUG Picks
    debug_plot_list = ['Predicted_Pg', 'Predicted_Pn', 'Predicted_PmP',
                       'Seiscomp_P', 'BAIT_EVAL_1', 'BAIT_EVAL_2', 'VELEST']
    if debug_mode:
        for _phase in refer_pkct[stat]:
            if _phase in debug_plot_list:
                try:
                    picktime = (refer_pkct[stat][_phase]
                                          [pickidx]["timeUTC_pick"])

                    # # ----- MB
                    # if _phase[-3:] == "PmP":
                    #    ax1.axvline(
                    #         picktime.datetime,
                    #         color="purple",
                    #        linewidth=1,
                    #        linestyle='solid',
                    #         label=_phase)
                    # -----

                    if _phase[0:3] == "Pre":
                        ax1.axvline(
                            picktime.datetime,
                            color="darkturquoise",
                            linewidth=1,
                            linestyle='dashed',
                            label=_phase)

                    elif _phase[0:3] == "BAI":
                        ax1.axvline(
                            picktime.datetime,
                            color="teal",
                            linewidth=1.5,
                            linestyle='solid',
                            label=_phase)

                    elif _phase[0:3] == "Sei":
                        ax1.axvline(
                            picktime.datetime,
                            color="gray",
                            linewidth=1,
                            linestyle='dashed',
                            label=_phase)
                    elif _phase[0:3] == "VEL":
                        ax1.axvline(
                            picktime.datetime,
                            color="gray",
                            linewidth=1,
                            linestyle='dashed',
                            label=_phase)

                    else:
                        logger.error("Wrong PHASE type in debug plot!")
                        raise QE.InvalidVariable()
                except KeyError:
                    logger.warning("Missing Debug Pick: %s - Station: %s" %
                                   (_phase, stat))

    # ---------------------------------- Closing
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='lower right')
    # ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

    # NEW: the next call should take care of the X-axis formatting on zoom
    AutoDateFormatter(AutoDateLocator())

    # Plot
    try:
        if show:
            plt.show()
    except KeyError:
        pass
    #
    return ax1


def plotPick_ADAPT_bootstrap(worktrace,
                             quake_pkct,
                             refer_pkct,
                             inax=None,
                             pickidx=None,
                             phaseplot=None,
                             plotwhat=('mean', 'median'),
                             debug_mode=False,
                             show=True):
    """
    This module serves for plotting the ADAPT FINAL pick container.
    RETURNS:
        - fig // The handle of the plotted fig

    *** NB: the station NAME (as idx) for the relative picks are
            taken directly in the obspy waveform stat

    *** NB: if pickIDX is specified, the exact pick of the list
            for each phase is choosen. The idx cannot be greater than
            len(MPO.slicesdelta)

    MB: changed in the  0.4.1


    """

    if not isinstance(worktrace, Trace):
        logger.error("Not a valid ObsPy trace ...")
        return False

    if not phaseplot:
        logger.error("No phaseplot specified ...")
        return False

    # ---------------------------------- Create Time+Data Vector
    stat = worktrace.stats.station
    timeVector = worktrace.times("matplotlib")
    dataVector = worktrace.data

    # ---------------------------------- Plot Waves
    if not inax:
        ax1 = plt.axes()
    else:
        ax1 = inax
    ax1.plot(timeVector, dataVector, color='black')

    # ---------------------------------- Plot ADAPT Picks and Decorator
    for phase in quake_pkct[stat]:
        if phase in phaseplot:
            if isinstance(pickidx, int) and pickidx >= 0:
                # selected pick list
                try:
                    weightobj = quake_pkct[stat][phase][pickidx]["weight"]
                except IndexError:
                    logger.error("Wrong index specified ...")
                    return False

                if weightobj:
                    stdall = weightobj.get_uncertainty(method="std")
                    meanpick = weightobj.get_picktime(method="mean")
                    medianpick = weightobj.get_picktime(method="median")
                    # # mean
                    # ax1.axvspan((meanpick-stdall).datetime,
                    #             (meanpick+stdall).datetime,
                    #             alpha=0.5, color='lightgray')
                    # median
                    ax1.axvspan((medianpick-stdall).datetime,
                                (medianpick+stdall).datetime,
                                alpha=0.4, color='lightgray')
                    if 'mean' in plotwhat:
                        ax1.axvline(
                            meanpick.datetime,
                            color="#fec615",
                            linewidth=2,
                            linestyle='solid',
                            label=phase+'_mean')
                    if 'median' in plotwhat:
                        ax1.axvline(
                            medianpick.datetime,
                            color="#900020",
                            linewidth=2,
                            linestyle='solid',
                            label=phase+'_median')

            else:
                # all pick in list
                for idx in range(len(quake_pkct[stat][phase])):
                    weightobj = quake_pkct[stat][phase][pickidx]["weight"]
                    if weightobj:
                        stdall = weightobj.get_uncertainty(method="all")
                        meanpick = weightobj.get_mean(out_type="date")
                        medianpick = weightobj.get_median(out_type="date")
                        # # mean
                        # ax1.axvspan(meanpick-stdall,
                        #             meanpick+stdall,
                        #             alpha=0.5, color='lightgray')
                        # median
                        ax1.axvspan(medianpick-stdall,
                                    medianpick+stdall,
                                    alpha=0.4, color='lightgray')
                        if 'mean' in plotwhat:
                            ax1.axvline(
                                meanpick.datetime,
                                color="#fec615",
                                linewidth=2,
                                linestyle='solid',
                                label=phase+'_mean')
                        if 'median' in plotwhat:
                            ax1.axvline(
                                medianpick.datetime,
                                color="#900020",
                                linewidth=2,
                                linestyle='solid',
                                label=phase+'_median')
    #

    # ---------------------------------- Plot BOOTSTRAP Picks
    try:
        picktime = quake_pkct[stat]['P1'][0]["weight"].get_triage_results()['bootmode']
        if picktime:
            ax1.axvline(
                picktime.datetime,
                color="purple",
                linewidth=2,
                linestyle='solid',
                label="BootStrap")
    except KeyError:
        logger.warning("No BOOTSTRAP pick for %s station!" % stat)

    # ---------------------------------- Plot REFERERENCE Picks
    try:
        for _phase in refer_pkct[stat]:
            if _phase[0:3] == "Ref":
                picktime = refer_pkct[stat][_phase][0]["timeUTC_pick"]
                ax1.axvline(
                    picktime.datetime,
                    color="#08f000",
                    linewidth=2,
                    linestyle='solid',
                    label=_phase)
    except KeyError:
        logger.warning("No REFERENCE pick for %s station!" % stat)

    # ---------------------------------- Plot DEBUG Picks
    debug_plot_list = ['Predicted_Pg', 'Predicted_Pn', 'Predicted_PmP',
                       'Seiscomp_P', 'BAIT_EVAL_1', 'BAIT_EVAL_2']
    if debug_mode:
        for _phase in refer_pkct[stat]:
            if _phase in debug_plot_list:
                try:
                    picktime = (refer_pkct[stat][_phase]
                                          [pickidx]["timeUTC_pick"])

                    # # ----- MB
                    # if _phase[-3:] == "PmP":
                    #    ax1.axvline(
                    #         picktime.datetime,
                    #         color="purple",
                    #        linewidth=1,
                    #        linestyle='solid',
                    #         label=_phase)
                    # -----

                    if _phase[0:3] == "Pre":
                        ax1.axvline(
                            picktime.datetime,
                            color="darkturquoise",
                            linewidth=1,
                            linestyle='dashed',
                            label=_phase)

                    elif _phase[0:3] == "BAI":
                        ax1.axvline(
                            picktime.datetime,
                            color="teal",
                            linewidth=1.5,
                            linestyle='solid',
                            label=_phase)

                    elif _phase[0:3] == "Sei":
                        ax1.axvline(
                            picktime.datetime,
                            color="gray",
                            linewidth=1,
                            linestyle='dashed',
                            label=_phase)

                    else:
                        logger.error("Wrong PHASE type in debug plot!")
                        raise QE.InvalidVariable()
                except KeyError:
                    logger.warning("Missing Debug Pick: %s - Station: %s" %
                                   (_phase, stat))

    # ---------------------------------- Closing
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='lower right')
    # ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

    # NEW: the next call should take care of the X-axis formatting on zoom
    AutoDateFormatter(AutoDateLocator())

    # Plot
    try:
        if show:
            plt.show()
    except KeyError:
        pass
    #
    return ax1


def composeFigure_ADAPT(opev, stidx, statDict, metastatdict, pickcont, worktrace,
                        multipickerobj=None, **kwargs):
    """
    This function organize a PICK figure by callind internally
    the necessary functions
    PARAMETERS:
        - opev          (ObsPy Event Obj.)
        - stidx         (str - statname)
        - statDict      (ADAPT Station Container)
        - metastatdict  (ADAPT StatContainer_Event obj)
        - pickcont
        - worktrace     (ObsPy Trace Class)
        - multipickobj  (adapt MultiPicker obj.)
    RETURNS:
        - matplolib Figure Class
    REFERENCES:
www.jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
    """
    eqid = opev.resource_id.id
    figtitlehead = (" - ".join([
                        eqid,
                        worktrace.stats.station, statDict[stidx]["network"],
                        "{:.2f}".format(metastatdict[eqid][stidx]["epidist"]) +
                    " km"])
                    )

    logger.info("Creating FIGURE: %s" % figtitlehead)

    #MB debug
    # --- CREATED FOR EGU 2019
    fig = plt.figure(figsize=(16, 9))
    grid = plt.GridSpec(2, 4, wspace=0.4, hspace=0.3)

    # Create SubplotAxis to be passed
    ax_wavepick = fig.add_subplot(grid[0, 0:2])
    ax_map = fig.add_subplot(grid[0, 2:], projection=STANDARD_CARTOPY_PROJ)
    ax_win_1 = fig.add_subplot(grid[1, 0:2])
    ax_win_2 = fig.add_subplot(grid[1, 2:])

    # --- ORIGINAL
    # fig = plt.figure(figsize=(16, 9))
    # grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

    # # Create SubplotAxis to be passed
    # ax_wavepick = fig.add_subplot(grid[0, 0:2])
    # ax_map = fig.add_subplot(grid[0, -1], projection=STANDARD_CARTOPY_PROJ)
    # ax_win_1 = fig.add_subplot(grid[1, 0])
    # ax_win_2 = fig.add_subplot(grid[1, 1])
    # ax_win_3 = fig.add_subplot(grid[1, 2])

    # Extract the slices on Z component:
    # (NEW after multipicker class change 18122018 --> One phase only per MPO)
    sliceLst = []
    for _ii, kk in enumerate(multipickerobj.slices_container):
        st = multipickerobj.slices_container[kk]
        sliceLst.append(st.select(channel="*Z")[0])
        # Name the single slice axis
        if _ii == 0:
            ax_win_1.set_title(("SLICE "+str(_ii+1)+": +/- "+kk+" sec."), {
                                             'fontweight': 'bold',
                                             'verticalalignment': 'baseline',
                                             'horizontalalignment': 'center'})
        elif _ii == 1:
            ax_win_2.set_title(("SLICE "+str(_ii+1)+": +/- "+kk+" sec."), {
                                             'fontweight': 'bold',
                                             'verticalalignment': 'baseline',
                                             'horizontalalignment': 'center'})
        # elif _ii == 2:
        #     ax_win_3.set_title(("SLICE "+str(_ii+1)+": +/- "+kk+" sec."), {
        #                                      'fontweight': 'bold',
        #                                      'verticalalignment': 'baseline',
        #                                      'horizontalalignment': 'center'})

    # Call the plot function
    ax_wavepick = plotPick_ADAPT(worktrace, pickcont, inax=ax_wavepick,
                                 plotPickPhase=kwargs["plotPickPhase"])
    ax_map = plotMap_ADAPT(opev, statDict, stidx, inax=ax_map)
    ax_win_1 = plotPick_ADAPT(sliceLst[0], pickcont, inax=ax_win_1,
                              plotPickPhase=kwargs["plotPickPhase"],
                              pickidx=0)
    ax_win_2 = plotPick_ADAPT(sliceLst[1], pickcont, inax=ax_win_2,
                              plotPickPhase=kwargs["plotPickPhase"],
                              pickidx=1)

    # Set figtitle
    fig.suptitle(figtitlehead,  fontsize=15, fontweight='bold')

    # Plot
    try:
        if kwargs["show"]:
            plt.show()
    except KeyError:
        pass
    # Save
    try:
        if kwargs["savefig"]:
            fig.savefig(kwargs["savefig"], bbox_inches='tight', dpi=310)
    except KeyError:
        pass
    #
    return fig, grid


def plot_Spectrogram(fx, tx, Z, norm=None, inax=None, show=False,
                     dbscale=True, log=False):
    """
    Simple function to plot spectrogram matrix
    if inax == None a new axis is created
    """
    cmap = mpl.cm.viridis
    if not inax:
        inax = plt.axes()
    # Plot
    if dbscale:
        zp = 10*np.log10(Z)
    else:
        zp = np.sqrt(Z)
    #
    if norm:
        im = inax.pcolor(tx, fx, zp, cmap=cmap, vmin=norm[0], vmax=norm[1])
    else:
        # im = inax.pcolormesh(tx, fx, zp, cmap=cmap, shading='gouraud')
        # alternatively use pcolor with NO shading option
        im = inax.pcolor(tx, fx, zp, cmap=cmap)

    # Accessories
    if log:
        inax.set_yscale('log')
    inax.set_ylabel('Frequency [Hz]')
    inax.set_xlabel('Time [sec]')
    cb = plt.colorbar(im, ax=inax)
    #
    if show:
        inax.axis('tight')
        plt.show()
    return inax, cb


# =====================================  NEW geographical map


def plot_map(extent=STANDARD_MAP_EXTENT,
             proj=STANDARD_CARTOPY_PROJ,
             inax=None,
             lonformat=LONGITUDE_FORMATTER,
             latformat=LATITUDE_FORMATTER,
             lonticks=(-5, 35, 5),
             latticks=(35, 60, 5)):
    """ Utility function to plot a simple geographical map

    """
    if not inax:
        mappax = plt.axes(projection=proj)
    else:
        mappax = inax

    mappax.set_extent(extent)

    # draw coastlines and borders
    mappax.add_feature(cfeature.NaturalEarthFeature("physical",
                                                    "coastline",
                                                    "50m",
                                                    edgecolor='black',
                                                    facecolor='none',
                                                    linestyle="-", lw=0.7))

    mappax.add_feature(cfeature.NaturalEarthFeature(
                                    "cultural",
                                    "admin_0_boundary_lines_land",
                                    "10m",
                                    edgecolor='black',
                                    facecolor='none',
                                    linestyle="--", lw=0.5))

    # draw meridians and parallels
    if isinstance(proj, ccrs.Mercator):
        gl = mappax.gridlines(crs=ccrs.PlateCarree(),
                              color='k', linestyle=(0, (1, 1)),
                              xlocs=range(*lonticks),
                              ylocs=range(*latticks))
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.ylabel_style = {'size': 15, 'color': 'gray'}
    return mappax, proj


def plot_circles(lon_list=[], lat_list=[], epidist=500.0,
                 inax=None, proj=STANDARD_CARTOPY_PROJ, **kwargs):
    """ Will create several cricle with specified attributes around
        a list of lon/lat input values.
        THe proj parameter will be used only if a new ax should be
        initialized.
    """
    if not lon_list or not lat_list:
        raise QE.MissingVariable("Please specify a LON and a LAT list!")
    if not len(lon_list) == len(lat_list):
        raise QE.InvalidVariable("LAT and LON list must be of the same size!")
    #
    if not inax:
        mappax = plt.axes(projection=proj)
    else:
        mappax = inax
    #
    mappax.tissot(rad_km=epidist, lons=lon_list, lats=lat_list, **kwargs)
    return mappax, proj


def plot_feature_xy(lon_list=[],
                    lat_list=[],
                    transform=ccrs.Geodetic(),
                    inax=None,
                    proj=STANDARD_CARTOPY_PROJ,
                    **kwargs):
    """ Utility function for plotting different stuff over a mappax
    """
    if not lon_list or not lat_list:
        raise QE.MissingVariable("Please specify a LON and a LAT list!")
    if not len(lon_list) == len(lat_list):
        raise QE.InvalidVariable("LAT and LON list must be of the same size!")
    #
    if not inax:
        mappax = plt.axes(projection=proj)
    else:
        mappax = inax
    #
    mappax.plot(lon_list, lat_list, transform=transform, **kwargs)
    return mappax, proj


def plot_map_comparison(ev,
                        statdict,
                        # metastatdict,
                        pickdict,
                        type="reference",
                        save=False,
                        store_path=None,
                        epidist=None,
                        show=False,
                        export_files=False,
                        annotate=False):
    """ Create a map with the plotted station with input
        Pickdict should be the final
    """
    logger.info("Creating comparison MAP")
    eqid = ev.resource_id.id
    mmag = ev.magnitudes[0].mag
    # === Extract Station NAME
    # if type.lower() in ("ref", "reference"):
    #     _sta_ref = [kk for kk, vv in statdict.items ()
    #                 if vv["meta"][eqid]["isreference"]]
    # elif type.lower() in ("auto", "automatic"):
    #     _sta_ref = [kk for kk, vv in statdict.items ()
    #                 if vv["meta"][eqid]["isautomatic"]]
    # else:
    #     raise QE.InvalidParameter("Type must be 'ref' or 'auto'")

    _sta_ref = [kk for kk, vv in statdict.items ()]

    _sta_picked = []
    for _sta, _dict in pickdict.items():
        matched = set(("P1", "*P1", "~P1")).intersection(pickdict[_sta].keys())
        if matched:
            for _pp in matched:
                if pickdict[_sta][_pp][0]["timeUTC_pick"]:
                    _sta_picked.append(_sta)
                    break

    _common = tuple(set(_sta_ref).intersection(_sta_picked))
    _only_ref = list(set(_sta_ref) - set(_common))
    _only_picker = list(set(_sta_picked) - set(_common))

    # NB: if _only_picker is empty ([]), it means that ALL the
    #        INPUT STATIONS ARE PICKED (could be a good arrival, outlier
    #        or whatever), but it's picked

    # === PLOT
    fig = plt.figure()
    evlo = ev.origins[0].longitude
    evla = ev.origins[0].latitude

    proj = ccrs.Orthographic(central_longitude=evlo, central_latitude=evla)
    map_extent = [0, 20, 41.5, 50]

    mpax, _ = plot_map(proj=proj, extent=map_extent)
    if epidist:
        mpax, _ = plot_circles([evlo], [evla], proj=proj, inax=mpax,
                               epidist=epidist, alpha=0.5, color="lightgray")

    if annotate:
        transform = ccrs.PlateCarree()._as_mpl_transform(mpax)

    # common
    if _common:
        _common_picked_lon = []
        _common_picked_lat = []
        _common_picked_name = []
        _common_NOT_picked_lon = []
        _common_NOT_picked_lat = []
        _common_NOT_picked_name = []
        for _xx in _common:
            try:
                if pickdict[_xx]["P1"][0]['timeUTC_pick']:
                    _common_picked_lon.append(statdict[_xx]['lon'])
                    _common_picked_lat.append(statdict[_xx]['lat'])
                    _common_picked_name.append(statdict[_xx]['fullname'])
                else:
                    _common_NOT_picked_lon.append(statdict[_xx]['lon'])
                    _common_NOT_picked_lat.append(statdict[_xx]['lat'])
                    _common_NOT_picked_name.append(statdict[_xx]['fullname'])
            except KeyError:
                _common_NOT_picked_lon.append(statdict[_xx]['lon'])
                _common_NOT_picked_lat.append(statdict[_xx]['lat'])
                _common_NOT_picked_name.append(statdict[_xx]['fullname'])
        #
        if _common_NOT_picked_lon and _common_NOT_picked_lat:
            mpax, _ = plot_feature_xy(
                                lon_list=_common_NOT_picked_lon,
                                lat_list=_common_NOT_picked_lat,
                                proj=proj,
                                inax=mpax, marker="o", color="g",
                                ms=7, mew=0.7, mec="k",
                                linestyle='None', label="common")
            if annotate:
                for _ss in range(0, len(_common_NOT_picked_lon)):
                    mpax.annotate(_common_NOT_picked_name[_ss],
                                  (_common_NOT_picked_lon[_ss],
                                   _common_NOT_picked_lat[_ss]),
                                  xycoords=transform,
                                  ha='right', va='top', size=10)

        if _common_picked_lon and _common_picked_lat:
            mpax, _ = plot_feature_xy(
                                lon_list=_common_picked_lon,
                                lat_list=_common_picked_lat,
                                proj=proj,
                                inax=mpax, marker="o", color="g",
                                ms=7, mew=0.7, mec="r",
                                linestyle='None', label="common picked")
            if annotate:
                for _ss in range(0, len(_common_picked_lon)):
                    mpax.annotate(_common_picked_name[_ss],
                                  (_common_picked_lon[_ss],
                                   _common_picked_lat[_ss]),
                                  xycoords=transform,
                                  ha='right', va='top', size=10)

    # ref_single
    if _only_ref:
        _onlyref_picked_lon = []
        _onlyref_picked_lat = []
        _onlyref_picked_name = []
        _onlyref_NOT_picked_lon = []
        _onlyref_NOT_picked_lat = []
        _onlyref_NOT_picked_name = []
        for _xx in _only_ref:
            try:
                if pickdict[_xx]["P1"][0]['timeUTC_pick']:
                    _onlyref_picked_lon.append(statdict[_xx]['lon'])
                    _onlyref_picked_lat.append(statdict[_xx]['lat'])
                    _onlyref_picked_name.append(statdict[_xx]['fullname'])
                else:
                    _onlyref_NOT_picked_lon.append(statdict[_xx]['lon'])
                    _onlyref_NOT_picked_lat.append(statdict[_xx]['lat'])
                    _onlyref_NOT_picked_name.append(statdict[_xx]['fullname'])
            except KeyError:
                _onlyref_NOT_picked_lon.append(statdict[_xx]['lon'])
                _onlyref_NOT_picked_lat.append(statdict[_xx]['lat'])
                _onlyref_NOT_picked_name.append(statdict[_xx]['fullname'])
        #
        if _onlyref_NOT_picked_lon and _onlyref_NOT_picked_lon:
            mpax, _ = plot_feature_xy(
                            lon_list=_onlyref_NOT_picked_lon,
                            lat_list=_onlyref_NOT_picked_lat,
                            proj=proj,
                            inax=mpax, marker="^", color="orange",
                            ms=5, mew=0.7, mec="k",
                            linestyle='None', label="only input")
            if annotate:
                for _ss in range(0, len(_onlyref_NOT_picked_lon)):
                    mpax.annotate(_onlyref_NOT_picked_name[_ss],
                                  (_onlyref_NOT_picked_lon[_ss],
                                   _onlyref_NOT_picked_lat[_ss]),
                                  xycoords=transform,
                                  ha='right', va='top', size=10)

        if _onlyref_picked_lon and _onlyref_picked_lon:
            mpax, _ = plot_feature_xy(
                            lon_list=_onlyref_picked_lon,
                            lat_list=_onlyref_picked_lat,
                            proj=proj,
                            inax=mpax, marker="^", color="orange",
                            ms=5, mew=0.7, mec="r",
                            linestyle='None', label="only input picked")
            if annotate:
                for _ss in range(0, len(_onlyref_picked_lon)):
                    mpax.annotate(_onlyref_picked_name[_ss],
                                  (_onlyref_picked_lon[_ss],
                                   _onlyref_picked_lon[_ss]),
                                  xycoords=transform,
                                  ha='right', va='top', size=10)

    # picker_single
    if _only_picker:
        _onlypicker_picked_lon = []
        _onlypicker_picked_lat = []
        _onlypicker_picked_name = []
        _onlypicker_NOT_picked_lon = []
        _onlypicker_NOT_picked_lat = []
        _onlypicker_NOT_picked_name = []
        for _xx in _only_picker:
            try:
                if pickdict[_xx]["P1"][0]['timeUTC_pick']:
                    _onlypicker_picked_lon.append(statdict[_xx]['lon'])
                    _onlypicker_picked_lat.append(statdict[_xx]['lat'])
                    _onlypicker_picked_name.append(statdict[_xx]['fullname'])
                else:
                    _onlypicker_NOT_picked_lon.append(statdict[_xx]['lon'])
                    _onlypicker_NOT_picked_lat.append(statdict[_xx]['lat'])
                    _onlypicker_NOT_picked_name.append(statdict[_xx]['fullname'])
            except KeyError:
                _onlypicker_NOT_picked_lon.append(statdict[_xx]['lon'])
                _onlypicker_NOT_picked_lat.append(statdict[_xx]['lat'])
                _onlypicker_NOT_picked_name.append(statdict[_xx]['fullname'])
        #
        if _onlypicker_NOT_picked_lon and _onlypicker_NOT_picked_lat:
            mpax, _ = plot_feature_xy(
                            lon_list=_onlypicker_NOT_picked_lon,
                            lat_list=_onlypicker_NOT_picked_lat,
                            proj=proj,
                            inax=mpax, marker="v", color="cyan",
                            ms=5, mew=0.7, mec="k",
                            linestyle='None', label="only output")
            if annotate:
                for _ss in range(0, len(_onlypicker_NOT_picked_lon)):
                    mpax.annotate(_onlypicker_NOT_picked_name[_ss],
                                  (_onlypicker_NOT_picked_lon[_ss],
                                   _onlypicker_NOT_picked_lat[_ss]),
                                  xycoords=transform,
                                  ha='right', va='top', size=10)

        if _onlypicker_picked_lon and _onlypicker_picked_lat:
            mpax, _ = plot_feature_xy(
                            lon_list=_onlypicker_picked_lon,
                            lat_list=_onlypicker_picked_lat,
                            proj=proj,
                            inax=mpax, marker="v", color="cyan",
                            ms=5, mew=0.7, mec="r",
                            linestyle='None', label="only output picked")
            if annotate:
                for _ss in range(0, len(_onlypicker_picked_lon)):
                    mpax.annotate(_onlypicker_picked_name[_ss],
                                  (_onlypicker_picked_lon[_ss],
                                   _onlypicker_picked_lat[_ss]),
                                  xycoords=transform,
                                  ha='right', va='top', size=10)

    # Epicenter
    mpax, _ = plot_feature_xy(
                           lon_list=[evlo],
                           lat_list=[evla],
                           proj=proj,
                           inax=mpax,
                           marker="*", color="red", ms=10, mew=1.5, mec="k")

    # === Title
    mpax.set_title("EQID: %s - MAG: %4.2f Mlv" % (eqid, mmag))
    mpax.legend(loc="lower left")

    # === Closing Map
    fig.tight_layout()
    if save:
        if not store_path:
            raise QE.MissingVariable(("SAVE is set to True, " +
                                      "Please provide a storing path!"))
        #
        plt.savefig(store_path, facecolor='w', edgecolor='w',
                    transparent=True, format='pdf')
    if show:
        plt.show()
    #
    return fig, mpax


def plot_map_bait(ev,
                  statdict,
                  metastatdict,
                  pickdict,
                  type="reference",
                  save=False,
                  store_path=None,
                  epidist=None,
                  show=False,
                  export_files=False,
                  annotate=False):
    """ Create a map with the plotted station with picked or unpicked
        BAIT picker
    """
    logger.info("Creating bait MAP")
    eqid = ev.resource_id.id
    mmag = ev.magnitudes[0].mag
    # === Extract Station NAME
    # _sta_all = len(metastatdict[eqid].keys())
    if type.lower() in ("ref", "reference"):
        _sta_ref = [_sta for _sta, _sta_info in metastatdict[eqid].items()
                    if metastatdict[eqid][_sta]["isreference"]]
    elif type.lower() in ("auto", "automatic"):
        _sta_ref = [_sta for _sta, _sta_info in metastatdict[eqid].items()
                    if metastatdict[eqid][_sta]["isautomatic"]]
    else:
        raise QE.InvalidParameter("Type must be 'ref' or 'auto'")

    _sta_down = [_sta for _sta, _sta_info in metastatdict[eqid].items()
                 if metastatdict[eqid][_sta]["isdownloaded"]]

    #

    _sta_picked_bait = []
    _sta_not_picked_bait = []
    for _sta, _dict in pickdict.items():
        for _kk in pickdict[_sta].keys():
            if (_kk[0:4].lower() == "bait" and
               pickdict[_sta][_kk][0]["timeUTC_pick"]):
                _sta_picked_bait.append(_sta)
            else:
                _sta_not_picked_bait.append(_sta)

    # === PLOT
    fig = plt.figure()
    evlo = ev.origins[0].longitude
    evla = ev.origins[0].latitude

    proj = ccrs.Orthographic(central_longitude=evlo, central_latitude=evla)
    map_extent = [0, 20, 41.5, 50]

    mpax, _ = plot_map(proj=proj, extent=map_extent)
    if epidist:
        mpax, _ = plot_circles([evlo], [evla], proj=proj, inax=mpax,
                               epidist=epidist, alpha=0.5, color="lightgray")
    if annotate:
        transform = ccrs.PlateCarree()._as_mpl_transform(mpax)

    for _sta, _dict in pickdict.items():
        _plot_attributes = {}

        if _sta in _sta_picked_bait:
            _plot_attributes["color"] = "teal"
        else:
            _plot_attributes["color"] = "lightgray"

        if _sta in _sta_down:
            _plot_attributes["markeredgecolor"] = "green"
        else:
            _plot_attributes["markeredgecolor"] = "red"

        if _sta in _sta_ref:
            _plot_attributes["marker"] = "^"
        else:
            _plot_attributes["marker"] = "s"

        _plot_attributes['ms'] = 7
        _plot_attributes['mew'] = 0.7

        mpax, _ = plot_feature_xy(
                            lon_list=[statdict[_sta]['lon']],
                            lat_list=[statdict[_sta]['lat']],
                            proj=proj,
                            inax=mpax, linestyle='None',
                            **_plot_attributes)
        if annotate:
            mpax.annotate(statdict[_sta]['fullname'],
                          (statdict[_sta]['lon'],
                           statdict[_sta]['lat']),
                          xycoords=transform,
                          ha='right', va='top', size=10)

    # Epicenter
    mpax, _ = plot_feature_xy(
                           lon_list=[evlo],
                           lat_list=[evla],
                           proj=proj,
                           inax=mpax,
                           marker="*", color="red", ms=10, mew=1.5, mec="k")

    # === Label DEF
    mpax.plot([], [], linestyle='None',
              marker="^", mec="gold", label="ossi")

    # linestyle=None
    legend_elements = [
        Line2D([], [], marker="s", mec="black", color='w',
               label='Only ADAPT'),
        Line2D([], [], marker="^", mec="black", color='w',
               label='Also in SC3'),
        Patch(facecolor='teal', edgecolor='teal', label='Valid Pick BAIT'),
        Patch(facecolor='lightgray', edgecolor='lightgray', label='BAIT fails'),
        Patch(facecolor='w', edgecolor='green', label='Station Downloaded'),
        Patch(facecolor='w', edgecolor='red', label="Missing Station's data")
        ]

    # === Title
    mpax.set_title("EQID: %s - MAG: %4.2f Mlv" % (eqid, mmag))
    mpax.legend(handles=legend_elements, loc="lower left")

    # === Closing Map
    fig.tight_layout()
    if save:
        if not store_path:
            raise QE.MissingVariable(("SAVE is set to True, " +
                                      "Please provide a storing path!"))
        #
        plt.savefig(store_path, facecolor='w', edgecolor='w',
                    transparent=True, format='pdf')
    if show:
        plt.show()
    #
    return fig, mpax


def plot_event_section(opev, pickdict, adapt_inventory, opstr,
                       pick_stat_alias=True, only_picked_traces=False,
                       pick_tag_display=["VEL_P", "P1"],
                       create_map=False, special_stations=None,
                       plot_station_name=False,
                       min_epicentral_dist=0.0, max_epicentral_dist=250.0,
                       min_time_from_origin=0.0, max_time_from_origin=150.0):
    """This function will create an event section figure

    special_stations must be a list/tuple

    # --- CREATED FOR EGU 2019
    fig = plt.figure(figsize=(16, 9))
    grid = plt.GridSpec(2, 4, wspace=0.4, hspace=0.3)

    # Create SubplotAxis to be passed
    ax_wavepick = fig.add_subplot(grid[0, 0:2])
    ax_map = fig.add_subplot(grid[0, 2:], projection=STANDARD_CARTOPY_PROJ)
    ax_win_1 = fig.add_subplot(grid[1, 0:2])
    ax_win_2 = fig.add_subplot(grid[1, 2:])


            _, _ = plot_map(proj=proj, extent=map_extent, inax=axd[str(figidx)])
            plot_feature_xy(lon_list=[evlo], lat_list=[evla],
                            inax=axd[str(figidx)],
                            marker="*",
                            markeredgecolor="black",
                            markerfacecolor="red",
                            markersize=8.5)

            plot_feature_xy(lon_list=[inventory[_elem[0]]['lon']],
                            lat_list=[inventory[_elem[0]]['lat']],
                            inax=axd[str(figidx)],
                            marker="^",
                            markeredgecolor="black",
                            markerfacecolor="green",
                            markersize=6.5)

    """

    # -- Expand Event
    eqid = opev.resource_id.id
    mmag = opev.magnitudes[0].mag
    evlo = opev.origins[0].longitude
    evla = opev.origins[0].latitude
    evdp = opev.origins[0].depth*MT
    evtm = opev.origins[0].time

    # -- Crete fig
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('%s - Depth: %5.1f km - MLv: %3.1f\n %s' % (
                    eqid, evdp, mmag, evtm), fontsize=13)

    grid = plt.GridSpec(1, 5, wspace=0.2)
    axm = None  # Needed for the function's return
    axm_statLon = []
    axm_statLat = []
    axm_statLon_ALL = []
    axm_statLat_ALL = []

    if create_map:
        axw = fig.add_subplot(grid[0, 0:4])
        axw.set_xlabel("Epicentral Distance (km)")
        axw.set_ylabel("Time from event-origin (s)")
        #
        axm = fig.add_subplot(grid[0, 4], projection=STANDARD_CARTOPY_PROJ)
    else:
        axw = fig.add_subplot(grid[0, 0:])
        axw.set_xlabel("Epicentral Distance (km)")
        axw.set_ylabel("Time from event-origin (s)")

    opstr = opstr.select(channel="*Z")
    # ======== Real Plot  --> WAVES
    epidist_statname = []
    for tr in opstr:
        # statname = tr.stats.station  # QUAKE
        statname = tr.stats.network+"."+tr.stats.station  # ADAPT

        # =================================  Chek if picked
        if only_picked_traces:
            statpick = statname
            if pick_stat_alias:
                try:
                    statpick = adapt_inventory[statpick]['alias']
                except:
                    import pdb; pdb.set_trace()
            #
            try:
                _ = pickdict[statpick]
            except KeyError:
                # Missing Pick at stations
                continue
        # =================================================

        statobj = adapt_inventory.getStat(statname, is_alias=False)
        axm_statLon_ALL.append(statobj['lon'])
        axm_statLat_ALL.append(statobj['lat'])

        epidist_km = calcEpiDist(
                            evla, evlo,
                            statobj['lat'],
                            statobj['lon'],
                            outdist='km')

        # Put a limit on Distance
        if (epidist_km < min_epicentral_dist or
           epidist_km > max_epicentral_dist):
            continue

        # Shift origin
        if tr.stats.starttime-evtm > 0:
            # traccia ritardo, aggiungi differtenza
            tt = tr.times()
            tdelta = tr.stats.starttime-evtm
            tt += tdelta

        elif evtm - tr.stats.starttime > 0:
            # traccia anticipo, sposta all'origin time
            tr.trim(evtm, tr.stats.endtime)
            tt = tr.times()
        else:
            tt = tr.times()

        # ------------ Plot Waves
        tr.data = normalizeTrace(tr.data, rangeVal=[epidist_km-1, epidist_km+1])
        if tr.data[0] >= epidist_km:
            tr.data -= abs(tr.data[0] - epidist_km)
        elif tr.data[0] < epidist_km:
            tr.data += abs(tr.data[0] - epidist_km)
        axw.plot(tr.data, tt, ls='-', lw=0.8, c='black', alpha=0.7)

        # ------------ Use station names
        epidist_statname.append((epidist_km, tr.stats.station))

        # ------------ PlotPicks - predicted
        statpick = statname
        if pick_stat_alias:
            statpick = adapt_inventory[statname]['alias']

        try:
            pickUTC = get_pick_slice(pickdict, statpick, searchkey="Pre")[0][1]
            picksec = pickUTC - evtm
            axw.plot(epidist_km, picksec,
                     marker='_',
                     markersize=22,
                     markeredgecolor='gray',
                     markeredgewidth=2.5,
                     alpha=0.9)
        except QE.MissingVariable:
            # Missing Pick at stations
            pass

        # ------------ PlotPicks - P-phase
        try:
            for pt in pickdict[statpick].keys():
                tmppk = pickdict[statpick][pt][0]['timeUTC_pick']
                if pt in pick_tag_display and tmppk:
                    picksec = tmppk - evtm
                    if special_stations and statname in special_stations:
                        axw.plot(epidist_km, picksec,
                                 marker='_',
                                 markersize=22,
                                 markeredgecolor='blue',
                                 markeredgewidth=2.5,
                                 alpha=0.9)
                    else:
                        axw.plot(epidist_km, picksec,
                                 marker='_',
                                 markersize=22,
                                 markeredgecolor='darkorange',
                                 markeredgewidth=2.2,
                                 alpha=0.9)
                    # Collect coordinates for possible map
                    axm_statLon.append(statobj['lon'])
                    axm_statLat.append(statobj['lat'])
        except KeyError:
            # Missing Pick at stations
            pass
    #
    axw.set_xlim([min_epicentral_dist, max_epicentral_dist])
    axw.set_ylim([min_time_from_origin, max_time_from_origin])
    #
    if plot_station_name:
        _xtick = tuple([de[0] for de in epidist_statname])
        _staname = tuple([de[1] for de in epidist_statname])
        axw.set_xticks(_xtick)
        axw.set_xticklabels(_staname)
        axw.tick_params(axis='x', rotation=90)

    # ======== Real Plot  --> MAP
    if create_map:
        # 300 km ~= 2.8 degree
        mymapextent = [evlo-3, evlo+3, evla-2.5, evla+2.5]
        _, _ = plot_map(proj=STANDARD_CARTOPY_PROJ, extent=mymapextent,
                        inax=axm)
        # Stations ALL
        plot_feature_xy(lon_list=axm_statLon_ALL,
                        lat_list=axm_statLat_ALL,
                        inax=axm,
                        marker="^",
                        markeredgecolor="black",
                        markerfacecolor="lightgray",
                        linestyle='None',
                        markersize=4.5)

        # Event
        plot_feature_xy(lon_list=[evlo, ], lat_list=[evla, ],
                        inax=axm,
                        marker="*",
                        markeredgecolor="black",
                        markerfacecolor="red",
                        linestyle='None',
                        markersize=8)

        # Stations Used
        if axm_statLon and axm_statLat:
            plot_feature_xy(lon_list=axm_statLon,
                            lat_list=axm_statLat,
                            inax=axm,
                            marker="^",
                            markeredgecolor="black",
                            markerfacecolor="green",
                            linestyle='None',
                            markersize=6)

    return fig, (axw, axm)


# TIPS (for further development)
# mapevent = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)

###
# import cartopy.crs as ccrs
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import matplotlib.pyplot as plt


# def main():
#     fig = plt.figure(figsize=(8, 10))

#     # Label axes of a Plate Carree projection with a central longitude of 180:
#     ax1 = fig.add_subplot(2, 1, 1,
#                           projection=ccrs.PlateCarree(central_longitude=180))
#     ax1.set_global()
#     ax1.coastlines()
#     ax1.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
#     ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
#     lon_formatter = LongitudeFormatter(zero_direction_label=True)
#     lat_formatter = LatitudeFormatter()
#     ax1.xaxis.set_major_formatter(lon_formatter)
#     ax1.yaxis.set_major_formatter(lat_formatter)

#     # Label axes of a Mercator projection without degree symbols in the labels
#     # and formatting labels to include 1 decimal place:
#     ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.Mercator())
#     ax2.set_global()
#     ax2.coastlines()
#     ax2.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
#     ax2.set_yticks([-78.5, -60, -25.5, 25.5, 60, 80], crs=ccrs.PlateCarree())
#     lon_formatter = LongitudeFormatter(number_format='.1f',
#                                        degree_symbol='',
#                                        dateline_direction_label=True)
#     lat_formatter = LatitudeFormatter(number_format='.1f',
#                                       degree_symbol='')
#     ax2.xaxis.set_major_formatter(lon_formatter)
#     ax2.yaxis.set_major_formatter(lat_formatter)

#     plt.show()


# if __name__ == '__main__':
#     main()
