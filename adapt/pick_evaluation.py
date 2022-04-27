import os
import sys
import logging
import numpy as np
#
from adapt.processing import normalizeTrace

logger = logging.getLogger(__name__)


# --------------------------------------------- Orchestrator
def doPickEvaluation(wt, pick, **kwargs):
    """
    This function will organize the call to the different
    methods for pick-evaluation.

    To properly debug/print results every test should return
    a BOOL (test result) and a DICT containing keys:
        - "message"
        - "value" ecc...

    RETURN:
        finaljudge: bool
    """
    testsResultsBool, testsResultsDict = [], {}
    # sorting in alphabetical order the testfunctions
    sortedkeys = sorted(kwargs, key=str.lower)
    for xx in sortedkeys:
        logger.info("%s - %r" % (xx, kwargs[xx]))
        testFunction = getattr(sys.modules[__name__], xx)
        ts, td = testFunction(wt.copy(), pick, **kwargs[xx])
        testsResultsBool.append(ts)
        testsResultsDict[xx] = td
    # Print results
    for xx in sortedkeys:
        logger.info("%s --> %s" % (testsResultsDict[xx]["message"],
                                   str(testsResultsDict[xx]["value"])))
    # Return results
    if False in testsResultsBool:
        logger.info("Pick not accepted")
        return False, testsResultsDict
    else:
        logger.info("Pick ACCEPTED")
        return True, testsResultsDict


# --------------------------------------------- Charachteristic Function
def createCF(inarray):
    ''' Simple method to create the carachteristic function of BaIt
        picking algorithm
    '''
    outarray = abs(inarray)**2  # original no square elevation
    outarray = normalizeTrace(outarray, rangeVal=[0, 1])
    return outarray


# --------------------------------------------- Functions
def SignalAmp(wt, pickUTC, timewin=None, thr_par_1=None):
    """
    This test evaluate the maximum amplitude of the first window
    after the pick and compare it to a threshold given by user.
    INPUT:
        - workTrace (obspy.Trace obj)
        - pickUC    (UTCDateTime object)
        - timewin   (float) interval in sec. after pick to evaluate
        - thr_par_1 (float)
    OUTPUT
        - bool (True/False)
        - outDict (dict) containing all the info necessary
    """
    outDict = {}
    wt.data = createCF(wt.data)
    wt.trim(pickUTC, pickUTC + timewin)
    # ------ Out + Log
    if wt.data.max() <= thr_par_1:
        outDict['result'] = False
        outDict['message'] = ' ' * 4 + 'False'
        outDict['value'] = wt.data.max()
        return False, outDict
    else:
        outDict['result'] = True
        outDict['message'] = ' ' * 4 + 'True'
        outDict['value'] = wt.data.max()
        return True, outDict


def SignalSustain(wt, pickUTC, timewin=None, timenum=None, snratio=None):
    """
    This test evaluate the mean value of signal windows in comparison
    with the noise window before the pick. The ratio should be
    higher than a threshold given by user.
    INPUT:
        - workTrace (obspy.Trace obj)
        - iteration number [to reach the proper pick]
        - config py file (BaIt_Config)
        - File ID (already opened) for logs
    OUTPUT
        - bool (True/False)
    """
    outDict = {}
    wt.data = createCF(wt.data)
    # Create slice
    Noise = wt.slice(pickUTC - timewin, pickUTC)
    WINDOWING = []  # MB list of numpy array
    for num in range(timenum):
        Signal = wt.slice(pickUTC + (num * timewin),
                          pickUTC + ((num + 1) * timewin))
        WINDOWING.append(Signal.data)
    # Check
    boolBOX = [window.mean() >= snratio * Noise.data.mean()
               for window in WINDOWING]
    outDict['value'] = [window.mean() / Noise.data.mean()
                        for window in WINDOWING]
    if False in boolBOX:
        outDict['result'] = False
        outDict['message'] = ' ' * 4 + 'False'
        return False, outDict
    else:  # MB se sono tutti Veri (sopra la soglia di SNR)
        outDict['result'] = True
        outDict['message'] = ' ' * 4 + 'True'
        return True, outDict


def LowFreqTrend(wt, it, CMN, LOG, timewin, conf=0.95):
    """
    This method should help avoiding mispicks due
    to the so-called filter effect by recognizing trends (pos or negative)

    return False if trend found --> bad pick
    """
    tfn = sys._getframe().f_code.co_name

    # ------ WORK
    # wt.data=BC.createCF(wt.data)
    wt.trim(wt.stats['BaIt_DICT'][str(it)]['baerpick'],
            wt.stats['BaIt_DICT'][str(it)]['baerpick'] + timewin)

    # asign=np.sign(wt.data)
    asign = np.sign(np.diff(wt.data))
    unique, counts = np.unique(asign, return_counts=True)
    dsign = dict(zip(unique, counts))
    #
    for key in (-1.0, 1.0):
        if key in dsign and dsign[key]:
            pass
        else:
            dsign[key] = 0
    # ------ Out + Log
    if dsign[1.0] / len(asign) >= conf or dsign[-1.0] / len(asign) >= conf:
        LOG.write((' ' * 4 + 'FALSE  %s: Pos. %5.2f  -  Neg. %5.2f  [%5.2f]' + os.linesep) %
                  (tfn, dsign[1.0] / len(asign), dsign[-1.0] / len(asign), conf))
        return False
    else:
        LOG.write((' ' * 4 + 'TRUE   %s: Pos. %5.2f  -  Neg. %5.2f  [%5.2f]' + os.linesep) %
                  (tfn, dsign[1.0] / len(asign), dsign[-1.0] / len(asign), conf))
        return True


# --------------------------------------------- POST validation

def FilterEffectTrend(
        wt,
        validPicks,
        CMN,
        LOG,
        deltapicks,
        timewin,
        conf=0.95):
    """
    Evaluating negatively if 2 picks are closed (deltapicks) and a pos/neg trend
    is present for timewin" seconds after the first pick
    *** NB: usually timewin < deltapicks !!
    """
    tfn = sys._getframe().f_code.co_name
    functSwitch = False
    # Check Valid picks
    if len(validPicks) <= 1:
        LOG.write(
            (' ' *
             4 +
             '%s  -> only 1 valid pick stored!' +
             os.linesep) %
            tfn)
        return wt           # No test will be suited
    # NEXT:
    # 1 to length() because we make difference betweenpicks
    # so we start with the second ---> DOENS'T REPRESENT ITERATION
    for xx in range(1, len(validPicks)):
        mydelta = validPicks[xx][0] - validPicks[xx - 1][0]
        if mydelta <= deltapicks:
            myarray = wt.slice(wt.stats['BaIt_DICT'][str(validPicks[xx - 1][1])]['baerpick'],
                               wt.stats['BaIt_DICT'][str(validPicks[xx - 1][1])]['baerpick'] + timewin).data

            asign = np.sign(np.diff(myarray))
            unique, counts = np.unique(asign, return_counts=True)
            dsign = dict(zip(unique, counts))
            # Complete Dictionary (avoiding missing key)
            for key in (-1.0, 1.0):
                if key in dsign and dsign[key]:
                    pass
                else:
                    dsign[key] = 0
            # Test
            if dsign[1.0] / len(asign) >= conf or dsign[-1.0] / \
                    len(asign) >= conf:
                # Trend FOUND
                LOG.write((' ' * 4 + 'FALSE-Positive @ Iteration: %d - %s: Pos. %5.2f  /  Neg. %5.2f  [%5.2f]' + os.linesep) %
                          (validPicks[xx - 1][1], tfn, dsign[1.0] / len(asign), dsign[-1.0] / len(asign), conf))
                wt.stats['BaIt_DICT'][str(
                    validPicks[xx - 1][1])]['evaluate'] = False
                functSwitch = True
    #
    if not functSwitch:
        # No Filter effect found in picks --> bumps to LOG
        LOG.write((' ' * 4 + '%s  -> all good!' + os.linesep) % tfn)
    return wt
