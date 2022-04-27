import os
import logging
import pandas as pd
import numpy as np
import adapt.errors as QE
import adapt.utils as QU
#

logger = logging.getLogger(__name__)

# --------------------------------------------------


def _percentage(cnt, tot, precision="{:6.2f}", out_type="str"):
    """
    If you don't get it, you're stupid ;)

    out_type="str"/"float"

    return a str object
    """
    perc = (cnt*100.0)/tot
    if out_type.lower() == "str":
        return precision.format(perc)
    elif out_type.lower() == "float":
        if isinstance(precision, int):
            return round(perc, precision)
        else:
            raise QE.InvalidType()
    else:
        raise QE.InvalidParameter()


def create_picker_percentage_dataframe_single(
                                    eqid,
                                    pickdict,
                                    statdict,
                                    metastatdict,
                                    storedir,
                                    slices=(1, 2, 3),
                                    mpsliceTag=("MP1",),
                                    pickersTag=("BK", "MyAIC", "KURT", "FP"),
                                    outfilename="PickTable.csv",
                                    # MB: case sensitive (next line)
                                    phases=["P", "Pg", "Pn", "P1"],
                                    savedf=True):
    """ This method will return a csv file that could be used in R
        for furher plotting

        NB: the PERCENTAGE field referes to the TOTCOUNT field, not the
          total amount of reference pick

    """

    # --------------- InnerFunctions
    # ------------------------------

    def _check_reference_dict(ddict, phasename, classnum):
        """ Recursively check and create the nested dict, for REFERENCE.
            phasename, classnum MUST BE string
        """
        if phasename not in ddict.keys():
            ddict[phasename] = {}
        if classnum not in ddict[phasename].keys():
            ddict[phasename][classnum] = 0
        #
        return ddict

    def _check_final_dict(indict, dphase, dclass, dpicker, dmptag, dwin):
        """ Recursively check and create the nested dict, for FINAL.
            dphase, dclass, dpicker, dmultipicktag, dwin MUST BE string

            NB: the picked count will be given by len(deltalist)
        """
        delta_container = {}
        delta_container['delta'] = []
        delta_container['pick_and_unpick_count'] = 0
        delta_container['tot_reference_count'] = 0
        delta_container['mean'] = None
        delta_container['std'] = None
        delta_container['percent'] = None
        #
        if dphase not in indict.keys():
            indict[dphase] = {}

        if dclass not in indict[dphase].keys():
            indict[dphase][dclass] = {}

        if dpicker not in indict[dphase][dclass].keys():
            indict[dphase][dclass][dpicker] = {}

        if dmptag not in indict[dphase][dclass][dpicker].keys():
            indict[dphase][dclass][dpicker][dmptag] = {}

        if dwin not in indict[dphase][dclass][dpicker][dmptag].keys():
            indict[dphase][dclass][dpicker][dmptag][dwin] = delta_container
        #
        return indict

    # --------------- CHECKS
    # ----------------------
    if (not isinstance(mpsliceTag, (list, tuple)) or
       not isinstance(pickersTag, (list, tuple)) or
       not isinstance(phases, (list, tuple))):
        logger.error("mpsliceTag - pickersTag - phases " +
                     "MUST BE list or tuple object!")
        raise QE.InvalidVariable()

    # --- Initialize the DataFrame
    header_columns = ['EVENTID',
                      'PHASENAME',
                      'PHASECLASS',
                      'PICKER',
                      'MULTIPICKTAG',
                      'SLICE',
                      # MB: len of delta
                      'COUNT',
                      # pick and unpicked data
                      'TOTCOUNT',
                      # total reference count (phase,class)
                      'REFERENCECOUNT',
                      'MEAN',
                      'STD',
                      'PERCENTAGE']
    df_table = pd.DataFrame(columns=header_columns)

    # Initialize the counters + variables
    eqid = pickdict.eqid
    ref_stat_downloaded = 0
    ref_stat_not_downloaded = 0
    noref_stat_downloaded = 0
    noref_stat_not_downloaded = 0
    big_final = {}
    reference_total = {}
    #

    logger.info("Creating %s PERCENTAGE data frame ..." % eqid)

    for _stat in pickdict:
        # ------------------------------- Check STATIONS in DICTS
        # -------------------------------------------------------
        if _stat not in statdict.keys():
            logger.error("Missing %s key in STATDICT" % _stat)
            raise QE.CheckError
        if _stat not in metastatdict[eqid].keys():
            logger.error("Missing %s key in METASTATDICT" % _stat)
            raise QE.CheckError

        # ------------------------------- Switch STATIONS
        # -----------------------------------------------
        # We don't want NON reference station
        if (metastatdict[eqid][_stat]['isreference'] and
           metastatdict[eqid][_stat]['isdownloaded']):
            ref_stat_downloaded += 1
            # =================================================== WORK
            # ===================================================
            # ===================================================
            full_ref_tag = metastatdict[eqid][_stat]["reference_picks"][0][0]
            ref_phase_name = full_ref_tag.split("_")[-1]
            ref_phase_time = (pickdict[_stat]
                                      [full_ref_tag][0]
                                      ["timeUTC_pick"])
            ref_phase_class = (pickdict[_stat]
                                       [full_ref_tag][0]
                                       ["pickclass"])

            if ref_phase_name not in phases:
                # MB: case sensitive
                continue
                # MB: to the next station,
                #     the reference time not of interest
            else:
                # MB increase the counter
                _check_reference_dict(reference_total,
                                      ref_phase_name,
                                      ref_phase_class)
                reference_total[ref_phase_name][ref_phase_class] += 1

            #
            #  BODY
            #

            for _phase in pickdict[_stat]:
                multipick_tag = _phase.split("_")[-1]
                picker_tag = _phase.split("_")[0]
                #
                if (multipick_tag not in mpsliceTag or
                   picker_tag.lower() in ("reference", "predicted")):
                    continue             # MB: to the NEXT PHASE
                #
                for _xx, _pd in enumerate(pickdict[_stat][_phase]):
                    _slice = _xx + 1
                    if _slice not in slices:
                        continue         # MB: to the next slicepick

                    # Check DICT keys
                    _check_final_dict(big_final,
                                      ref_phase_name,
                                      ref_phase_class,
                                      picker_tag,
                                      multipick_tag,
                                      _slice)

                    # Check that it has picked
                    if pickdict[_stat][_phase][_xx]["timeUTC_pick"]:
                        _delta = (
                              pickdict[_stat][_phase][_xx]["timeUTC_pick"] -
                              ref_phase_time)
                        try:
                            # append delta
                            (big_final[ref_phase_name]
                                      [ref_phase_class]
                                      [picker_tag]
                                      [multipick_tag]
                                      [_slice]['delta'].append(_delta)
                             )
                            # increase pick_and_unpick_count
                            (big_final[ref_phase_name]
                                      [ref_phase_class]
                                      [picker_tag]
                                      [multipick_tag]
                                      [_slice]['pick_and_unpick_count']) += 1
                        except KeyError:
                            logger.error("WEIRD: All keys should be created " +
                                         "already, check why!")
                            raise QE.CheckError()
                    else:
                        try:
                            # increase totalcount
                            (big_final[ref_phase_name]
                                      [ref_phase_class]
                                      [picker_tag]
                                      [multipick_tag]
                                      [_slice]['pick_and_unpick_count']) += 1
                        except KeyError:
                            logger.error("WEIRD: All keys should be created " +
                                         "already, check why!")
                            raise QE.CheckError()

                # BAIT RELATED CHECK

        elif (metastatdict[eqid][_stat]['isreference'] and
              not metastatdict[eqid][_stat]['isdownloaded']):
            ref_stat_not_downloaded += 1
            continue
        elif (not metastatdict[eqid][_stat]['isreference'] and
              not metastatdict[eqid][_stat]['isdownloaded']):
            noref_stat_not_downloaded += 1
            continue
        elif (not metastatdict[eqid][_stat]['isreference'] and
              metastatdict[eqid][_stat]['isdownloaded']):
            noref_stat_downloaded += 1
            continue
        else:
            logger.error("Unusual switch situation: Station %s" % _stat)
            raise QE.CheckError

    # =================================================== Write DF, CSV
    # ===================================================
    # ===================================================
    # ----- Populate DataFrame
    for pname in big_final.keys():
        for clnum in big_final[pname].keys():
            for picker in big_final[pname][clnum].keys():
                for mpt in big_final[pname][clnum][picker].keys():
                    for win in big_final[pname][clnum][picker][mpt].keys():
                        # Working on delta_container
                        # -------------------------
                        tmp_work_dict = (
                            big_final[pname][clnum][picker][mpt][win])

                        # -------------------------
                        # Stuff to store anyway
                        tmp_work_dict['tot_reference_count'] = (
                                              reference_total[pname][clnum])

                        if tmp_work_dict['delta']:
                            tmp_work_dict['percent'] = round(
                                    (len(tmp_work_dict['delta']) * 100) /
                                    tmp_work_dict['pick_and_unpick_count'], 3)
                            if len(tmp_work_dict['delta']) == 1:
                                tmp_work_dict['mean'] = round(
                                            np.mean(tmp_work_dict['delta']), 3)
                                tmp_work_dict['std'] = None
                            elif len(tmp_work_dict['delta']) >= 2:
                                tmp_work_dict['mean'] = round(
                                            np.mean(tmp_work_dict['delta']), 3)
                                tmp_work_dict['std'] = round(
                                            np.std(tmp_work_dict['delta']), 3)

                        else:
                            tmp_work_dict['mean'] = None
                            tmp_work_dict['std'] = None
                            tmp_work_dict['percent'] = 0.0

                        # Appending
                        df_table = df_table.append({
                                    'EVENTID': eqid,
                                    'PHASENAME': pname,
                                    'PHASECLASS': clnum,
                                    'PICKER': picker,
                                    'MULTIPICKTAG': mpt,
                                    'SLICE': win,
                                    # MB: len of delta
                                    'COUNT': len(tmp_work_dict['delta']),
                                    # pick and unpicked data
                                    'TOTCOUNT': (
                                      tmp_work_dict['pick_and_unpick_count']),
                                    # total reference count (phase,class)
                                    'REFERENCECOUNT': (
                                      tmp_work_dict['tot_reference_count']),
                                    'MEAN': tmp_work_dict['mean'],
                                    'STD': tmp_work_dict['std'],
                                    'PERCENTAGE': tmp_work_dict['percent']},
                                    ignore_index=True)
    # ----- Printing OUT
    if savedf:
        df_table.to_csv(os.sep.join([storedir, outfilename]),
                        sep=',',
                        index=False,
                        na_rep="NA")


def create_picker_percentage_dataframe_complete(
                                    eqid,
                                    pickdictALL,
                                    statdict,
                                    metastatdict,
                                    storedir,
                                    slices=(1, 2, 3),
                                    mpsliceTag=("MP1",),
                                    pickersTag=("BK", "MyAIC", "KURT", "FP"),
                                    outfilename="PickTable.csv",
                                    # MB: case sensitive (next line)
                                    phases=["P", "Pg", "Pn", "P1"],
                                    savedf=True):
    """ This method will return a csv file that could be used in R
        for furher plotting

        NB: the PERCENTAGE field referes to the TOTCOUNT field, not the
          total amount of reference pick

    """

    # --------------- InnerFunctions
    # ------------------------------

    def _check_reference_dict(ddict, phasename, classnum):
        """ Recursively check and create the nested dict, for REFERENCE.
            phasename, classnum MUST BE string
        """
        if phasename not in ddict.keys():
            ddict[phasename] = {}
        if classnum not in ddict[phasename].keys():
            ddict[phasename][classnum] = 0
        #
        return ddict

    def _check_final_dict(indict, dphase, dclass, dpicker, dmptag, dwin):
        """ Recursively check and create the nested dict, for FINAL.
            dphase, dclass, dpicker, dmultipicktag, dwin MUST BE string

            NB: the picked count will be given by len(deltalist)
        """
        delta_container = {}
        delta_container['delta'] = []
        delta_container['pick_and_unpick_count'] = 0
        delta_container['tot_reference_count'] = 0
        delta_container['mean'] = None
        delta_container['std'] = None
        delta_container['percent'] = None
        #
        if dphase not in indict.keys():
            indict[dphase] = {}

        if dclass not in indict[dphase].keys():
            indict[dphase][dclass] = {}

        if dpicker not in indict[dphase][dclass].keys():
            indict[dphase][dclass][dpicker] = {}

        if dmptag not in indict[dphase][dclass][dpicker].keys():
            indict[dphase][dclass][dpicker][dmptag] = {}

        if dwin not in indict[dphase][dclass][dpicker][dmptag].keys():
            indict[dphase][dclass][dpicker][dmptag][dwin] = delta_container
        #
        return indict

    # --------------- CHECKS
    # ----------------------
    if not isinstance(pickdictALL, dict):
        logger.error("pickdict MUST BE a dictionary with eqid as key")
        raise QE.InvalidVariable()

    if (not isinstance(mpsliceTag, (list, tuple)) or
       not isinstance(pickersTag, (list, tuple)) or
       not isinstance(phases, (list, tuple))):
        logger.error("mpsliceTag - pickersTag - phases " +
                     "MUST BE list or tuple object!")
        raise QE.InvalidVariable()

    logger.info("Creating ALL EVENTS PERCENTAGE data frame ...")

    # --- Initialize the DataFrame
    header_columns = ['EVENTID',
                      'PHASENAME',
                      'PHASECLASS',
                      'PICKER',
                      'MULTIPICKTAG',
                      'SLICE',
                      # MB: len of delta
                      'COUNT',
                      # pick and unpicked data
                      'TOTCOUNT',
                      # total reference count (phase,class)
                      'REFERENCECOUNT',
                      'MEAN',
                      'STD',
                      'PERCENTAGE']
    df_table = pd.DataFrame(columns=header_columns)

    # Initialize the counters + variables
    ref_stat_downloaded = 0
    ref_stat_not_downloaded = 0
    noref_stat_downloaded = 0
    noref_stat_not_downloaded = 0
    big_final = {}
    reference_total = {}

    for event in pickdictALL:
        pickdict = pickdictALL[event]
        eqid = pickdict.eqid

        for _stat in pickdict:
            # ------------------------------- Check STATIONS in DICTS
            # -------------------------------------------------------
            if _stat not in statdict.keys():
                logger.error("Missing %s key in STATDICT" % _stat)
                raise QE.CheckError
            if _stat not in metastatdict[eqid].keys():
                logger.error("Missing %s key in METASTATDICT" % _stat)
                raise QE.CheckError

            # ------------------------------- Switch STATIONS
            # -----------------------------------------------
            # We don't want NON reference station
            if (metastatdict[eqid][_stat]['isreference'] and
               metastatdict[eqid][_stat]['isdownloaded']):
                ref_stat_downloaded += 1
                # =================================================== WORK
                # ===================================================
                # ===================================================
                full_ref_tag = (metastatdict[eqid][_stat]
                                            ["reference_picks"][0][0])
                ref_phase_name = full_ref_tag.split("_")[-1]
                ref_phase_time = (pickdict[_stat]
                                          [full_ref_tag][0]
                                          ["timeUTC_pick"])
                ref_phase_class = (pickdict[_stat]
                                           [full_ref_tag][0]
                                           ["pickclass"])

                if ref_phase_name not in phases:
                    # MB: case sensitive
                    continue
                    # MB: to the next station,
                    #     the reference time not of interest
                else:
                    # MB increase the counter
                    _check_reference_dict(reference_total,
                                          ref_phase_name,
                                          ref_phase_class)
                    reference_total[ref_phase_name][ref_phase_class] += 1

                #
                #  BODY
                #

                for _phase in pickdict[_stat]:
                    multipick_tag = _phase.split("_")[-1]
                    picker_tag = _phase.split("_")[0]
                    #
                    if (multipick_tag not in mpsliceTag or
                       picker_tag.lower() in ("reference", "predicted")):
                        continue             # MB: to the NEXT PHASE
                    #
                    for _xx, _pd in enumerate(pickdict[_stat][_phase]):
                        _slice = _xx + 1
                        if _slice not in slices:
                            continue         # MB: to the next slicepick

                        # Check DICT keys
                        _check_final_dict(big_final,
                                          ref_phase_name,
                                          ref_phase_class,
                                          picker_tag,
                                          multipick_tag,
                                          _slice)

                        # Check that it has picked
                        if pickdict[_stat][_phase][_xx]["timeUTC_pick"]:
                            _delta = (
                                  pickdict[_stat][_phase][_xx]["timeUTC_pick"] -
                                  ref_phase_time)
                            try:
                                # append delta
                                (big_final[ref_phase_name]
                                          [ref_phase_class]
                                          [picker_tag]
                                          [multipick_tag]
                                          [_slice]['delta'].append(_delta)
                                 )
                                # increase pick_and_unpick_count
                                (big_final[ref_phase_name]
                                          [ref_phase_class]
                                          [picker_tag]
                                          [multipick_tag]
                                          [_slice]['pick_and_unpick_count']) += 1
                            except KeyError:
                                logger.error("WEIRD: All keys should be created " +
                                             "already, check why!")
                                raise QE.CheckError()
                        else:
                            try:
                                # increase totalcount
                                (big_final[ref_phase_name]
                                          [ref_phase_class]
                                          [picker_tag]
                                          [multipick_tag]
                                          [_slice]['pick_and_unpick_count']) += 1
                            except KeyError:
                                logger.error("WEIRD: All keys should be created " +
                                             "already, check why!")
                                raise QE.CheckError()

                    # BAIT RELATED CHECK

            elif (metastatdict[eqid][_stat]['isreference'] and
                  not metastatdict[eqid][_stat]['isdownloaded']):
                ref_stat_not_downloaded += 1
                continue
            elif (not metastatdict[eqid][_stat]['isreference'] and
                  not metastatdict[eqid][_stat]['isdownloaded']):
                noref_stat_not_downloaded += 1
                continue
            elif (not metastatdict[eqid][_stat]['isreference'] and
                  metastatdict[eqid][_stat]['isdownloaded']):
                noref_stat_downloaded += 1
                continue
            else:
                logger.error("Unusual switch situation: Station %s" % _stat)
                raise QE.CheckError

    # =================================================== Write DF, CSV
    # ===================================================
    # ===================================================
    # ----- Populate DataFrame
    for pname in big_final.keys():
        for clnum in big_final[pname].keys():
            for picker in big_final[pname][clnum].keys():
                for mpt in big_final[pname][clnum][picker].keys():
                    for win in big_final[pname][clnum][picker][mpt].keys():
                        # Working on delta_container
                        # -------------------------
                        tmp_work_dict = (
                            big_final[pname][clnum][picker][mpt][win])

                        # -------------------------
                        # Stuff to store anyway
                        tmp_work_dict['tot_reference_count'] = (
                                              reference_total[pname][clnum])

                        if tmp_work_dict['delta']:
                            tmp_work_dict['percent'] = round(
                                    (len(tmp_work_dict['delta']) * 100) /
                                    tmp_work_dict['pick_and_unpick_count'], 3)
                            if len(tmp_work_dict['delta']) == 1:
                                tmp_work_dict['mean'] = round(
                                            np.mean(tmp_work_dict['delta']), 3)
                                tmp_work_dict['std'] = None
                            elif len(tmp_work_dict['delta']) >= 2:
                                tmp_work_dict['mean'] = round(
                                            np.mean(tmp_work_dict['delta']), 3)
                                tmp_work_dict['std'] = round(
                                            np.std(tmp_work_dict['delta']), 3)

                        else:
                            tmp_work_dict['mean'] = None
                            tmp_work_dict['std'] = None
                            tmp_work_dict['percent'] = 0.0

                        # Appending
                        df_table = df_table.append({
                                    'EVENTID': 'ALL',
                                    'PHASENAME': pname,
                                    'PHASECLASS': clnum,
                                    'PICKER': picker,
                                    'MULTIPICKTAG': mpt,
                                    'SLICE': win,
                                    # MB: len of delta
                                    'COUNT': len(tmp_work_dict['delta']),
                                    # pick and unpicked data
                                    'TOTCOUNT': (
                                      tmp_work_dict['pick_and_unpick_count']),
                                    # total reference count (phase,class)
                                    'REFERENCECOUNT': (
                                      tmp_work_dict['tot_reference_count']),
                                    'MEAN': tmp_work_dict['mean'],
                                    'STD': tmp_work_dict['std'],
                                    'PERCENTAGE': tmp_work_dict['percent']},
                                    ignore_index=True)
    # ----- Printing OUT
    if savedf:
        df_table.to_csv(os.sep.join([storedir, outfilename]),
                        sep=',',
                        index=False,
                        na_rep="NA")


def createPickerSliceTable(eqid,
                           pickdict,
                           metastatdict,
                           storedir,
                           slices=("1", "2", "3"),
                           mpsliceTag=("ROUND1",),
                           pickersTag=("BK", "MyAIC"),
                           outfilename="PickTable.csv",
                           phases=["P", "Pg", "Pn", "P1"],
                           savedf=True):
    """
    This function should help to extract important info in a table about
    the Reference (INPUT) picks and the ones done by the ADAPT framework

    From v0.0.9 mpsliceasstag is the pick-tag used for slicing
    the multipicker object from which we obtained the pickdict
    """
    logger.info("Creating Reference data frame ...")

    # --------------- CHECKS
    if (not isinstance(mpsliceTag, (list, tuple)) or
       not isinstance(mpsliceTag, (list, tuple))):
        logger.error("mpsliceTag and pickersTag MUST BE list or tuple object!")
        raise QE.InvalidVariable()

    # --- Initialize the DataFrame
    df_table = pd.DataFrame(columns=["Phase",
                                     "Count_In",
                                     "NoiseLevel_Estimate_passed"])
    for _ss in slices:
        for _pp in pickersTag:
            df_table["_".join(("Slice", _ss, _pp))] = None

    # ---  Initialize the dict
    tmpdict = {}
    for ii in phases:
        tmpdict[ii] = {"Count_In": 0,
                       "NoiseLevel_Estimate_passed": 0}
        for _ss in slices:
            for _pp in pickersTag:
                tmpdict[ii]["_".join(("Slice", _ss, _pp))] = 0

    # the base is the input Station List (reference)
    for stat in pickdict:
        if (metastatdict[eqid][stat]["isdownloaded"] and
           metastatdict[eqid][stat]["isreference"]):
            #
            fpp_refe = (metastatdict[eqid][stat]
                        ["reference_picks"][0][0].split("_")[-1])
            # Initial Count //Checks
            try:
                tmpdict[fpp_refe]["Count_In"] += 1
            except KeyError:
                # Station has a first arrival different from the
                # phases listed in Input
                continue

            if not metastatdict[eqid][stat]["ispickable"]:
                # station didn't pass the multipicking noise estimator (Z)
                continue
            else:
                tmpdict[fpp_refe]["NoiseLevel_Estimate_passed"] += 1

            # Inner Calculation
            for picktag in pickdict[stat].keys():
                try:
                    _pp = picktag.split("_")[0]
                    _mpsat = picktag.split("_")[-1]
                except IndexError:
                    # MB: 17072019
                    # If here, it's probably because the phasekey is
                    # simply P1,P2 (non splittable by '_'). These are
                    # typicallly stored as FINAL pick, not useful here.
                    continue

                if _pp in pickersTag and _mpsat in mpsliceTag:
                    for xx, ss in enumerate(pickdict[stat][picktag]):
                        # ss This is a list of dict
                        # xx is associated with the slice number
                        if str(xx+1) in slices:
                            try:
                                if ss["timeUTC_pick"]:
                                    tmpdict[fpp_refe]["_".join(("Slice", str(xx+1), _pp))] += 1
                            except KeyError:
                                continue

    # # ----- Percentage (to implement)
    # for phase in tmpdict.keys():
    #     pass

    # ----- Adding
    for phase in tmpdict.keys():
        df_table = df_table.append({"Phase": phase, **tmpdict[phase]},
                                   ignore_index=True)

    # ----- Printing OUT
    if savedf:
        df_table.to_csv(os.sep.join([storedir, outfilename]),
                        sep=',',
                        index=False)


def generateRdataframe_BAITEVAL(eqid,
                                pickdict,
                                statdict,
                                metastatdict,
                                storedir,
                                outfilename="deltaR_BaItEVAL.csv",
                                savedf=True):
    """
    Same as the other but only for the BAIT_EVAL tag on pickDict
    """
    logger.info("Creating RefDelta data frame BAIT_EVAL for plotting ...")
    # --------------- Initiliaze the dataframe
    df_pick = pd.DataFrame(columns=["EVENTID",
                                    "STATION",
                                    "NETWORK",
                                    "EPIDIST",
                                    "PHASENAME",
                                    "PHASECLASS",
                                    "PICKER",
                                    "SLICE",
                                    "DELTA"
                                    ])
    # --------------- Looping
    for stat in pickdict:
        if (metastatdict[eqid][stat]["isdownloaded"] and
           metastatdict[eqid][stat]["isreference"]):
            try:
                # extract first reference phase
                fpp_refe = metastatdict[eqid][stat]["reference_picks"][0][0]  # .split("_")[-1])

            except KeyError:
                # This station was not picked --> no data
                logger.warning("Station %s NOT picked" % stat)
                continue

            # Take the reference phase TIME from the input handpick
            # ... exctract the Slice/predicted by the PREDICTED phase
            try:
                reftime = (pickdict[stat]
                                   [fpp_refe][0]
                                   ["timeUTC_pick"])
            except KeyError:
                logger.error("STAT: %s - Missing ReferenceTime!" % stat)
                raise QE.MissingVariable({"type": "MissingVariable",
                            "message": "ReferenceTime pick missing ... fishy"})

            # Extract delta
            try:
                # _baittime = pickdict[stat]["BAIT_EVAL_1"][0]["timeUTC_pick"]
                _baittime = metastatdict[eqid][stat]['bait_picks'][0][0]
                deltapick = (_baittime - reftime)
            except (TypeError, KeyError):
                deltapick = None
            #
            df_pick = df_pick.append({
                  "EVENTID": eqid,
                  "STATION": stat,
                  "NETWORK": statdict[stat]["network"],
                  "EPIDIST": "{:6.2f}".format(metastatdict
                                              [eqid]
                                              [stat]
                                              ["epidist"]),
                  "PHASENAME": fpp_refe.split("_")[-1],
                  "PHASECLASS": (pickdict
                                 [stat]
                                 [fpp_refe][0]
                                 ["pickclass"]),
                  "PICKER": 'BAIT_EVAL',
                  "SLICE": '1',
                  "DELTA": deltapick},
                  ignore_index=True)

    # ----- Storing OUT
    if savedf:
        df_pick.to_csv(os.sep.join([storedir, outfilename]),
                       sep=',',
                       index=False,
                       na_rep="NA")


def generateRdataframe(eqid,
                       pickdict,
                       statdict,
                       metastatdict,
                       storedir,
                       mpsliceTag=("ROUND1"),
                       pickersTag=("BK", "MyAIC"),
                       outfilename="deltaR.csv",
                       savedf=True):
    """
    This function will create pandas dataframe (and store to ASCII)
    to analyze statistics.

    Typical split after picktag.split("_")

    pickersTag = [0]
    mpsliceTag = [1]

    ['BK', 'ROUND1']
    ['MyAIC', 'ROUND1']
    ['BK', 'ROUND2']
    ['MyAIC', 'ROUND2']
    ['Reference', 'P1']
    ['Reference', 'S1']
    ['Predicted', 'Pg']
    ['Predicted', 'Pn']
    ['Predicted', 'PmP']
    ['Associated', 'ROUND1']
    ['Associated', 'ROUND2']

    REFERENCE:
    1) https://stackoverflow.com/questions/44513738/pandas-create-empty-dataframe-with-only-column-names
    2) https://stackoverflow.com/questions/17091769/python-pandas-fill-a-dataframe-row-by-row
    """
    logger.info("Creating ReferenceDelta data frame for plotting ...")

    # --------------- CHECKS
    if (not isinstance(mpsliceTag, (list, tuple)) or
       not isinstance(mpsliceTag, (list, tuple))):
        logger.error("mpsliceTag and pickersTag MUST BE list or tuple object!")
        raise QE.InvalidVariable()

    # --------------- Initiliaze the dataframe
    df_pick = pd.DataFrame(columns=["EVENTID",
                                    "STATION",
                                    "NETWORK",
                                    "EPIDIST",
                                    "PHASENAME",
                                    "PHASECLASS",
                                    "PICKER",
                                    "SLICE",
                                    "DELTA"
                                    ])
    # --------------- Looping
    for stat in pickdict:
        if (metastatdict[eqid][stat]["isdownloaded"] and
           metastatdict[eqid][stat]["isreference"]):
            try:
                # extract first reference phase
                fpp_refe = metastatdict[eqid][stat]["reference_picks"][0][0]  # .split("_")[-1])

            except KeyError:
                # This station was not picked --> no data
                logger.warning("Station %s NOT picked" % stat)
                continue

            # Take the reference phase TIME from the input handpick
            # ... exctract the Slice/predicted by the PREDICTED phase
            try:
                reftime = (pickdict[stat]
                                   [fpp_refe][0]
                                   ["timeUTC_pick"])
            except KeyError:
                logger.error("STAT: %s - Missing ReferenceTime!" % stat)
                raise QE.MissingVariable(
                          {"type": "MissingVariable",
                           "message": "ReferenceTime pick missing ... fishy"})

            # Extract delta for each Slice of each Picker
            for picktag in pickdict[stat].keys():

                try:
                    _pp = picktag.split("_")[0]
                    _mpsat = picktag.split("_")[-1]
                except IndexError:
                    # MB: 17072019
                    # If here, it's probably because the phasekey is
                    # simply P1,P2 (non splittable by '_'). These are
                    # typicallly stored as FINAL pick, not useful here.
                    continue

                if _pp in pickersTag and _mpsat in mpsliceTag:
                    for xx, ss in enumerate(pickdict[stat][picktag]):
                        # This is a list of dict
                        try:
                            deltapick = ss["timeUTC_pick"] - reftime
                        except TypeError:
                            # We are here if ss["timeUTC_pick"]==None
                            deltapick = None
                        #
                        df_pick = df_pick.append({
                              "EVENTID": eqid,
                              "STATION": stat,
                              "NETWORK": statdict[stat]["network"],
                              "EPIDIST": "{:6.2f}".format(metastatdict
                                                          [eqid]
                                                          [stat]
                                                          ["epidist"]),
                              "PHASENAME": fpp_refe.split("_")[-1],
                              "PHASECLASS": (pickdict
                                             [stat]
                                             [fpp_refe][0]
                                             ["pickclass"]),
                              "PICKER": _pp,
                              "SLICE": str(xx+1),
                              "DELTA": deltapick},
                              ignore_index=True)

    # ----- Storing OUT
    if savedf:
        df_pick.to_csv(os.sep.join([storedir, outfilename]),
                       sep=',',
                       index=False,
                       na_rep="NA")


def createMisinterpTable(eqid,
                         pickdict,
                         statdict,
                         metastatdict,
                         storedir,
                         savedf=True):
    """
    This function will create pandas dataframe (and store to ASCII)
    to analyze statistics.

    REFERENCE:
    1) https://stackoverflow.com/questions/44513738/pandas-create-empty-dataframe-with-only-column-names
    2) https://stackoverflow.com/questions/17091769/python-pandas-fill-a-dataframe-row-by-row
    """
    logger.info("Creating Misinterpretation data frame ...")
    # --------------- Initiliaze the dataframe
    df_misint_table = pd.DataFrame(columns=["Phase Input",
                                            "Phase Predicted",
                                            "Count"])
    df_misint = pd.DataFrame(columns=["EVENTID",
                                      "STATION",
                                      "PREDICTED",
                                      "INPUT"])
    # --------------- Looping
    df_misint_table_dict = {}
    for stat in pickdict:
        if (metastatdict[eqid][stat]["isdownloaded"] and
           metastatdict[eqid][stat]["isreference"]):
            try:
                # derived from prediction phase
                fpp_pred = (metastatdict
                            [eqid][stat]
                            ["predicted_picks"][0][0].split("_")[-1])
                fpp_refe = (metastatdict
                            [eqid][stat]
                            ["reference_picks"][0][0].split("_")[-1])
                logger.debug("PRED: %r REFE: %r" % (fpp_pred, fpp_refe))
            except KeyError:
                # This station was not picked --> no data
                logger.warning("Station %s NOT picked" % stat)
                continue

            # Check if predicted and reference coincide
            if fpp_pred != fpp_refe:
                df_misint = df_misint.append({"EVENTID": eqid,
                                              "STATION": stat,
                                              "PREDICTED": fpp_pred,
                                              "INPUT": fpp_refe},
                                             ignore_index=True)
                # --- Adding it 21/11/2018
                if fpp_refe not in df_misint_table_dict.keys():
                    df_misint_table_dict[fpp_refe] = {}
                if fpp_pred not in df_misint_table_dict[fpp_refe].keys():
                    df_misint_table_dict[fpp_refe][fpp_pred] = 0
                df_misint_table_dict[fpp_refe][fpp_pred] += 1
                # ---
                logger.error("STAT: %s - Prediction: %s  /  Input: %s" %
                             (stat, fpp_pred, fpp_refe))

    # ----- Rearrange Dict
    for phaseIN in df_misint_table_dict.keys():
        for phasePRED in df_misint_table_dict[phaseIN]:
            df_misint_table = df_misint_table.append({
                            "Phase Input": phaseIN,
                            "Phase Predicted": phasePRED,
                            "Count": df_misint_table_dict[phaseIN][phasePRED]},
                            ignore_index=True)

    # ----- Storing OUT
    if savedf:
        df_misint_table.to_csv(os.sep.join([
                                        storedir,
                                        "Misinterpretation_CountTable.csv"]),
                               sep=',',
                               index=False)
        df_misint.to_csv(os.sep.join([storedir, "Misinterpretation.csv"]),
                         sep=',',
                         index=False,
                         na_rep="NA")


def countReferenceClass(eqid,
                        pickdict,
                        metastatdict,
                        storedir,
                        classes=("0", "1", "2", "3", "4"),
                        # mpsliceTag=("ROUND1"),
                        # pickersTag=("BK", "MyAIC"),
                        outfilename="ReferenceClassTable.csv",
                        phases=["P", "Pg", "Pn", "P1"],
                        savedf=True):
    """
    This function helps to extract the phase-class count
    - added in 0.1.6 (Mercalli) -
    """
    logger.info("Creating Reference Class Count data frame ...")

    # --- Initialize the DataFrame
    df_table = pd.DataFrame(columns=["Phase",
                                     "Count_In",
                                     "NoiseLevel_Estimate_passed"])
    for _ss in classes:
        df_table[_ss] = None

    # ---  Initialize the dict
    tmpdict = {}
    for ii in phases:
        tmpdict[ii] = {"Count_In": 0,
                       "NoiseLevel_Estimate_passed": 0}
        for _ss in classes:
            tmpdict[ii][_ss] = 0

    # the base is the input Station List (reference)
    for stat in pickdict:
        if (metastatdict[eqid][stat]["isdownloaded"] and
           metastatdict[eqid][stat]["isreference"]):
            #
            fpp_refe = (metastatdict[eqid][stat]
                        ["reference_picks"][0][0].split("_")[-1])
            # Initial Count //Checks
            try:
                tmpdict[fpp_refe]["Count_In"] += 1
            except KeyError:
                # Station has a first arrival different from the
                # phases listed in Input
                continue

            if not metastatdict[eqid][stat]["ispickable"]:
                # station didn't pass the multipicking noise estimator (Z)
                continue
            else:
                tmpdict[fpp_refe]["NoiseLevel_Estimate_passed"] += 1

            # Class
            pc = pickdict[stat]["Reference_"+fpp_refe][0]["pickclass"]
            tmpdict[fpp_refe][str(pc)] += 1

    # # ----- Percentage (to implement)
    # for phase in tmpdict.keys():
    #     pass

    # ----- Adding
    for phase in tmpdict.keys():
        df_table = df_table.append({"Phase": phase, **tmpdict[phase]},
                                   ignore_index=True)

    # ----- Printing OUT
    if savedf:
        df_table.to_csv(os.sep.join([storedir, outfilename]),
                        sep=',',
                        index=False)


def createHeatMatrix():
    """
    This function should serve as helper to create heat matrix for
    each station based on slice num and phase.
    It should help define the goodness of each algorithm
    """
    pass


def createWeightMatrix(eqid,
                       metastatdict,
                       workd,          # QuakePick_ALL
                       finald,         # QuakePick_FINAL
                       # The index for the next list of phase is ALWAYS 0
                       auto_phase=("P1"),
                       ref_phase=("Reference_P", "Reference_Pn",
                                  "Reference_P1", "Reference_Pg")):
    """
    MPX like creation of matrix.
    Added on v0.2.1 (Vidale)

    *** NB this function return a dict for ONLY ONE event.

    d[rc] = {tot_cnt: %d
             [ac] = {
                     cnt: %d
                     std: %f
                     deltas: [..., ...]
                    }
            }

    OUT:
      [file] refclass autoclass count stdDelta
      [dict] d[ref][auto]={std: %f, cnt: %d, delta=[]}
    """
    logger.info("Creating CLASS-MATRIX dict/file ...")

    # --- Initialize the dictionary
    matDict = {}
    totDict = {} # Total dict containing all the ref_phase (REF and DOWNLOADED)

    # the base is the input Station List (reference)
    for stat in workd:
        if (metastatdict[eqid][stat]["isdownloaded"] and
           metastatdict[eqid][stat]["isreference"] and
           metastatdict[eqid][stat]["ispickable"]):  # bugfix from EGU2019 #MB
            #
            for _ph in workd[stat].keys():
                if _ph not in ref_phase:
                    continue
                else:
                    # Extract REFERENCE Pick (always pick the first)
                    rtp = workd[stat][_ph][0]["timeUTC_pick"]
                    rcl = workd[stat][_ph][0]["pickclass"]
                    if rcl not in totDict:
                        totDict[rcl] = 0
                    totDict[rcl] += 1

                    # Check if AUTO picker Picked the REFERENCE
                    if stat not in finald:
                        continue

                    # Extract AUTO Pick
                    for _aph in finald[stat].keys():
                        if _aph not in auto_phase:
                            continue
                        else:
                            try:
                                atp = finald[stat][_aph][0]["timeUTC_pick"]
                                acl = finald[stat][_aph][0]["pickclass"]
                            except KeyError:
                                # Missing PHASE in the final dict
                                atp = None
                                acl = None

                            # Calculate Delta (AUTO - REF)
                            if atp:
                                # Make sure that atp exist
                                try:
                                    _del = atp - rtp
                                except TypeError:
                                    logger.error(
                                        "Missing ReferencePick @ %s [WEIRD!]" %
                                        stat)
                                    raise QE.InvalidType()

                                # Check Dict KEYS
                                if rcl not in matDict:
                                    matDict[rcl] = {}
                                if acl not in matDict[rcl]:
                                    matDict[rcl][acl] = {"cnt": 0,
                                                         "std": None,
                                                         "deltas": [],
                                                         "statnames": []}

                                # Populate
                                matDict[rcl][acl]["cnt"] += 1
                                matDict[rcl][acl]["deltas"].append(_del)
                                matDict[rcl][acl]["statnames"].append(stat)

    # STD and dict validation
    for _rc in matDict:
        for _ac in matDict[_rc]:
            # Checks
            if (matDict[_rc][_ac]["cnt"] !=
               len(matDict[_rc][_ac]["deltas"])):
                logger.error("Delta Length and Count differ @ %s [WEIRD!]"
                             % stat)
                raise QE.CheckError()
            # STD
            matDict[_rc][_ac]["std"] = np.std(matDict[_rc][_ac]["deltas"])

    # Adding the TOTCOUNT
    for _rc in matDict:
        matDict[_rc]["tot_cnt"] = totDict[_rc]
    #
    return matDict


def mergeWeightMatrix(indict):
    """
    This script will take care to merge ALL the single-event tuning
    matrix and create a single one dict.

    Eventually, the out-dictionary is stored

    d[eqid]:
        d[rc] = {tot_cnt: %d
                 [ac] = {
                         cnt: %d
                         std: %f
                         deltas: [..., ...]
                        }
                }
    """
    outDict = {}
    #
    for _id in indict:
        for _rc in indict[_id]:
            # Check
            if _rc not in outDict:
                outDict[_rc] = {}
                outDict[_rc]["tot_cnt"] = 0
            #
            for _ac in indict[_id][_rc]:
                if _ac == "tot_cnt":
                    outDict[_rc][_ac] += indict[_id][_rc][_ac]
                else:
                    # Check
                    if _ac not in outDict[_rc]:
                        outDict[_rc][_ac] = {"cnt": 0,
                                             "std": None,
                                             "deltas": [],
                                             "statnames": []}
                    #
                    outDict[_rc][_ac]["cnt"] += indict[_id][_rc][_ac]["cnt"]
                    outDict[_rc][_ac]["deltas"] += indict[_id][_rc][_ac]["deltas"]
                    outDict[_rc][_ac]["statnames"] += indict[_id][_rc][_ac]["statnames"]

    # STD and dict validation
    for _rc in outDict:
        for _ac in outDict[_rc]:
            if _ac == "tot_cnt":  # avoiding count key
                continue
            # Checks
            if (outDict[_rc][_ac]["cnt"] !=
               len(outDict[_rc][_ac]["deltas"])):
                logger.error("DeltasLen and Count differ! [WEIRD!] %d - %d" %
                             (outDict[_rc][_ac]["cnt"],
                              len(outDict[_rc][_ac]["deltas"])))
                raise QE.CheckError()
            # STD
            outDict[_rc][_ac]["std"] = np.std(outDict[_rc][_ac]["deltas"])
    #
    return outDict


def writeWeigthMatrix2File(indict, storedir="./", outfile="matrix_bar.gmt",
                           out_type="GMT"):
    """
    Scripts to dump into a file a matrix for gmt plot or other type.

    *** NB the dict must be of the form

    d[rc] = {tot_cnt: %d
             [ac] = {
                     cnt: %d
                     std: %f
                     deltas: [..., ...]
                    }
            }
    """
    if out_type.lower() == "gmt":
        # dumps the dictionary in a file for GMTscript
        with open(os.sep.join([storedir, outfile]), "w") as OUT:
            for _rc in indict:
                for _ac in indict[_rc]:
                    if _ac == "tot_cnt":  # avoiding count key
                        continue
                    OUT.write(("%s %s %d %6.3f"+os.linesep) % (
                               str(_rc), str(_ac),
                               indict[_rc][_ac]["cnt"],
                               indict[_rc][_ac]["std"]))

    elif out_type.lower() == "debug":
        # dumps the dictionary in a file for GMTscript
        with open(os.sep.join([storedir, outfile]), "w") as OUT:
            for _rc in indict:
                for _ac in indict[_rc]:
                    if _ac == "tot_cnt":  # avoiding count key
                        continue
                    OUT.write(("%s %s %d %6.3f %s"+os.linesep) % (
                               str(_rc), str(_ac),
                               indict[_rc][_ac]["cnt"],
                               indict[_rc][_ac]["std"],
                               indict[_rc][_ac]["statnames"]))

    else:
        logger.error("Wrong out_type %s ['gmt', 'debug']" % out_type)
        return False
    #
    return True


def weighterTable_reference(eqid,
                            pickdict,
                            quakefinalpick,
                            statdict,
                            metastatdict,
                            storedir,
                            savedf=True,
                            outfilename="WeighterStats.csv"):
    """
    This function will help to understand how the Weighter class works.
    As `pickdict` in input we should use the FINAL one. That contains the
    statistic object inside

    EVENTID, STATION, NETWORK, EPIDIST, PHASENAME, PHASECLASS, MEAN, MEDIAN, STDALL

    """
    logger.info("Creating WEIGHTER data frame for plotting ...")
    # --------------- Initiliaze the dataframe
    df_pick = pd.DataFrame(columns=["EVENTID",
                                    "STATION",
                                    "NETWORK",
                                    "EPIDIST",
                                    "PHASENAME",
                                    # "PHASEPOLARITY",
                                    "PHASECLASS",
                                    # "REFERENCEPHASENAME",
                                    # "REFERENCECLASS",
                                    "ATTRIBUTE",
                                    "ATTRIBUTEVALUE"
                                    ])
    # --------------- Looping
    for stat in pickdict:
        if (metastatdict[eqid][stat]["isdownloaded"] and
           metastatdict[eqid][stat]["isreference"]):
            try:
                # extract first reference phase
                fpp_refe = metastatdict[eqid][stat]["reference_picks"][0][0]

            except KeyError:
                # This station was not picked --> no data
                logger.warning("Station %s NOT picked" % stat)
                continue

            # Take the reference phase TIME from the input handpick
            # ... exctract the Slice/predicted by the PREDICTED phase
            try:
                reftime = (pickdict[stat]
                                   [fpp_refe][0]
                                   ["timeUTC_pick"])
            except KeyError:
                logger.error("STAT: %s - Missing ReferenceTime!" % stat)
                raise QE.MissingVariable(
                          {"type": "MissingVariable",
                           "message": "ReferenceTime pick missing ... fishy"}
                      )

            # Extract statistics if STATION has been stored in final dict
            if stat in quakefinalpick.keys():
                for picktag in quakefinalpick[stat].keys():
                    _pp = picktag.split("_")[0]
                    # _mpsat = picktag.split("_")[1]
                    if _pp == "P1":
                        for xx, ss in enumerate(quakefinalpick[stat][picktag]):
                            # This is a list of dict -- > should be only one

                            maronn = ss['weight'].get_triage_results()

                            for _attr in ("MEAN", "MEDIAN", "STDALL"):
                                if _attr == "MEAN":
                                    try:
                                        attributevalue = (
                                          maronn['mean'] -
                                          reftime
                                        )
                                    except TypeError:
                                        attributevalue = None

                                elif _attr == "MEDIAN":
                                    try:
                                        attributevalue = (
                                          maronn['median'] -
                                          reftime
                                        )
                                    except TypeError:
                                        attributevalue = None

                                elif _attr == "STDALL":
                                    try:
                                        attributevalue = maronn['std']
                                    except TypeError:
                                        attributevalue = None

                                else:
                                    # NB: it's impossible to end up here
                                    raise QE.CheckError()

                                # --- Populate
                                df_pick = df_pick.append({
                                      "EVENTID": eqid,
                                      "STATION": stat,
                                      "NETWORK": statdict[stat]["network"],
                                      "EPIDIST": "{:6.2f}".format(metastatdict
                                                                  [eqid]
                                                                  [stat]
                                                                  ["epidist"]),
                                      "PHASENAME": fpp_refe.split("_")[-1],
                                      "PHASECLASS": (pickdict
                                                     [stat]
                                                     [fpp_refe][0]
                                                     ["pickclass"]),
                                      "ATTRIBUTE": _attr,
                                      "ATTRIBUTEVALUE": attributevalue},
                                      ignore_index=True)
            else:
                continue

    # ----- Storing OUT
    if savedf:
        df_pick.to_csv(os.sep.join([storedir, outfilename]),
                       sep=',',
                       index=False,
                       na_rep="NA")


def adapt_comprehensive_table_reference(eqid,
                                        reference_pick_dict,   # FINAL PICKS
                                        reference_phase_list,
                                        statdict,
                                        metastatdict,
                                        storedir,
                                        savedf=True,
                                        outfilename="Reference_Features.csv"):
    """
    This function will help to understand how the Weighter class works.

    Dumping out all the possible PICKS made by ADAPT (plus the analisys element),
    and eventually comparyng those with the reference (if it's the case)

    For column attributes check inside the script

    !!! 28-01-2020 The key MyAIC should be ALL CAPS ==> MYAIC
        This is due to the internal change with the upper() method.

    """
    logger.info("Creating COMPREHENSIVE ADAPT data frame for plotting ...")

    # --------------- Initiliaze the dataframe

    # NB: Always specify the column names!!!
    columns = ["EVENTID",
               "STATION",
               "NETWORK",
               "REF_PHASE",
               "REF_POLARITY",
               "REF_CLASS",
               "REF_TIME",
               "EPIDIST",
               #
               "max_signal_to_noise_ratio",
               "mean_signal_to_noise_ratio",
               "max_signal_to_longnoise_ratio",
               "mean_signal_to_longnoise_ratio",
               "max_signal_to_startnoise_ratio",
               "mean_signal_to_startnoise_ratio",
               "max_longnoise_to_startnoise_ratio",
               "mean_longnoise_to_startnoise_ratio",
               "signal_below_threshold",
               "noise_over_threshold",
               "absolute_energy_noise",
               "absolute_energy_signal",
               "std_noise",
               "std_signal",
               "dominant_frequencies_shift",
               "dominant_frequencies_ratio",
               "dominant_frequency_noise",
               "dominant_frequency_noise"
               ]

    df_pick = pd.DataFrame(columns=columns)

    # From weighter class
    # self.triage_dict = {'tot_obs': 0,
    #                     'valid_obs': [],
    #                     'spare_obs': [],
    #                     'stable_pickers': [],
    #                     'outliers': []}
    # self.triage_results = {'mean': None,
    #                        'median': None,
    #                        'mean-median': None,
    #                        'std': None,
    #                        'var': None}

    # FOR THE MOMENT I DON'T CARE ABOUT 2 REFERENCE...I'll find the way
    # I will store anyway the P1P2 from ADAPT

    # --------------- Looping
    # I want to extract ALL THE POSSIBLE ONE, EVEN THE UNPICKED by both
    for stat in reference_pick_dict:

        fillingdict = {ii: None for ii in columns}

        # --- collecting STAT info
        try:
            fillingdict["EVENTID"] = eqid
            fillingdict["STATION"] = stat
            fillingdict["NETWORK"] = statdict[stat]["network"]
            fillingdict["EPIDIST"] = "{:6.2f}".format(metastatdict
                                                      [eqid]
                                                      [stat]
                                                      ["epidist"])
        except KeyError as err:
            # ReallyBAD --> continue or throw error
            logger.error("STAT %s  ... missing from inventory" % (err.args[0]))
            raise QE.MissingVariable()

        # =========> Here WE RUN THROUGH all the phases <============
        for _xx, _mph in enumerate(reference_pick_dict[stat]):
            if _mph.split("_")[-1] not in reference_phase_list:
                continue
            #
            _tmpd = reference_pick_dict[stat][_mph][0]  # In the dict there's should be only 1 pick per phase
            #
            fillingdict["REF_PHASE"] = _mph.split("_")[-1]
            fillingdict["REF_POLARITY"] = _tmpd['pickpolar']
            fillingdict["REF_CLASS"] = _tmpd['pickclass']
            fillingdict["REF_TIME"] = _tmpd["timeUTC_pick"]

            for _kk, _vv in _tmpd['features'].items():
                fillingdict[_kk] = _vv

            # Storing
            df_pick = df_pick.append(fillingdict, ignore_index=True)

    # ----- Saving
    if savedf:
        df_pick.to_csv(os.sep.join([storedir, outfilename]),
                       sep=',',
                       index=False,
                       float_format="%10.5f",
                       na_rep="NA")


def adapt_comprehensive_table_production(
                              eqid,
                              pickdict,         # ALL of them
                              quakefinalpick,   # FINAL PICKS
                              statdict,
                              metastatdict,
                              storedir,
                              savedf=True,
                              outfilename="WeighterStats.csv"):
    """
    This function will help to understand how the Weighter class works.

    Dumping out all the possible PICKS made by ADAPT (plus the analisys element),
    and eventually comparyng those with the reference (if it's the case)

    For column attributes check inside the script

    !!! 28-01-2020 The key MyAIC should be ALL CAPS ==> MYAIC
        This is due to the internal change with the upper() method.

    """
    logger.info("Creating COMPREHENSIVE ADAPT data frame for plotting ...")

    # --------------- Initiliaze the dataframe

    # NB: Always specify the column names!!!
    columns = ["EVENTID",
               "STATION",
               # "CHANNEL", # to be implemented
               "NETWORK",
               "EPIDIST",
               "PHASE",
               "POLARITY",
               "CLASS",
               "TIME",
               "REF_PHASE",
               "REF_CLASS",
               "REF_POLARITY",
               "REF_TIME",
               "DELTA",  # Always ADAPT- Reference
               "MEAN",
               "MEDIAN",
               "MEANminMEDIAN",
               "STD",
               "VAR",
               # v0.6.28
               "MAD",
               "AAD",
               "TOT_OBS",
               "STABLE_PICKERS",
               "SPARE_OBS",
               "VALID_OBS",
               "OUTLIERS",
               # MB: 24-10-2019 for ML problem
               "DELTA_ASSOCIATED",
               "DELTA_PREDICTED",
               "ASSOCIATEDminPREDICTION",

               # MB: New v0.7.4: chenged pickers tag HOS + AIC
               "AIC_BK_1",
               "AIC_BK_2",
               "AIC_FP_1",
               "AIC_FP_2",
               "AIC_HOS_1",
               "AIC_HOS_2",
               "BK_FP_1",
               "BK_FP_2",
               "HOS_BK_1",
               "HOS_BK_2",
               "HOS_FP_1",
               "HOS_FP_2",

               # "FP_BK_1",
               # "FP_MYAIC_1",
               # "FP_KURT_1",
               # "BK_MYAIC_1",
               # "BK_KURT_1",
               # "MYAIC_KURT_1",
               # "KURT_MYAIC_1",
               # "FP_BK_2",
               # "FP_MYAIC_2",
               # "FP_KURT_2",
               # "BK_MYAIC_2",
               # "BK_KURT_2",
               # "MYAIC_KURT_2",
               # "KURT_MYAIC_2",

               # MB: 28-01-2020 for CLASSIFICATIN and ML problem.
               "AIC_1_MAX_SIGNAL_TO_NOISE_RATIO",
               "AIC_1_MEAN_SIGNAL_TO_NOISE_RATIO",
               "AIC_1_NOISE_OVER_THRESHOLD",
               "AIC_1_DOMINANT_FREQUENCIES_SHIFT",
               "AIC_1_DOMINANT_FREQUENCIES_RATIO",
               #
               "AIC_2_MAX_SIGNAL_TO_NOISE_RATIO",
               "AIC_2_MEAN_SIGNAL_TO_NOISE_RATIO",
               "AIC_2_NOISE_OVER_THRESHOLD",
               "AIC_2_DOMINANT_FREQUENCIES_SHIFT",
               "AIC_2_DOMINANT_FREQUENCIES_RATIO",
               #
               "HOS_1_MAX_SIGNAL_TO_NOISE_RATIO",
               "HOS_1_MEAN_SIGNAL_TO_NOISE_RATIO",
               "HOS_1_NOISE_OVER_THRESHOLD",
               "HOS_1_DOMINANT_FREQUENCIES_SHIFT",
               "HOS_1_DOMINANT_FREQUENCIES_RATIO",
               #
               "HOS_2_MAX_SIGNAL_TO_NOISE_RATIO",
               "HOS_2_MEAN_SIGNAL_TO_NOISE_RATIO",
               "HOS_2_NOISE_OVER_THRESHOLD",
               "HOS_2_DOMINANT_FREQUENCIES_SHIFT",
               "HOS_2_DOMINANT_FREQUENCIES_RATIO",
               #
               "FP_1_MAX_SIGNAL_TO_NOISE_RATIO",
               "FP_1_MEAN_SIGNAL_TO_NOISE_RATIO",
               "FP_1_NOISE_OVER_THRESHOLD",
               "FP_1_DOMINANT_FREQUENCIES_SHIFT",
               "FP_1_DOMINANT_FREQUENCIES_RATIO",
               #
               "FP_2_MAX_SIGNAL_TO_NOISE_RATIO",
               "FP_2_MEAN_SIGNAL_TO_NOISE_RATIO",
               "FP_2_NOISE_OVER_THRESHOLD",
               "FP_2_DOMINANT_FREQUENCIES_SHIFT",
               "FP_2_DOMINANT_FREQUENCIES_RATIO",
               #
               "BK_1_MAX_SIGNAL_TO_NOISE_RATIO",
               "BK_1_MEAN_SIGNAL_TO_NOISE_RATIO",
               "BK_1_NOISE_OVER_THRESHOLD",
               "BK_1_DOMINANT_FREQUENCIES_SHIFT",
               "BK_1_DOMINANT_FREQUENCIES_RATIO",
               #
               "BK_2_MAX_SIGNAL_TO_NOISE_RATIO",
               "BK_2_MEAN_SIGNAL_TO_NOISE_RATIO",
               "BK_2_NOISE_OVER_THRESHOLD",
               "BK_2_DOMINANT_FREQUENCIES_SHIFT",
               "BK_2_DOMINANT_FREQUENCIES_RATIO",
               #
               "MP_MAX_SIGNAL_TO_NOISE_RATIO",
               "MP_MEAN_SIGNAL_TO_NOISE_RATIO",
               "MP_MAX_SIGNAL_TO_LONGNOISE_RATIO",
               "MP_MEAN_SIGNAL_TO_LONGNOISE_RATIO",
               "MP_MAX_SIGNAL_TO_STARTNOISE_RATIO",
               "MP_MEAN_SIGNAL_TO_STARTNOISE_RATIO",
               "MP_MAX_LONGNOISE_TO_STARTNOISE_RATIO",
               "MP_MEAN_LONGNOISE_TO_STARTNOISE_RATIO",
               "MP_SIGNAL_BELOW_THRESHOLD",
               "MP_NOISE_OVER_THRESHOLD",
               "MP_ABSOLUTE_ENERGY_NOISE",
               "MP_ABSOLUTE_ENERGY_SIGNAL",
               "MP_STD_NOISE",
               "MP_STD_SIGNAL",
               "MP_DOMINANT_FREQUENCIES_SHIFT",
               "MP_DOMINANT_FREQUENCIES_RATIO",
               "MP_DOMINANT_FREQUENCY_NOISE",
               "MP_DOMINANT_FREQUENCY_SIGNAL",
               # v0.6.28
               "ISDOWNLOADED",
               "ISPICKABLE",
               "ISINPUT",
               "ISSELECTED",
               # Additional v0.7.10
               "MODE",
               "MODEALT",
               "BOOTMODE",
               "BOOTMEAN",
               "BOOTMADMODE",
               "DELTAbootmode",
               # Additional v0.7.20
               "MMD",
               #
               "PICKERS_INVOLVED"
               ]

    # 03-02-2020 added noise over threshold

    # 13-02-2020 added frequency features

    # 18-02-2020 added final MP features --> I was receiving this error:
    #       `ValueError: cannot reindex from a duplicate axis`
    # https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis-mean:
    # - For people who are still struggling with this error, it can also
    # happen if you accidentally create a duplicate column with the same
    # name. Remove duplicate columns like so:
    # >>> df = df.loc[:,~df.columns.duplicated()]

    df_pick = pd.DataFrame(columns=columns)

    # From weighter class
    # self.triage_dict = {'tot_obs': 0,
    #                     'valid_obs': [],
    #                     'spare_obs': [],
    #                     'stable_pickers': [],
    #                     'outliers': []}
    # self.triage_results = {'mean': None,
    #                        'median': None,
    #                        'mean-median': None,
    #                        'std': None,
    #                        'var': None}

    # FOR THE MOMENT I DON'T CARE ABOUT 2 REFERENCE...I'll find the way
    # I will store anyway the P1P2 from ADAPT

    # --------------- Looping
    # I want to extract ALL THE POSSIBLE ONE, EVEN THE UNPICKED by both
    for stat in pickdict:

        fillingdict = {ii: None for ii in columns}
        # fillingdict = {}
        QUAKEMISSING = False
        AUTOMATICMISSING = False

        # v0.6.28 ==> adding information for faster access
        if metastatdict[eqid][stat]["isautomatic"]:
            fillingdict["ISINPUT"] = True
        else:
            fillingdict["ISINPUT"] = False

        if metastatdict[eqid][stat]["isdownloaded"]:
            fillingdict["ISDOWNLOADED"] = True
        else:
            fillingdict["ISDOWNLOADED"] = False

        if metastatdict[eqid][stat]["ispickable"]:
            fillingdict["ISPICKABLE"] = True
        else:
            fillingdict["ISPICKABLE"] = False

        if metastatdict[eqid][stat]["isselected"]:
            fillingdict["ISSELECTED"] = True
        else:
            fillingdict["ISSELECTED"] = False

        # --- collecting STAT info
        try:
            fillingdict["EVENTID"] = eqid
            fillingdict["STATION"] = stat
            # fillingdict["CHANNEL"] = statdict[stat]["channel"] # to be implemented
            fillingdict["NETWORK"] = statdict[stat]["network"]
            fillingdict["EPIDIST"] = "{:6.2f}".format(metastatdict
                                                      [eqid]
                                                      [stat]
                                                      ["epidist"])
        except KeyError as err:
            # ReallyBAD --> continue or throw error
            logger.error("STAT %s  ... missing from inventory" % (err.args[0]))
            raise QE.MissingVariable()

        # For a same station we have possibly MULTIPLE PHASENAME (P1,P2)
        # Check IT beforehand <---
        try:
            n_final_phase = quakefinalpick[stat].keys()
        except KeyError as err:
            logger.warning("STAT %s Missing ADAPTPICK" % (err.args[0]))
            QUAKEMISSING = True

        try:
            if metastatdict[eqid][stat]["automatic_picks"]:
                automaticlist = (metastatdict[eqid][stat]["automatic_picks"])
            else:
                # This station was not picked --> no data
                automaticlist = []
                logger.warning("STAT %s Missing AUTOMATIC" % stat)
                AUTOMATICMISSING = True

        except KeyError as err:
            # This station was not picked --> no data
            logger.warning("STAT %s Missing REFERENCE" % (err.args[0]))
            AUTOMATICMISSING = True

        # =========> Here WE RUN THROUGH all the phases <============
        if not QUAKEMISSING:
            for _xx, _mph in enumerate(n_final_phase):
                _tmpd = quakefinalpick[stat][_mph][0]  # In the QUAKEFINAL dict there's should be only 1 pick per phase
                # Main
                fillingdict["PHASE"] = _mph
                fillingdict["POLARITY"] = _tmpd['pickpolar']
                fillingdict["CLASS"] = _tmpd['pickclass']
                fillingdict["TIME"] = _tmpd["timeUTC_pick"]

                # --- The TRIAGE RESULTS could not be there ...
                _triage_results = _tmpd["weight"].get_triage_results()
                if _triage_results:
                    for _kk, vv in _triage_results.items():
                        if _kk == "mean-median":
                            fillingdict["MEANminMEDIAN"] = vv
                        else:
                            fillingdict[_kk.upper()] = vv
                else:
                    fillingdict["MEAN"] = None
                    fillingdict["MEDIAN"] = None
                    fillingdict["MEANminMEDIAN"] = None
                    fillingdict["STD"] = None
                    fillingdict["VAR"] = None
                    fillingdict["MMD"] = None
                    fillingdict["AAD"] = None
                    fillingdict["MAD"] = None
                    fillingdict["MODE"] = None
                    fillingdict["MODEALT"] = None
                    fillingdict["BOOTMODE"] = None
                    fillingdict["BOOTMEAN"] = None
                    fillingdict["BOOTMADMODE"] = None

                # --- The TRIAGE DICT should always be there ...
                for _kk, vv in _tmpd["weight"].get_triage_dict().items():
                    if _kk in ('valid_obs',
                               'spare_obs',
                               'stable_pickers',
                               'outliers',
                               'pickers_involved'):
                        fillingdict[_kk.upper()] = len(vv)
                    elif _kk in ('failed_pickers'):
                        pass
                    else:
                        fillingdict[_kk.upper()] = vv
                        # {'tot_obs': 0,
                        #   'valid_obs': [],
                        #   'spare_obs': [],
                        #   'stable_pickers': [],
                        #   'outliers': [],
                        #   'failed_pickers': [],
                        #   'pickers_involved': []}
                        # pairs = [('FP', 'BK'), ('FP', 'MyAIC'),
                        #          ('FP', 'KURT'), ('BK', 'MyAIC'),
                        #          ('BK', 'KURT'), ('MyAIC', 'KURT'),
                        #          ('KURT', 'MyAIC')]

                # Added on 0.5.4 to extract features info from MP stage
                # NB!! At the moment is just MP1
                # v0.6.8 -> extract also other multipicking stage based
                #           on the last charachter on mp
                for _mphn, _mphn_val in pickdict[stat].items():
                    # # old
                    # if _mphn.split("_")[-1].lower() not in ("mp1", "mp2"):
                    #     continue  # to the next phase
                    # new
                    if _mphn.split("_")[-1].lower() != "mp" + _mph[-1]:
                        continue  # to the next phase
                    #
                    picknm = _mphn.split("_")[0]
                    _wd = pickdict[stat][_mphn]
                    for _ppx in range(len(_wd)):
                        try:
                            for _ff, _ffv in (
                              _wd[_ppx]['features']["SINGLE_TIME"].items()):
                                fillingdict[picknm.upper()+"_"+str(_ppx+1)+"_"+_ff.upper()] = float(_ffv)
                        except TypeError:
                            # Features SINGLETIMES missing
                            # - Fixed on 0.6.2 (instead of continue, just pass)
                            pass

                # Added on 0.5.9 to extract features info from FINAL stage
                try:
                    for _ff, _ffv in (
                      _tmpd['features']["SINGLE_TIME"].items()):
                        fillingdict["MP_"+_ff.upper()] = _ffv
                except TypeError:
                    # Features SINGLETIMES missing
                    # - Fixed on 0.6.2 (instead of continue, just pass)
                    pass

                # # Added v0.5.1 ==> not used for the moment
                # fillingdict['QUAKE_OUTLIER'] = _tmpd["outlier"]   # bool
                # fillingdict['QUAKE_EVALUATE'] = _tmpd["evaluate"]  # bool

                # --- Calculate DELTA BETWEEN Associated BAIT and PREDICTED
                # MB: if there's a QUAKE pick, there's for sure a BAIT
                if _tmpd["timeUTC_pick"]:
                    try:
                        fillingdict["DELTA_ASSOCIATED"] = float(
                                    _tmpd["timeUTC_pick"] -
                                    metastatdict[eqid][stat]['bait_picks']
                                                            [_xx][0])
                    except IndexError:
                        logger.warning(("[%s - %s] Missing BAIT at index %d") %
                                       (eqid, stat, _xx))
                        pass

                    try:
                        fillingdict["DELTA_PREDICTED"] = float(
                               _tmpd["timeUTC_pick"] -
                               metastatdict[eqid][stat]['predicted_picks']
                                                       [_xx][1])
                    except IndexError:
                        logger.warning(("[%s - %s] Missing PRED at index %d") %
                                       (eqid, stat, _xx))
                        pass

                    try:
                        fillingdict["ASSOCIATEDminPREDICTION"] = float(
                          metastatdict[eqid][stat]['bait_picks'][_xx][0] -
                          metastatdict[eqid][stat]['predicted_picks'][_xx][1])
                    except IndexError:
                        pass

                # --- Calculate REFERENCE KEYS
                # Store it ONLY for the first pick if it exist AND if
                # it is not an OTLIER AND the EVALUATION PASSED
                if not AUTOMATICMISSING:
                    try:
                        fpp_refe = automaticlist[_xx][0]
                        fillingdict["REF_PHASE"] = fpp_refe.split('_')[1]
                        fillingdict["REF_CLASS"] = str(pickdict
                                                       [stat]
                                                       [fpp_refe][0]
                                                       ["pickclass"])
                        fillingdict["REF_POLARITY"] = str(pickdict
                                                          [stat]
                                                          [fpp_refe][0]
                                                          ["pickpolar"])

                        # v0.6.8 for csv reason
                        if not fillingdict["REF_POLARITY"].strip():
                            fillingdict["REF_POLARITY"] = None

                        fillingdict["REF_TIME"] = (pickdict
                                                   [stat]
                                                   [fpp_refe][0]
                                                   ["timeUTC_pick"])
                    except IndexError:
                        # OutOfBounds --> No ref for this ADAPT pick
                        fillingdict["REF_PHASE"] = None
                        fillingdict["REF_CLASS"] = None
                        fillingdict["REF_POLARITY"] = None
                        fillingdict["REF_TIME"] = None

                # --- Calculate DELTA
                # Store it ONLY if pick exist AND if
                # it is not an OTLIER AND the EVALUATION PASSED, and if
                # the reference pick exist
                if (not AUTOMATICMISSING and
                   _tmpd["timeUTC_pick"] and
                   not _tmpd["outlier"] and
                   _tmpd["evaluate"] and
                   fillingdict["REF_TIME"]):
                    # MB: QUAKEMISSING refer to the station, not to the pick
                    #     If the station is stored, the P1 is anyway inserted.
                    #     Quake could fail on pick...still
                    fillingdict["DELTA"] = float((_tmpd["timeUTC_pick"] -
                                                 fillingdict["REF_TIME"]))
                else:
                    fillingdict["DELTA"] = None

                df_pick = df_pick.append(fillingdict, ignore_index=True)

        elif not AUTOMATICMISSING:
            # Just take the first Reference phase, so we now it was one
            fpp_refe = automaticlist[0][0]
            if "Seiscomp_P" in fpp_refe:
                fillingdict["REF_PHASE"] = fpp_refe.split('_')[1]
                fillingdict["REF_CLASS"] = str(pickdict
                                               [stat]
                                               [fpp_refe][0]
                                               ["pickclass"])
                fillingdict["REF_POLARITY"] = str(pickdict
                                                  [stat]
                                                  [fpp_refe][0]
                                                  ["pickpolar"])
                # v0.6.8 for csv reason
                if not fillingdict["REF_POLARITY"].strip():
                    fillingdict["REF_POLARITY"] = None

                fillingdict["REF_TIME"] = (pickdict
                                           [stat]
                                           [fpp_refe][0]
                                           ["timeUTC_pick"])
            else:
                # The reference pick IS NOT a P
                fillingdict["REF_PHASE"] = None
                fillingdict["REF_CLASS"] = None
                fillingdict["REF_POLARITY"] = None
                fillingdict["REF_TIME"] = None

            # Added on 0.5.4 to extract features info from MP stage
            # NB!! At the moment is just MP1
            # v0.6.8 -> extract also other multipicking stage based
            #           on the last charachter on mp
            for _mphn, _mphn_val in pickdict[stat].items():
                # # old
                # if _mphn.split("_")[-1].lower() not in ("mp1", "mp2"):
                #     continue  # to the next phase
                # new --> in this case only pos
                if _mphn.split("_")[-1].lower() != "mp1":
                    continue  # to the next phase
                #
                picknm = _mphn.split("_")[0]
                _wd = pickdict[stat][_mphn]
                for _ppx in range(len(_wd)):
                    try:
                        for _ff, _ffv in (
                          _wd[_ppx]['features']["SINGLE_TIME"].items()):
                            fillingdict[picknm.upper()+"_"+str(_ppx+1)+"_"+_ff.upper()] = float(_ffv)
                    except TypeError:
                        # Features SINGLETIMES missing
                        # - Fixed on 0.6.2 (instead of continue, just pass)
                        pass

            df_pick = df_pick.append(fillingdict, ignore_index=True)

        else:
            # Added on 0.5.4 to extract features info from MP stage
            # NB!! At the moment is just MP1
            # v0.6.8 -> extract also other multipicking stage based
            #           on the last charachter on mp
            for _mphn, _mphn_val in pickdict[stat].items():
                # # old
                # if _mphn.split("_")[-1].lower() not in ("mp1", "mp2"):
                #     continue  # to the next phase
                # new
                if _mphn.split("_")[-1].lower() != "mp1":
                    continue  # to the next phase
                #
                picknm = _mphn.split("_")[0]
                _wd = pickdict[stat][_mphn]
                for _ppx in range(len(_wd)):
                    try:
                        for _ff, _ffv in (
                          _wd[_ppx]['features']["SINGLE_TIME"].items()):
                            fillingdict[picknm.upper()+"_"+str(_ppx+1)+"_"+_ff.upper()] = float(_ffv)
                    except TypeError:
                        # Features SINGLETIMES missing
                        # - Fixed on 0.6.2 (instead of continue, just pass)
                        pass

            df_pick = df_pick.append(fillingdict, ignore_index=True)

    # ----- Saving
    if savedf:
        df_pick = df_pick.apply(pd.to_numeric, errors='ignore')
        df_pick.to_csv(os.sep.join([storedir, outfilename]),
                       sep=',',
                       index=False,
                       float_format="%.6f",
                       na_rep="NA")


# # === TIPS
# # Dict comprehension

# # { (some_key if condition else default_key):(something_if_true if condition
# #           else something_if_false) for key, value in dict_.items() }


def adapt_comprehensive_table_tuning(
                              eqid,
                              pickdict,         # ALL of them
                              quakefinalpick,   # FINAL PICKS
                              statdict,
                              metastatdict,
                              storedir,
                              savedf=True,
                              outfilename="WeighterStats.csv"):
    """
    This function will help to understand how the Weighter class works.

    Dumping out all the possible PICKS made by adapt (plus the analisys element),
    and eventually comparyng those with the reference (if it's the case)

    For column attributes check inside the script

    !!! 28-01-2020 The key MyAIC should be ALL CAPS ==> MYAIC
        This is due to the internal change with the upper() method.

    """
    logger.info("Creating COMPREHENSIVE adapt data frame for plotting ...")

    # --------------- Initiliaze the dataframe

    # NB: Always specify the column names!!!
    columns = ["EVENTID",
               "STATION",
               # "CHANNEL", # to be implemented
               "NETWORK",
               "EPIDIST",
               "PHASE",
               "POLARITY",
               "CLASS",
               "TIME",
               "REF_PHASE",
               "REF_CLASS",
               "REF_POLARITY",
               "REF_TIME",
               #
               "DELTA",  # Always adapt- Reference
               "MEAN",
               "MEDIAN",
               "MEANminMEDIAN",
               "STD",
               "VAR",
               "TOT_OBS",
               "STABLE_PICKERS",
               "SPARE_OBS",
               "VALID_OBS",
               "OUTLIERS",
               # MB: 24-10-2019 for ML problem
               "DELTA_ASSOCIATED",
               "DELTA_PREDICTED",
               "ASSOCIATEDminPREDICTION",

               # MB: New v0.7.4: chenged pickers tag HOS + AIC
               "AIC_BK_1",
               "AIC_BK_2",
               "AIC_FP_1",
               "AIC_FP_2",
               "AIC_HOS_1",
               "AIC_HOS_2",
               "BK_FP_1",
               "BK_FP_2",
               "HOS_BK_1",
               "HOS_BK_2",
               "HOS_FP_1",
               "HOS_FP_2",

               # "FP_BK_1",
               # "FP_MYAIC_1",
               # "FP_KURT_1",
               # "BK_MYAIC_1",
               # "BK_KURT_1",
               # "MYAIC_KURT_1",
               # "KURT_MYAIC_1",
               # "FP_BK_2",
               # "FP_MYAIC_2",
               # "FP_KURT_2",
               # "BK_MYAIC_2",
               # "BK_KURT_2",
               # "MYAIC_KURT_2",
               # "KURT_MYAIC_2",

               # MB: 28-01-2020 for CLASSIFICATIN and ML problem.
               "AIC_1_MAX_SIGNAL_TO_NOISE_RATIO",
               "AIC_1_MEAN_SIGNAL_TO_NOISE_RATIO",
               "AIC_1_NOISE_OVER_THRESHOLD",
               "AIC_1_DOMINANT_FREQUENCIES_SHIFT",
               "AIC_1_DOMINANT_FREQUENCIES_RATIO",
               #
               "AIC_2_MAX_SIGNAL_TO_NOISE_RATIO",
               "AIC_2_MEAN_SIGNAL_TO_NOISE_RATIO",
               "AIC_2_NOISE_OVER_THRESHOLD",
               "AIC_2_DOMINANT_FREQUENCIES_SHIFT",
               "AIC_2_DOMINANT_FREQUENCIES_RATIO",
               #
               "HOS_1_MAX_SIGNAL_TO_NOISE_RATIO",
               "HOS_1_MEAN_SIGNAL_TO_NOISE_RATIO",
               "HOS_1_NOISE_OVER_THRESHOLD",
               "HOS_1_DOMINANT_FREQUENCIES_SHIFT",
               "HOS_1_DOMINANT_FREQUENCIES_RATIO",
               #
               "HOS_2_MAX_SIGNAL_TO_NOISE_RATIO",
               "HOS_2_MEAN_SIGNAL_TO_NOISE_RATIO",
               "HOS_2_NOISE_OVER_THRESHOLD",
               "HOS_2_DOMINANT_FREQUENCIES_SHIFT",
               "HOS_2_DOMINANT_FREQUENCIES_RATIO",
               #
               "FP_1_MAX_SIGNAL_TO_NOISE_RATIO",
               "FP_1_MEAN_SIGNAL_TO_NOISE_RATIO",
               "FP_1_NOISE_OVER_THRESHOLD",
               "FP_1_DOMINANT_FREQUENCIES_SHIFT",
               "FP_1_DOMINANT_FREQUENCIES_RATIO",
               #
               "FP_2_MAX_SIGNAL_TO_NOISE_RATIO",
               "FP_2_MEAN_SIGNAL_TO_NOISE_RATIO",
               "FP_2_NOISE_OVER_THRESHOLD",
               "FP_2_DOMINANT_FREQUENCIES_SHIFT",
               "FP_2_DOMINANT_FREQUENCIES_RATIO",
               #
               "BK_1_MAX_SIGNAL_TO_NOISE_RATIO",
               "BK_1_MEAN_SIGNAL_TO_NOISE_RATIO",
               "BK_1_NOISE_OVER_THRESHOLD",
               "BK_1_DOMINANT_FREQUENCIES_SHIFT",
               "BK_1_DOMINANT_FREQUENCIES_RATIO",
               #
               "BK_2_MAX_SIGNAL_TO_NOISE_RATIO",
               "BK_2_MEAN_SIGNAL_TO_NOISE_RATIO",
               "BK_2_NOISE_OVER_THRESHOLD",
               "BK_2_DOMINANT_FREQUENCIES_SHIFT",
               "BK_2_DOMINANT_FREQUENCIES_RATIO",
               #
               "MP_MAX_SIGNAL_TO_NOISE_RATIO",
               "MP_MEAN_SIGNAL_TO_NOISE_RATIO",
               "MP_MAX_SIGNAL_TO_LONGNOISE_RATIO",
               "MP_MEAN_SIGNAL_TO_LONGNOISE_RATIO",
               "MP_MAX_SIGNAL_TO_STARTNOISE_RATIO",
               "MP_MEAN_SIGNAL_TO_STARTNOISE_RATIO",
               "MP_MAX_LONGNOISE_TO_STARTNOISE_RATIO",
               "MP_MEAN_LONGNOISE_TO_STARTNOISE_RATIO",
               "MP_SIGNAL_BELOW_THRESHOLD",
               "MP_NOISE_OVER_THRESHOLD",
               "MP_ABSOLUTE_ENERGY_NOISE",
               "MP_ABSOLUTE_ENERGY_SIGNAL",
               "MP_STD_NOISE",
               "MP_STD_SIGNAL",
               "MP_DOMINANT_FREQUENCIES_SHIFT",
               "MP_DOMINANT_FREQUENCIES_RATIO",
               "MP_DOMINANT_FREQUENCY_NOISE",
               "MP_DOMINANT_FREQUENCY_SIGNAL",
               # Additional
               "MODE",
               "MODEALT",
               "BOOTMODE",
               "BOOTMEAN",
               "BOOTMADMODE",
               "DELTAbootmode",
               # Additional v0.7.20
               "MMD",
               #
               "PICKERS_INVOLVED"
               ]

    # 03-02-2020 added noise over threshold

    # 13-02-2020 added frequency features

    # 18-02-2020 added final MP features --> I was receiving this error:
    #       `ValueError: cannot reindex from a duplicate axis`
    # https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis-mean:
    # - For people who are still struggling with this error, it can also
    # happen if you accidentally create a duplicate column with the same
    # name. Remove duplicate columns like so:
    # >>> df = df.loc[:,~df.columns.duplicated()]

    df_pick = pd.DataFrame(columns=columns)
    df_pick_one = pd.DataFrame(columns=columns)
    df_pick_two = pd.DataFrame(columns=columns)

    # From weighter class
    # self.triage_dict = {'tot_obs': 0,
    #                     'valid_obs': [],
    #                     'spare_obs': [],
    #                     'stable_pickers': [],
    #                     'outliers': []}
    # self.triage_results = {'mean': None,
    #                        'median': None,
    #                        'mean-median': None,
    #                        'std': None,
    #                        'var': None}

    # FOR THE MOMENT I DON'T CARE ABOUT 2 REFERENCE...I'll find the way
    # I will store anyway the P1P2 from adapt

    # --------------- Looping
    # I want to extract ALL THE POSSIBLE ONE, EVEN THE UNPICKED by both

    # v0.6.30: This statistics is not affected from the new calculation
    # of ALL epicentral distance in Inventory. This is because this statistics
    # goes with PickContainer instead!
    for stat in pickdict:
        fillingdict = {ii: None for ii in columns}
        # fillingdict = {}
        QUAKEMISSING = False
        REFERENCEMISSING = False

        # --- collecting STAT info
        try:
            fillingdict["EVENTID"] = eqid
            fillingdict["STATION"] = stat
            # fillingdict["CHANNEL"] = statdict[stat]["channel"] # to be implemented
            fillingdict["NETWORK"] = statdict[stat]["network"]
            fillingdict["EPIDIST"] = "{:6.2f}".format(metastatdict
                                                      [eqid]
                                                      [stat]
                                                      ["epidist"])
        except KeyError as err:
            # ReallyBAD --> continue or throw error
            logger.error("STAT %s  ... missing from inventory" % (err.args[0]))
            raise QE.MissingVariable()

        # For a same station we have possibly MULTIPLE PHASENAME (P1,P2)
        # Check IT beforehand <---
        try:
            n_final_phase = quakefinalpick[stat].keys()
        except KeyError as err:
            logger.warning("STAT %s Missing ADAPT PICK" % (err.args[0]))
            QUAKEMISSING = True

        try:
            if metastatdict[eqid][stat]["reference_picks"]:
                referencelist = (metastatdict[eqid][stat]["reference_picks"])
            else:
                # This station was not picked --> no data
                referencelist = []
                logger.warning("STAT %s Missing REFERENCE" % stat)
                REFERENCEMISSING = True

        except KeyError as err:
            # This station was not picked --> no data
            logger.warning("STAT %s Missing REFERENCE" % (err.args[0]))
            REFERENCEMISSING = True

        # =========> Here WE RUN THROUGH all the phases <============
        if not QUAKEMISSING:
            for _xx, _mph in enumerate(n_final_phase):
                _tmpd = quakefinalpick[stat][_mph][0]  # In the ADAPTFINAL dict there's should be only 1 pick per phase
                # Main
                fillingdict["PHASE"] = _mph
                fillingdict["POLARITY"] = _tmpd['pickpolar']
                fillingdict["CLASS"] = _tmpd['pickclass']
                fillingdict["TIME"] = _tmpd["timeUTC_pick"]

                # --- The TRIAGE RESULTS could not be there ...
                _triage_results = _tmpd["weight"].get_triage_results()
                if _triage_results:
                    for _kk, vv in _triage_results.items():
                        if _kk == "mean-median":
                            fillingdict["MEANminMEDIAN"] = vv
                        else:
                            fillingdict[_kk.upper()] = vv
                else:
                    fillingdict["MEAN"] = None
                    fillingdict["MEDIAN"] = None
                    fillingdict["MEANminMEDIAN"] = None
                    fillingdict["STD"] = None
                    fillingdict["VAR"] = None
                    fillingdict["MAD"] = None
                    fillingdict["AAD"] = None
                    fillingdict["MMD"] = None
                    fillingdict["MODE"] = None
                    fillingdict["MODEALT"] = None
                    fillingdict["BOOTMODE"] = None
                    fillingdict["BOOTMEAN"] = None
                    fillingdict["BOOTMADMODE"] = None

                # --- The TRIAGE DICT should always be there ...
                for _kk, vv in _tmpd["weight"].get_triage_dict().items():
                    if _kk in ('valid_obs',
                               'spare_obs',
                               'stable_pickers',
                               'outliers',
                               'pickers_involved'):
                        fillingdict[_kk.upper()] = len(vv)
                    elif _kk in ('failed_pickers'):
                        pass
                    else:
                        fillingdict[_kk.upper()] = vv
                        # {'tot_obs': 0,
                        #   'valid_obs': [],
                        #   'spare_obs': [],
                        #   'stable_pickers': [],
                        #   'outliers': [],
                        #   'failed_pickers': [],
                        #   'pickers_involved': []}
                        # pairs = [('FP', 'BK'), ('FP', 'MyAIC'),
                        #          ('FP', 'KURT'), ('BK', 'MyAIC'),
                        #          ('BK', 'KURT'), ('MyAIC', 'KURT'),
                        #          ('KURT', 'MyAIC')]

                # Added on 0.5.4 to extract features info from MP stage
                # NB!! At the moment is just MP1
                # v0.6.8 -> extract also other multipicking stage based
                #           on the last charachter on mp
                for _mphn, _mphn_val in pickdict[stat].items():
                    # # old
                    # if _mphn.split("_")[-1].lower() not in ("mp1", "mp2"):
                    #     continue  # to the next phase
                    # new
                    if _mphn.split("_")[-1].lower() != "mp" + _mph[-1]:
                        continue  # to the next phase
                    #
                    picknm = _mphn.split("_")[0]
                    _wd = pickdict[stat][_mphn]
                    for _ppx in range(len(_wd)):
                        try:
                            for _ff, _ffv in (
                              _wd[_ppx]['features']["SINGLE_TIME"].items()):
                                fillingdict[picknm.upper()+"_"+str(_ppx+1)+"_"+_ff.upper()] = float(_ffv)
                        except TypeError:
                            # Features SINGLETIMES missing
                            # - Fixed on 0.6.2 (instead of continue, just pass)
                            pass

                # Added on 0.5.9 to extract features info from FINAL stage
                try:
                    for _ff, _ffv in (
                      _tmpd['features']["SINGLE_TIME"].items()):
                        fillingdict["MP_"+_ff.upper()] = _ffv
                except TypeError:
                    # Features SINGLETIMES missing
                    # - Fixed on 0.6.2 (instead of continue, just pass)
                    pass

                # --- Calculate DELTA BETWEEN Associated BAIT and PREDICTED
                # MB: if there's a ADAPT pick, there's for sure a BAIT
                if _tmpd["timeUTC_pick"]:
                    try:
                        fillingdict["DELTA_ASSOCIATED"] = float(
                                    _tmpd["timeUTC_pick"] -
                                    metastatdict[eqid][stat]['bait_picks']
                                                            [_xx][0])
                    except IndexError:
                        logger.warning(("[%s - %s] Missing BAIT at index %d") %
                                       (eqid, stat, _xx))
                        pass

                    try:
                        fillingdict["DELTA_PREDICTED"] = float(
                               _tmpd["timeUTC_pick"] -
                               metastatdict[eqid][stat]['predicted_picks']
                                                       [_xx][1])
                    except IndexError:
                        logger.warning(("[%s - %s] Missing PRED at index %d") %
                                       (eqid, stat, _xx))
                        pass

                    try:
                        fillingdict["ASSOCIATEDminPREDICTION"] = float(
                          metastatdict[eqid][stat]['bait_picks'][_xx][0] -
                          metastatdict[eqid][stat]['predicted_picks'][_xx][1])
                    except IndexError:
                        pass

                # --- Calculate REFERENCE KEYS
                # Store it ONLY for the first pick if it exist AND if
                # it is not an OTLIER AND the EVALUATION PASSED
                if not REFERENCEMISSING:
                    try:
                        fpp_refe = referencelist[_xx][0]
                        #
                        if "Reference_P" in fpp_refe:
                            fillingdict["REF_PHASE"] = fpp_refe.split('_')[1]
                            fillingdict["REF_CLASS"] = str(pickdict
                                                           [stat]
                                                           [fpp_refe][0]
                                                           ["pickclass"])
                            fillingdict["REF_POLARITY"] = str(pickdict
                                                              [stat]
                                                              [fpp_refe][0]
                                                              ["pickpolar"])
                            # v0.6.8 for csv reason
                            if not fillingdict["REF_POLARITY"].strip():
                                fillingdict["REF_POLARITY"] = None

                            fillingdict["REF_TIME"] = (pickdict
                                                       [stat]
                                                       [fpp_refe][0]
                                                       ["timeUTC_pick"])

                        else:
                            # The reference pick IS NOT a P
                            fillingdict["REF_PHASE"] = None
                            fillingdict["REF_CLASS"] = None
                            fillingdict["REF_POLARITY"] = None
                            fillingdict["REF_TIME"] = None
                    except IndexError:
                        # OutOfBounds --> No ref for this ADAPT pick
                        fillingdict["REF_PHASE"] = None
                        fillingdict["REF_CLASS"] = None
                        fillingdict["REF_POLARITY"] = None
                        fillingdict["REF_TIME"] = None

                # --- Calculate DELTA
                # Store it ONLY if pick exist AND if
                # it is not an OTLIER AND the EVALUATION PASSED, and if
                # the reference pick exist
                if (not REFERENCEMISSING and
                   _tmpd["timeUTC_pick"] and
                   not _tmpd["outlier"] and
                   _tmpd["evaluate"] and
                   fillingdict["REF_TIME"]):
                    # MB: QUAKEMISSING refer to the station, not to the pick
                    #     If the station is stored, the P1 is anyway inserted.
                    #     Quake could fail on pick...still
                    fillingdict["DELTA"] = float((_tmpd["timeUTC_pick"] -
                                                 fillingdict["REF_TIME"]))
                    # fillingdict["DELTAbootmode"] = float((_tmpd["timeUTC_pick"] -
                    #                                      fillingdict["BOOTMODE"]))
                    fillingdict["DELTAbootmode"] = float((fillingdict["BOOTMODE"] -
                                                          fillingdict["REF_TIME"]))
                else:
                    fillingdict["DELTA"] = None
                    fillingdict["DELTAbootmode"] = None

                # Decide which dataframe store based on Nth REFERENCE P*
                if [True if "Reference_P" in _xx[0] else False for _xx in referencelist].count(True) >= 2:
                    df_pick_two = df_pick_two.append(fillingdict, ignore_index=True)
                else:
                    # This dictionary will take care of SIngle Reference P + additional picked station by ADAPT
                    df_pick_one = df_pick_one.append(fillingdict, ignore_index=True)
                #
                df_pick = df_pick.append(fillingdict, ignore_index=True)
                # =========================================

        elif not REFERENCEMISSING:
            # Just take the first Reference phase, so we now it was one
            fpp_refe = referencelist[0][0]
            if "Reference_P" in fpp_refe:
                fillingdict["REF_PHASE"] = fpp_refe.split('_')[1]
                fillingdict["REF_CLASS"] = str(pickdict
                                               [stat]
                                               [fpp_refe][0]
                                               ["pickclass"])
                fillingdict["REF_POLARITY"] = str(pickdict
                                                  [stat]
                                                  [fpp_refe][0]
                                                  ["pickpolar"])
                # v0.6.8 for csv reason
                if not fillingdict["REF_POLARITY"].strip():
                    fillingdict["REF_POLARITY"] = None

                fillingdict["REF_TIME"] = (pickdict
                                           [stat]
                                           [fpp_refe][0]
                                           ["timeUTC_pick"])
            else:
                # The reference pick IS NOT a P
                fillingdict["REF_PHASE"] = None
                fillingdict["REF_CLASS"] = None
                fillingdict["REF_POLARITY"] = None
                fillingdict["REF_TIME"] = None

            # Added on 0.5.4 to extract features info from MP stage
            # NB!! At the moment is just MP1
            # v0.6.8 -> extract also other multipicking stage based
            #           on the last charachter on mp
            for _mphn, _mphn_val in pickdict[stat].items():
                # # old
                # if _mphn.split("_")[-1].lower() not in ("mp1", "mp2"):
                #     continue  # to the next phase
                # new --> in this case only pos
                if _mphn.split("_")[-1].lower() != "mp1":
                    continue  # to the next phase
                #
                picknm = _mphn.split("_")[0]
                _wd = pickdict[stat][_mphn]
                for _ppx in range(len(_wd)):
                    try:
                        for _ff, _ffv in (
                          _wd[_ppx]['features']["SINGLE_TIME"].items()):
                            fillingdict[picknm.upper()+"_"+str(_ppx+1)+"_"+_ff.upper()] = float(_ffv)
                    except TypeError:
                        # Features SINGLETIMES missing
                        # - Fixed on 0.6.2 (instead of continue, just pass)
                        pass

            # Decide which dataframe store based on Nth REFERENCE P*
            if [True if "Reference_P" in _xx[0] else False for _xx in referencelist].count(True) >= 2:
                df_pick_two = df_pick_two.append(fillingdict, ignore_index=True)
            else:
                # This dictionary will take care of SIngle Reference P + additional picked station by ADAPT
                df_pick_one = df_pick_one.append(fillingdict, ignore_index=True)
            #
            df_pick = df_pick.append(fillingdict, ignore_index=True)

        else:
            # Added on 0.5.4 to extract features info from MP stage
            # NB!! At the moment is just MP1
            # v0.6.8 -> extract also other multipicking stage based
            #           on the last charachter on mp
            for _mphn, _mphn_val in pickdict[stat].items():
                # # old
                # if _mphn.split("_")[-1].lower() not in ("mp1", "mp2"):
                #     continue  # to the next phase
                # new
                if _mphn.split("_")[-1].lower() != "mp1":
                    continue  # to the next phase
                #
                picknm = _mphn.split("_")[0]
                _wd = pickdict[stat][_mphn]
                for _ppx in range(len(_wd)):
                    try:
                        for _ff, _ffv in (
                          _wd[_ppx]['features']["SINGLE_TIME"].items()):
                            fillingdict[picknm.upper()+"_"+str(_ppx+1)+"_"+_ff.upper()] = float(_ffv)
                    except TypeError:
                        # Features SINGLETIMES missing
                        # - Fixed on 0.6.2 (instead of continue, just pass)
                        pass

            # Decide which dataframe store based on Nth REFERENCE P*
            if [True if "Reference_P" in _xx[0] else False for _xx in referencelist].count(True) >= 2:
                df_pick_two = df_pick_two.append(fillingdict, ignore_index=True)
            else:
                # This dictionary will take care of SIngle Reference P + additional picked station by ADAPT
                df_pick_one = df_pick_one.append(fillingdict, ignore_index=True)
            #
            df_pick = df_pick.append(fillingdict, ignore_index=True)

    # ----- Saving
    if savedf:
        df_pick = df_pick.apply(pd.to_numeric, errors='ignore')
        df_pick.to_csv(os.sep.join([storedir, outfilename]),
                       sep=',',
                       index=False,
                       float_format="%.6f",
                       na_rep="NA")
        #
        df_pick_one = df_pick_one.apply(pd.to_numeric, errors='ignore')
        df_pick_one.to_csv(os.sep.join([storedir, "SINGLE_P_ReferencePick.csv"]),
                           sep=',',
                           index=False,
                           float_format="%.6f",
                           na_rep="NA")
        #
        df_pick_two = df_pick_two.apply(pd.to_numeric, errors='ignore')
        df_pick_two.to_csv(os.sep.join([storedir, "MULTI_P_ReferencePick.csv"]),
                           sep=',',
                           index=False,
                           float_format="%.6f",
                           na_rep="NA")


def final_automatic_statistics(eqid,
                               picks,  # The COMPLETE one
                               stations_metadata,
                               stations,
                               radius,
                               outfilename="Automatic_STATS.stats",
                               export_gmt_files=True,
                               export_gmt_files_basedir=".",
                               additional_statistics=[]):
    """ This function will create a file on EVID home with all necessary
        information about the automatic procedeure.
        NB: to be used only in PRODUCTION MODE
    """
    with open(outfilename, "w") as OUT:
        OUT.write("@@@ Event KPID" + os.linesep)
        OUT.write(eqid + os.linesep)

        OUT.write("@@@ Total STATIONS SELECTED From Inventory:" + os.linesep)
        # v0.6.30: the previous assumption (len of metaStadDict)
        # is not valid anymore. A tag 'isselected' has been added.
        stat_quake_selected = set(
                            [_xx for _xx in stations_metadata[eqid].keys()
                             if stations_metadata[eqid][_xx]["isselected"]])
        OUT.write(("%d" + os.linesep) % len(stat_quake_selected))

        OUT.write("@@@ Total STATIONS DOWNLOADED Has Data:" + os.linesep)
        # This refer to the ACTUALLY DOWNLOADED from the above pool
        OUT.write(("%d" + os.linesep) %
                  len([_xx for _xx in stations_metadata[eqid].keys()
                      if stations_metadata[eqid][_xx]["isdownloaded"]]))

        #
        chan_test_list = []
        for _xx in stations_metadata[eqid].keys():
            if stations_metadata[eqid][_xx]["missingchannel"]:
                if "Z" in stations_metadata[eqid][_xx]["missingchannel"]:
                    chan_test_list.append(True)

        OUT.write("@@@ Total STATIONS Channel Test FAILED (*Z):" + os.linesep)
        # This refer to the ACTUALLY DOWNLOADED from the above pool
        OUT.write(("%d" + os.linesep) % (np.sum(chan_test_list)))

        stream_test_list = []
        for _xx in stations_metadata[eqid].keys():
            if stations_metadata[eqid][_xx]["ispickable"]:
                stream_test_list.append(True)
            else:
                stream_test_list.append(False)
        OUT.write("@@@ Total STATIONS Stream Quality Test PASSED:" + os.linesep)
        # If no Z channel is found in STREAM, it's rejected (not worthy to pick)
        OUT.write(("%d" + os.linesep) % np.sum(stream_test_list))

        OUT.write("@@@ Total STATIONS from SC3 input file ALL:" + os.linesep)
        # This refer to the INPUT SC3/REFERENCE stations listed in MANUPICK
        auto_stat_all = set([_xx for _xx in stations_metadata[eqid].keys()
                             if stations_metadata[eqid][_xx]["isautomatic"]])
        OUT.write(("%d" + os.linesep) % len(auto_stat_all))

        OUT.write("@@@ Total STATIONS from SC3 input file INSIDE RADIUS:" +
                  os.linesep)
        # This refer to the INPUT SC3/REFERENCE stations listed in MANUPICK
        # falling inside the selction radius in ADAPT
        auto_stat = set([_xx for _xx in auto_stat_all
                         if stations_metadata[eqid][_xx]["epidist"] <= radius])
        OUT.write(("%d" + os.linesep) % len(auto_stat))

        #
        _p1 = 0
        _p1_out = 0
        _p1_phase = 0
        _p1_reject = 0
        _sta_judger = []
        for _sta, _dict in picks.items():
            matched = set(("P1", "*P1", "~P1")).intersection(picks[_sta].keys())
            if matched:
                _sta_judger.append(_sta)  # This station ARRIVED at the court!
                for _pp in matched:
                    ptime = picks[_sta][_pp][0]["timeUTC_pick"]
                    #
                    if _pp == "P1":
                        if ptime:
                            _p1 += 1
                        else:
                            _p1_reject += 1
                    elif _pp == "*P1":
                        if ptime:
                            _p1_out += 1
                        else:
                            _p1_reject += 1
                    elif _pp == "~P1":
                        if ptime:
                            _p1_phase += 1
                        else:
                            _p1_reject += 1
        #
        OUT.write("@@@ Total STATIONS analyzed by JUDGER (after MP):" + os.linesep)
        OUT.write(("%d" + os.linesep) % len(_sta_judger))
        # The following numbers represent all the possible type of validobs
        OUT.write("@@@ Total P1 picked:" + os.linesep)
        OUT.write(("%d" + os.linesep) % _p1)
        OUT.write("@@@ Total *P1 picked:" + os.linesep)
        OUT.write(("%d" + os.linesep) % _p1_out)
        OUT.write("@@@ Total ~P1 picked:" + os.linesep)
        OUT.write(("%d" + os.linesep) % _p1_phase)
        OUT.write("@@@ Total MP results rejected:" + os.linesep)
        OUT.write(("%d" + os.linesep) % _p1_reject)
        # If the above numbers DOESN'T SUM UP with `Total STATIONS JUDGER`
        # it meanse that the difference stations are missing due to
        #  either for: MISSING BAIT or TOTAL-MPobs < 3

        OUT.write("@@@ Total COMMON STATIONS picked by SC3 and ADAPT:" + os.linesep)
        # Common stations analyzed by SC3
        common = list(set(auto_stat).intersection(_sta_judger))
        OUT.write(("%d" + os.linesep) % len(common))

        # COMMON FINAL P1
        compone = []
        for _sta in common:
            for _pp in picks[_sta].keys():
                if _pp == "P1" and picks[_sta][_pp][0]["timeUTC_pick"]:
                    compone.append(_sta)
        OUT.write("@@@ Total COMMON PICKED as a P1:" + os.linesep)
        # Common stations that are picked both by SC3 and ADAPT (as P1 [anycase])
        OUT.write(("%d" + os.linesep) % len(compone))

        # ADDICTIONAL STATISTICS:
        #  The object must be a list of list (or iterable) containing
        #  [0] -> the message
        #  [1] -> the integer value
        if additional_statistics:
            for _as in additional_statistics:
                OUT.write("@@@ " + _as[0] + os.linesep)
                OUT.write(("%d" + os.linesep) % int(_as[1]))

    # GMT
    if export_gmt_files:
        with open(os.sep.join([export_gmt_files_basedir, "Selected_ADAPT.gmt"]), "w") as QUAKESEL:
            for _ss in stations_metadata[eqid].keys():
                QUAKESEL.write(("%9.5f %8.5f %8.5f %s"+os.linesep) % (
                                                    stations[_ss]["lon"],
                                                    stations[_ss]["lat"],
                                                    stations[_ss]["elev_m"] / 10**3,
                                                    stations[_ss]["fullname"]))
        #
        with open(os.sep.join([export_gmt_files_basedir, "Downloaded_ADAPT.gmt"]), "w") as QUAKEDOWN:
            for _ss in [_xx for _xx in stations_metadata[eqid].keys()
                        if stations_metadata[eqid][_xx]["isdownloaded"]]:
                QUAKEDOWN.write(("%9.5f %8.5f %8.5f %s"+os.linesep) % (
                                                    stations[_ss]["lon"],
                                                    stations[_ss]["lat"],
                                                    stations[_ss]["elev_m"] / 10**3,
                                                    stations[_ss]["fullname"]))
        #
        with open(os.sep.join([export_gmt_files_basedir, "Judger_ADAPT.gmt"]), "w") as QUAKEJUDGE:
            for _ss in _sta_judger:
                QUAKEJUDGE.write(("%9.5f %8.5f %8.5f %s"+os.linesep) % (
                                                    stations[_ss]["lon"],
                                                    stations[_ss]["lat"],
                                                    stations[_ss]["elev_m"] / 10**3,
                                                    stations[_ss]["fullname"]))
        #
        with open(os.sep.join([export_gmt_files_basedir, "Manupick_SC3_all.gmt"]), "w") as SC3MANUPICKALL:
            for _ss in auto_stat_all:
                SC3MANUPICKALL.write(("%9.5f %8.5f %8.5f %s"+os.linesep) % (
                                                    stations[_ss]["lon"],
                                                    stations[_ss]["lat"],
                                                    stations[_ss]["elev_m"] / 10**3,
                                                    stations[_ss]["fullname"]))
        #
        with open(os.sep.join([export_gmt_files_basedir, "Manupick_SC3_radius.gmt"]), "w") as SC3MANUPICKRADIUS:
            for _ss in auto_stat:
                SC3MANUPICKRADIUS.write(("%9.5f %8.5f %8.5f %s"+os.linesep) % (
                                                    stations[_ss]["lon"],
                                                    stations[_ss]["lat"],
                                                    stations[_ss]["elev_m"] / 10**3,
                                                    stations[_ss]["fullname"]))
        #
        with open(os.sep.join([export_gmt_files_basedir, "CommonPicked_SC3_ADAPT_P1.gmt"]), "w") as COMMONPICK:
            for _ss in compone:
                COMMONPICK.write(("%9.5f %8.5f %8.5f %s"+os.linesep) % (
                                                    stations[_ss]["lon"],
                                                    stations[_ss]["lat"],
                                                    stations[_ss]["elev_m"] / 10**3,
                                                    stations[_ss]["fullname"]))


def final_bait_statistics(eqid, bait_obj, allpicks,
                          storedir="./BaIt_Statistics"):
    """ This function will create all the csv (and file objects) for
        all the EVENTS/STATIONS pairs run.
        It will create a file for each test (inside for each pick of bait)
        and also a file containing the overall statistics
    """
    # Check store dir
    if not os.path.isdir(storedir):
        os.makedirs(storedir, exist_ok=True)
    #
    logger.info("Producing FINAL BaIt statistics")
    FINALRES = {'1': {
                        'AMP_test': pd.DataFrame(columns=(
                                                    'EQID', 'STAT',
                                                    'RESIDUAL', 'VALUE',
                                                    'TESTPASS', 'PICKPASS')),
                        'SUSTAIN_test': pd.DataFrame(columns=(
                                                    'EQID', 'STAT',
                                                    'RESIDUAL', 'VALUE',
                                                    'TESTPASS', 'PICKPASS')),
                        'LOWFREQ_test': pd.DataFrame(columns=(
                                                    'EQID', 'STAT',
                                                    'RESIDUAL', 'VALUE',
                                                    'TESTPASS', 'PICKPASS')),
                        'BAIT_BKmissed': {
                                'count': 0,
                                'EventStation': pd.DataFrame(columns=('EQID', 'STAT'))
                                },
                        'TOT_AMP_test_fail': 0,
                        'TOT_AMP_test_pass': 0,
                        'TOT_SUSTAIN_test_fail': 0,
                        'TOT_SUSTAIN_test_pass': 0,
                        'TOT_LOWFREQ_test_fail': 0,
                        'TOT_LOWFREQ_test_pass': 0
                    },
                '2': {
                        'AMP_test': pd.DataFrame(columns=(
                                                    'EQID', 'STAT',
                                                    'RESIDUAL', 'VALUE',
                                                    'TESTPASS', 'PICKPASS')),
                        'SUSTAIN_test': pd.DataFrame(columns=(
                                                    'EQID', 'STAT',
                                                    'RESIDUAL', 'VALUE',
                                                    'TESTPASS', 'PICKPASS')),
                        'LOWFREQ_test': pd.DataFrame(columns=(
                                                    'EQID', 'STAT',
                                                    'RESIDUAL', 'VALUE',
                                                    'TESTPASS', 'PICKPASS')),
                        'BAIT_BKmissed': {
                                'count': 0,
                                'EventStation': pd.DataFrame(columns=('EQID', 'STAT'))
                                },
                        'TOT_AMP_test_fail': 0,
                        'TOT_AMP_test_pass': 0,
                        'TOT_SUSTAIN_test_fail': 0,
                        'TOT_SUSTAIN_test_pass': 0,
                        'TOT_LOWFREQ_test_fail': 0,
                        'TOT_LOWFREQ_test_pass': 0
                    },
                '3': {
                        'AMP_test': pd.DataFrame(columns=(
                                                    'EQID', 'STAT',
                                                    'RESIDUAL', 'VALUE',
                                                    'TESTPASS', 'PICKPASS')),
                        'SUSTAIN_test': pd.DataFrame(columns=(
                                                    'EQID', 'STAT',
                                                    'RESIDUAL', 'VALUE',
                                                    'TESTPASS', 'PICKPASS')),
                        'LOWFREQ_test': pd.DataFrame(columns=(
                                                    'EQID', 'STAT',
                                                    'RESIDUAL', 'VALUE',
                                                    'TESTPASS', 'PICKPASS')),
                        'BAIT_BKmissed': {
                                'count': 0,
                                'EventStation': pd.DataFrame(columns=('EQID', 'STAT'))
                                },
                        'TOT_AMP_test_fail': 0,
                        'TOT_AMP_test_pass': 0,
                        'TOT_SUSTAIN_test_fail': 0,
                        'TOT_SUSTAIN_test_pass': 0,
                        'TOT_LOWFREQ_test_fail': 0,
                        'TOT_LOWFREQ_test_pass': 0
                    },
                'BAIT_trimFAILURE': {
                        'count': 0,
                        'EventStation': pd.DataFrame(columns=('EQID', 'STAT'))
                        }
                }

    accepted_count = 0
    rejected_count = 0
    for _xx, (_ss, _bb) in enumerate(bait_obj.items()):
        # ======== TRIM EVALUATION FAILED
        try:
            wd = _bb.baitdict
        except AttributeError:
            # Empty stream after BAIT EVAL TRIM
            FINALRES['BAIT_trimFAILURE']['count'] += 1
            FINALRES['BAIT_trimFAILURE']['EventStation'] = (
                FINALRES['BAIT_trimFAILURE']['EventStation'].append(
                    {'EQID': eqid, 'STAT': _ss}, ignore_index=True)
            )
            continue

        # ======== General Pass or not
        _check_valid = _bb.extract_true_pick()
        if not _check_valid:
            rejected_count += 1
        else:
            accepted_count += 1

        # ======================= LOOP OVER BAIT
        # =======================
        for _yy in range(1, 4):
            try:
                wd = _bb.baitdict[str(_yy)]
            except KeyError:
                # Bait FINISHED picking --> the first should always be there
                break  # -> next bait obj

            # ======== First run of BK fails
            if not wd['pickUTC']:
                FINALRES[str(_yy)]['BAIT_BKmissed']['count'] += 1
                FINALRES[str(_yy)]['BAIT_BKmissed']['EventStation'] = (
                    FINALRES[str(_yy)]['BAIT_BKmissed']['EventStation'].append(
                        {'EQID': eqid, 'STAT': _ss}, ignore_index=True)
                )
                continue

            # ======== EXTRACT INFOS
            else:
                if wd['evaluatePick']:
                    _accepted = 1
                else:
                    _accepted = 0
                #
                _residual = compare_times_bait_predicted(
                                wd['pickUTC'], _ss, allpicks)

                # --- AmpTest
                fdAMP = {}
                fdAMP["EQID"] = eqid
                fdAMP["STAT"] = _ss

                if wd['evaluatePick_tests']['SignalAmp'][0]:
                    fdAMP["TESTPASS"] = 1
                    FINALRES[str(_yy)]['TOT_AMP_test_pass'] += 1
                else:
                    fdAMP["TESTPASS"] = 0
                    FINALRES[str(_yy)]['TOT_AMP_test_fail'] += 1

                fdAMP["VALUE"] = wd['evaluatePick_tests']['SignalAmp'][1]
                fdAMP["PICKPASS"] = _accepted
                fdAMP["RESIDUAL"] = _residual
                FINALRES[str(_yy)]['AMP_test'] = (
                    FINALRES[str(_yy)]['AMP_test'].append(
                                    fdAMP, ignore_index=True)
                )

                # --- SustainTest:
                # I extract the first window BELOW the ratio threshold
                rt = 1.25
                fdSUST = {}
                fdSUST["EQID"] = eqid
                fdSUST["STAT"] = _ss

                if wd['evaluatePick_tests']['SignalSustain'][0]:
                    fdSUST["TESTPASS"] = 1
                    FINALRES[str(_yy)]['TOT_SUSTAIN_test_pass'] += 1
                else:
                    fdSUST["TESTPASS"] = 0
                    FINALRES[str(_yy)]['TOT_SUSTAIN_test_fail'] += 1

                if not fdSUST["TESTPASS"]:
                    # Search where if failed
                    fdSUST["VALUE"] = [_cc for _cc, _vv in enumerate(wd['evaluatePick_tests']['SignalSustain'][1])
                                       if _vv <= rt][0]
                else:
                    # Otherwise take the last window as reference
                    fdSUST["VALUE"] = len(wd['evaluatePick_tests']['SignalSustain'][1])

                fdSUST["PICKPASS"] = _accepted
                fdSUST["RESIDUAL"] = _residual
                FINALRES[str(_yy)]['SUSTAIN_test'] = (
                    FINALRES[str(_yy)]['SUSTAIN_test'].append(fdSUST, ignore_index=True)
                )

                # --- LowFreqTest
                fdLOW = {}
                fdLOW["EQID"] = eqid
                fdLOW["STAT"] = _ss

                if wd['evaluatePick_tests']['LowFreqTrend'][0]:
                    fdLOW["TESTPASS"] = 1
                    FINALRES[str(_yy)]['TOT_LOWFREQ_test_pass'] += 1
                else:
                    fdLOW["TESTPASS"] = 0
                    FINALRES[str(_yy)]['TOT_LOWFREQ_test_fail'] += 1

                if (wd['evaluatePick_tests']['LowFreqTrend'][1][0] >=
                   wd['evaluatePick_tests']['LowFreqTrend'][1][1]):
                    fdLOW["VALUE"] = +wd['evaluatePick_tests']['LowFreqTrend'][1][0]
                else:
                    fdLOW["VALUE"] = -wd['evaluatePick_tests']['LowFreqTrend'][1][1]

                fdLOW["PICKPASS"] = _accepted
                fdLOW["RESIDUAL"] = _residual
                FINALRES[str(_yy)]['LOWFREQ_test'] = (
                    FINALRES[str(_yy)]['LOWFREQ_test'].append(
                                    fdLOW, ignore_index=True)
                )

            # ======== Export Csv
            logger.debug("Exporting CSVs: round %d" % _yy)
            FINALRES[str(_yy)]['AMP_test'] = FINALRES[str(_yy)]['AMP_test'].apply(pd.to_numeric, errors='ignore')
            FINALRES[str(_yy)]['AMP_test'].to_csv(
                                   os.sep.join([storedir, "FINAL_Amp_test_" + str(_yy) + ".csv"]),
                                   sep=',',
                                   index=False,
                                   float_format="%.6f",
                                   na_rep="NA")

            FINALRES[str(_yy)]['SUSTAIN_test'] = FINALRES[str(_yy)]['SUSTAIN_test'].apply(pd.to_numeric, errors='ignore')
            FINALRES[str(_yy)]['SUSTAIN_test'].to_csv(
                                   os.sep.join([storedir, "FINAL_Sustain_test_" + str(_yy) + ".csv"]),
                                   sep=',',
                                   index=False,
                                   float_format="%.6f",
                                   na_rep="NA")

            FINALRES[str(_yy)]['LOWFREQ_test'] = FINALRES[str(_yy)]['LOWFREQ_test'].apply(pd.to_numeric, errors='ignore')
            FINALRES[str(_yy)]['LOWFREQ_test'].to_csv(
                                   os.sep.join([storedir, "FINAL_LowFreq_test_" + str(_yy) + ".csv"]),
                                   sep=',',
                                   index=False,
                                   float_format="%.6f",
                                   na_rep="NA")

            FINALRES[str(_yy)]['BAIT_BKmissed']['EventStation'].to_csv(
                       os.sep.join([storedir, "BK_failures_" + str(_yy) + ".csv"]),
                       sep=',',
                       index=False,
                       float_format="%.6f",
                       na_rep="NA")

    logger.info("Storing Pickles")
    QU.savePickleObj(FINALRES, os.sep.join([storedir, "BaitTestResults.pkl"]))

    # ======= Adding resume lines
    logger.info("Creating BaIt Resume")
    with open(os.sep.join([storedir, "BaIt_Resume.txt"]), "w") as OUT:
        OUT.write("@@@ Event ID" + os.linesep)
        OUT.write(("%s" + os.linesep) % eqid)

        OUT.write("@@@ Total STATIONS: Input" + os.linesep)
        OUT.write(("%d" + os.linesep) % (_xx+1))  # enumerate start from 0

        OUT.write("@@@ Total STATIONS: TrimFails" + os.linesep)
        OUT.write(("%d" + os.linesep) % FINALRES['BAIT_trimFAILURE']['count'])

        OUT.write("@@@ Total STATIONS: 1st BKFail" + os.linesep)
        OUT.write(("%d" + os.linesep) % FINALRES['1']['BAIT_BKmissed']['count'])

        OUT.write("@@@ Total 1st AMP test fail" + os.linesep)
        OUT.write(("%d" + os.linesep) % FINALRES['1']['TOT_AMP_test_fail'])
        OUT.write("@@@ Total 1st AMP test pass" + os.linesep)
        OUT.write(("%d" + os.linesep) % FINALRES['1']['TOT_AMP_test_pass'])

        OUT.write("@@@ Total 1st SUSTAIN test fail" + os.linesep)
        OUT.write(("%d" + os.linesep) % FINALRES['1']['TOT_SUSTAIN_test_fail'])
        OUT.write("@@@ Total 1st SUSTAIN test pass" + os.linesep)
        OUT.write(("%d" + os.linesep) % FINALRES['1']['TOT_SUSTAIN_test_pass'])

        OUT.write("@@@ Total 1st LOWFREQ test fail" + os.linesep)
        OUT.write(("%d" + os.linesep) % FINALRES['1']['TOT_LOWFREQ_test_fail'])
        OUT.write("@@@ Total 1st LOWFREQ test pass" + os.linesep)
        OUT.write(("%d" + os.linesep) % FINALRES['1']['TOT_LOWFREQ_test_pass'])

        OUT.write("@@@ Total STATIONS: Passed to MultiPick1" + os.linesep)
        OUT.write(("%d" + os.linesep) % accepted_count)
        OUT.write("@@@ Total STATIONS: Rejected for MultiPick1 (all cases)" + os.linesep)
        OUT.write(("%d" + os.linesep) % rejected_count)


def compare_times_bait_predicted(time_bait, station, allpicks_obj):
    """ Return residual among FIRST PREDICTED ARRIVAL and BAIT-TIME """
    predarr = QU.get_pick_slice(allpicks_obj,
                                station,
                                "^Predicted_",
                                phase_pick_indexnum=0,
                                arrival_order='all')
    predpick = predarr[0][1]
    #
    return time_bait - predpick


def final_automatic_statistics_events(outfilename="Events_Automatic_STATS.txt",
                                      additional_statistics=[]):
    """ Statistics at event level --> end of all """
    with open(outfilename, 'w') as OUT:
        if additional_statistics:
            for _as in additional_statistics:
                OUT.write("@@@ " + _as[0] + os.linesep)
                OUT.write(("%d" + os.linesep) % int(_as[1]))
                for _sl in _as[2]:
                    OUT.write(("%s" + os.linesep) % _sl)

# === TIPS
# Dict comprehension

# { (some_key if condition else default_key):(something_if_true if condition
#           else something_if_false) for key, value in dict_.items() }
