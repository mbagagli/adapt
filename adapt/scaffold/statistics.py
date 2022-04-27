"""
STATISTIC module:

contains all the latest functions for final ADAPT pick extraction

TIPS:

We will use dataframe count() function to count the number of Non Null values in the dataframe.

access a row: df['key']   or  df[index(int)]
access line:  df.loc['key'] or df.loc['']

lo() and iloc() are equal, bu the latter can use boolean query

https://pandas-docs.github.io/pandas-docs-travis/user_guide/indexing.html


"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import adapt.database as QD
import adapt.errors as QE
#
from adapt.utils import loadPickleObj, savePickleObj, calcEpiDist
from adapt.parser import Event2dict
#
logger = logging.getLogger(__name__)


class AdaptDatabase(object):
    """Base class for PANDAS - CSV statistic database
    """

    def __init__(self, tag="ADAPT Stats Container", columns=[]):
        self.tag = tag
        self.df = self._setup_dataframe(columns)
        #
        self.inv = None

    def _import_adapt_inventory(self, inventory):
        """
        If user needs to use the inventory, this method will keep it
        as object attribute.
        """
        if isinstance(inventory, str):
            self.inv = loadPickleObj(inventory)
        elif isinstance(inventory, QD.StatContainer):
            # at this stage we load a StatContainer object.
            self.inv = inventory
        else:
            raise ValueError("The inventory parameter must be either a path" +
                             "or a adapt.database.StationContainer")

    def _current_shape(self):
        row, col = self.df.shape
        return row, col

    def _setup_dataframe(self, columns):
        if not columns:
            return pd.DataFrame()
        else:
            if isinstance(columns, (list, tuple)):
                pd.DataFrame(columns=columns)
            else:
                raise ValueError("Columns attribute must be an iterable!")

    def _check_column_field(self, colname, defaultfill=np.nan):
        """ This function will see if a column exist or create one """
        if not isinstance(colname, (str, list, tuple)):
            raise ValueError("Colname must be either a string or an iterable!")
        #
        if isinstance(colname, (list, tuple)):
            for cc in colname:
                if cc not in self.df.columns:
                    self.df[cc] = defaultfill
        else:
            if colname not in self.df.columns:
                self.df[colname] = defaultfill

    def _append_record(self, record, check_cols=True):
        """ Append the given record, creating columns at needance.
            Record can be either a dictionary or a pd.Series
        """
        idx, _ = self._current_shape()
        if check_cols:
            self._check_column_field(tuple(record.keys()))
        #
        # self.df = self.df.append(pd.Series(record), ignore_index=True)
        self.df.loc[idx] = pd.Series(record)

    def _extract_event_record(self, eventobj):
        """ Add the record line """
        return Event2dict(eventobj)

    def _extract_phase_record(self, phasedict, idx=0, default=np.nan):
        """ Extract a dictionary with useful information about the
            PHASES inveolved (PickDict[STATION]) !!!

            The point of this s
        """
        tmpDict = {}
        for ph, phd in phasedict.items():
            tmpDict['phasename'] = ph
            tmpDict['adapt_year'] = phd[idx]['timeUTC_pick'].year if phd[idx]['timeUTC_pick'] else default
            tmpDict['adapt_month'] = phd[idx]['timeUTC_pick'].month  if phd[idx]['timeUTC_pick'] else default
            tmpDict['adapt_day'] = phd[idx]['timeUTC_pick'].day  if phd[idx]['timeUTC_pick'] else default
            tmpDict['adapt_hour'] = phd[idx]['timeUTC_pick'].hour  if phd[idx]['timeUTC_pick'] else default
            tmpDict['adapt_minute'] = phd[idx]['timeUTC_pick'].minute  if phd[idx]['timeUTC_pick'] else default
            tmpDict['adapt_second'] = (phd[idx]['timeUTC_pick'].second +
                                       phd[idx]['timeUTC_pick'].microsecond*(10**-6))  if phd[idx]['timeUTC_pick'] else default
            tmpDict['tlatetearly'] = (phd[idx]['timeUTC_late'] -
                                      phd[idx]['timeUTC_early'])  if phd[idx]['timeUTC_pick'] else default

            # --- The TRIAGE DICT should always be there ...
            trdct = phd[idx]['weight'].get_triage_dict()
            trres = phd[idx]['weight'].get_triage_results()

            tmpDict['std'] = phd[idx]['weight'].get_uncertainty(method="std") if trres else default
            tmpDict['mad'] = phd[idx]['weight'].get_uncertainty(method="mad") if trres else default

            for _kk, vv in trdct.items():
                tmpDict['tot_obs'] = trdct['tot_obs'] if trdct['tot_obs'] else 0
                if _kk in ('valid_obs',
                           'spare_obs',
                           'stable_pickers',
                           'outliers',
                           'pickers_involved',
                           'failed_pickers'):
                    tmpDict[_kk] = len(vv)
            # 20.12.2020
            lenOne, lenTwo, winNum = self._additional_validobs_analysis(
                                            trdct['valid_obs'])
            tmpDict['win_one'] = lenOne
            tmpDict['win_two'] = lenTwo
            tmpDict['n_win'] = winNum
            #
            try:
                # This can be either True or False
                tmpDict['evaluate_tests'] = (
                        phd[idx]['evaluate_obj'].get_verdict())
            except QE.MissingAttribute:
                # If here, the evaluation dict has been missing (not specified)
                # Inside adapt the pick is accepted anyway (if triage is ok)
                # Here we put NA based on the fact that it wasn't specified.
                tmpDict['evaluate_tests'] = default
        #
        return tmpDict

    def _additional_validobs_analysis(self, validobslist):
        """ Adding information in valid obs windowing """
        lenW1 = len([ii for ii in validobslist if ii[0][-1] == "1"])
        lenW2 = len([ii for ii in validobslist if ii[0][-1] == "2"])
        #
        if lenW1 > 0 and lenW2 > 0:
            numW = 2
        elif lenW1 > 0 or lenW2 > 0:
            numW = 1
        else:
            numW = 0
        #
        return lenW1, lenW2, numW

    def is_empty(self):
        """ Simply check that some entries are present. BOOLEAN return
        """
        row, col = self._current_shape()
        if row == 0 and col == 0:
            return True
        else:
            return False

    def insert_adapt_pickcontainer(self, event, pickdict):
        """ This method should only append the values
        collect, extract and append

        LA STAZIONE E EPICENTRO LA FAI QUA!!!!!

        """
        logger.info("Working on: %r" % event)
        # --------- Event
        workdict, otdict = {}, {}
        evdict = self._extract_event_record(event)
        ot = evdict.pop('origintime')
        otdict['year'] = ot.year
        otdict['month'] = ot.month
        otdict['day'] = ot.day
        otdict['hour'] = ot.hour
        otdict['minute'] = ot.minute
        otdict['second'] = ot.second + ot.microsecond*(10**-6)
        workdict.update(evdict)  # without origintime
        workdict.update(otdict)  # adding otdict

        # --------- Phase
        for ss, sd in pickdict.items():
            # Metti stazione+ epicentro
            workdict['station'] = ss
            if self.inv:
                try:
                    workdict['alias'] = self.inv[ss]['alias']
                except KeyError:
                    workdict['alias'] = np.nan
                #
                workdict['epidist'] = calcEpiDist(evdict['lat'],
                                                  evdict['lon'],
                                                  self.inv[ss]['lat'],
                                                  self.inv[ss]['lon'],
                                                  outdist='km')
            else:
                workdict['alias'] = np.nan
                workdict['epidist'] = np.nan
            #
            phasedict = self._extract_phase_record(sd)

            workdict.update(phasedict)
            #
            self._append_record(workdict, check_cols=True)

    def expand_columns(self, record, keymerge):
        """ Append the given recor in new columns.
            It needs one and only KEY (colname) of the
            previous database, in order to keep things neat.
            Record can be either a dictionary or a pd.DataFrame
        """
        if isinstance(record, (dict, pd.DataFrame)):
            newcols = pd.DataFrame(record)
            self.df = self.df.merge(newcols, how='left', on=keymerge)
        else:
            raise QE.InvalidType("record must be either a dict or pandas "
                                 "dataframe object!")

    def export_dataframe(self,
                         outfile="adapt_statistics.csv",
                         outformat="CSV",
                         floatformat=None,
                         column_caps=False):
        """ Export dataframe object in class. Specify the format """
        # df_pick = df_pick.apply(pd.to_numeric, errors='ignore')
        if column_caps:
            self.df.columns.str.upper()

        # Write out
        if outformat.lower() == 'csv':
            if isinstance(floatformat, str):
                self.df.to_csv(outfile,
                               sep=',',
                               index=False,
                               float_format=floatformat,
                               na_rep="NA", encoding='utf-8')
            else:
                self.df.to_csv(outfile,
                               sep=',',
                               index=False,
                               na_rep="NA", encoding='utf-8')
        else:
            raise QE.InvalidParameter("Sorry..only CNV export allowed")

        # Restore
        if column_caps:
            self.df.columns.str.lower()

    def store_dataframe(self, outfile="adapt_statistics.pkl"):
        """ Store this class into a pickled object for later use.
        Import the class with the given classmethod """
        savePickleObj(self.df, outfile)

    def clear_dataframe(self):
        """ Simply erase the object DataFrame """
        self.df = pd.DataFrame()

    def sort_dataframe(self, fields, inplace=True):
        """ Sort the database based on numeric column fields """
        if not isinstance(fields, (list, tuple)):
            raise QE.InvalidType("PGive either a list or a tuple for column fields!")
        #
        if inplace:
            self.df.sort_values(by=fields, inplace=True, ignore_index=True)
        else:
            return self.df.sort_values(by=fields, inplace=False, ignore_index=True)

    @classmethod
    def import_db(self, data):
        """ Import the picks from a pandas dataframe """
        if isinstance(data, pd.DataFrame):
            self.df = data
        else:
            self.df = pd.read_csv(data)

# Intanto appendi
# poi se vuoi aggiungere roba, controlli EVENT e ID e appendi.
# Prima di aggiungere una colonna/attributo controlla sempre che esista
# con il _check_column_field
