""" This module wants to create all necessary funcitons/class to
    manipulate the inputlist catalog

    NB: Every QuakeCleaner class must work on separate catalog!
"""
import logging
from pathlib import Path
#
from pandas import DataFrame, to_numeric
from obspy import Catalog
import adapt.parser as QP
import adapt.errors as QE
import adapt.utils as QU
import adapt.catalog.cleaner_functions as TESTFUNCTIONS


ADAPTMAGRADIUS = (
    ["-inf", 3.0, 250],               # (low_MAG,hig_MAG,radius[km])
    [3.0, 4.0, 350],
    [4.0, "inf", 500]
    )

logger = logging.getLogger(__name__)


# ================================================


class QuakeCleaner(object):
    """ Main class to manipulate the filters method in the science """
    def __init__(self, event_list_path, inventory_path, querydict=None):
        """ This class initilize a catalog to be queryed, starting from
            a ADAPT events list and a ADAPT StationDictionary

            event_list_path: ADAPT event list "EQID MANUPDE MANUPICK"
            inventory_path: ADAPT StatDictionary
            querydict: list of function to evaluate
        """
        self.catalogpath = Path(event_list_path)
        self.inventorypath = Path(inventory_path)
        # Added after _setup methods
        self.catalog = None
        self.inventory = None
        self.catalogpickdict = {}
        self.querypardict = querydict
        self.queryresdict = {}
        self._build_inventory()
        self._build_catalog()

    # ------------------------------------------------- Private methods

    def _build_inventory(self):
        if not self.inventorypath.exists():
            raise QE.InvalidVariable("Inventory path is wrong! Update it!")
        #
        logger.info("Building inventory ...")
        self.inventory = QU.loadPickleObj(self.inventorypath)

    def _build_catalog(self):
        if not self.catalogpath.exists():
            raise QE.InvalidVariable("Catalog path is wrong! Update it!")
        #
        logger.info("Building catalog ...")
        self.catalog = Catalog()
        with open(self.catalogpath, 'r') as IN:
            for _xx, _line in enumerate(IN):
                _line = _line.strip()
                fields = _line.split()
                kpid = fields[0]
                manupde = Path(fields[1])
                manupick = Path(fields[2])
                ev = QP.manupde2Event(
                                manupde, kpid, "CatalogFilter")
                evpd, _ = QP.manupick2PickContainer(manupick, kpid, "CLEANER")
                # Store
                self.catalog.append(ev)
                self.catalogpickdict[kpid] = evpd

        logger.info("Successfuly loaded %d EVENTS" % (_xx+1))

    def _define_quake_radius(self, magnitude, radiusdict=ADAPTMAGRADIUS):
        """ Extract the distance in km of the ADAPT radius """
        for ra in radiusdict:
            if float(ra[0]) < magnitude <= float(ra[1]):
                radius = ra[2]
                break
        return radius

    def _extract_catalog_event(self, eqid=None):
        """ Simply query EQID along the catalog attribute to search.

        this method is a wrapper around the adapt.utils function named
        extract_catalog_event

        """
        return QU.extract_catalog_event(self.catalog, eqid)

    def _calculate_epicentral_distance(self, ela, elo, sla, slo, outdist="km"):
        """ return the epicentral distance among event and station """
        return QU.calcEpiDist(ela, elo, sla, slo, outdist=outdist)

    # -------------------------------------------------- Public methods

    def query_events(self):
        """ Run the given set of functions over the given catalog """
        if not self.catalog:
            raise QE.MissingAttribute("Missing class catalog! abort")
        if not self.querypardict:
            raise QE.MissingAttribute("Missing filtering parameter. Set it!")
        #
        sortedkeys = sorted(self.querypardict, key=str.lower)

        for _ev in self.catalog:
            logger.info("@@@ Working with EVENT: %s" % _ev.resource_id.id)
            print("Working with EVENT: %s" % _ev.resource_id.id)
            self.queryresdict[_ev.resource_id.id] = {}
            for xx in sortedkeys:
                logger.debug("%s: %s - %r" % (_ev.resource_id.id,
                                              xx, self.querypardict[xx]))
                _funct = getattr(TESTFUNCTIONS, xx)
                #try:
                # Call the test
                od = _funct(self._extract_catalog_event(
                                                    eqid=_ev.resource_id.id),
                            self.catalogpickdict[_ev.resource_id.id],
                            self.inventory,
                            **self.querypardict[xx])
                # Store results
                self.queryresdict[_ev.resource_id.id][xx] = od

                # except:
                #     # Something went wrong
                #     self.queryresdict[xx] = {}

                # Print results
                logger.debug("%s: %s - %s" % (
                                _ev.resource_id.id, xx,
                                str(self.queryresdict[_ev.resource_id.id][xx]))
                             )

    def set_inventory_path(self, newpath):
        if not isinstance(newpath, str):
            raise QE.InvalidType("New path must be a string!")
        #
        self.inventorypath = Path(newpath)
        self._build_inventory()

    def set_catalog_path(self, newpath):
        if not isinstance(newpath, str):
            raise QE.InvalidType("New path must be a string!")
        #
        self.catalogpath = Path(newpath)
        self._build_catalog()

    def set_query_functions(self, newdict):
        if not isinstance(newdict, dict):
            raise QE.InvalidType("The given input must be a dict!")
        #
        self.querypardict = newdict

    def get_passed_eqid(self):
        """ Automatically filter internally for the FAILED events """
        pass

    def get_failed_eqid(self):
        """ Automatically filter internally for the FAILED events """
        pass

    def get_results_dict(self):
        """ Return the results dictionary of the events query """
        if not self.queryresdict:
            raise QE.MissingAttribute("Missing results. Run query first!")
        #
        return self.queryresdict

    def store_results_csv(self, outpath):
        """ Create and store a CSV with the final results
        """

        # ---- Checks
        if not isinstance(outpath, str):
            raise QE.InvalidType("OUTPATH must be a string!")
        if not self.queryresdict:
            raise QE.MissingAttribute("Missing results. Run query first!")

        outpath = Path(outpath)
        if outpath.exists():
            logger.warning("OUTPATH already exists! Change destination ...")
            return False

        # ---- Works
        columns = ["EQID", "LON", "LAT", "DEP", "OT", "MAG"]
        # I just need one EVENT RES DICT, as the same function (and
        # relative results-keys) are the same
        first_evid = next(iter(self.queryresdict))
        for _ffname in self.querypardict.keys():
            columns = (columns +
                       list(self.queryresdict[first_evid][_ffname].keys()))
        df = DataFrame(columns=[cc.upper() for cc in columns])

        for _ev in self.catalog:
            fillingdict = {}
            # Header
            fillingdict["EQID"] = _ev.resource_id.id
            fillingdict["LON"] = _ev.origins[0].longitude
            fillingdict["LAT"] = _ev.origins[0].latitude
            fillingdict["DEP"] = _ev.origins[0].depth / 1000.0  # km
            fillingdict["OT"] = _ev.origins[0].time.isoformat()
            fillingdict["MAG"] = _ev.magnitudes[0].mag

            # Results
            for _funtest, _funres in self.queryresdict[_ev.resource_id.id].items():
                for _resname, _resval in _funres.items():
                    fillingdict[_resname.upper()] = _resval

            # End event
            df = df.append(fillingdict, ignore_index=True)

        # End CSV
        df = df.apply(to_numeric, errors='ignore')
        df.to_csv(outpath,
                  sep=',',
                  index=False,
                  float_format="%.4f",
                  na_rep="NA")
        #
        return True
