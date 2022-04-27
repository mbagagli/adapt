from adapt.database import PickContainer
import adapt.scaffold.savior as QS
import adapt.utils as QU
import pprint


# ------------- TESTS


def test_pickcontainer():
    """ Test to evaluate the performance of SavePickContainerClass"""
    #
    _tmp_pick = QU.loadPickleObj('./tests_data/test_analysis_pd.pkl')
    pd = PickContainer.from_dictionary(_tmp_pick)
    #
    mys = QS.PickContainer2ObsPyPick(pd)
    mys.store()
    pprint.pprint(mys.get_output())

