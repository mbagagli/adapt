from obspy import read, UTCDateTime
import adapt.utils as QU
from adapt.picks.evaluation import Gandalf
import adapt.processing as QPR


def test_evaluation():
    """
    Test function for the creation of an
    obspy Inventory class
    """
    errors = []
    # --- Work
    conf = QU.get_adapt_config("./tests_data/test_config_layer_v041.yml")
    st_raw = read()
    st_proc = QPR.processStream(st_raw,
                                copystream=True,
                                **conf)
    eval_obj = Gandalf(st_proc,
                       st_raw,
                       UTCDateTime("2009-08-24T00:20:07.7"),
                       channel="*Z",
                       functions_dict={'max_signal2noise_ratio': {
                                        'threshold': 2.0},
                                       'mean_signal2noise_ratio': {
                                        'threshold': 2.0},
                                       'low_freq_trend': {
                                        'sign_window': 0.4,
                                        'confidence': 0.9, 'use_raw': True}
                                       })
    #
    eval_obj.work()
    od = eval_obj.get_outdict()
    res = eval_obj.get_verdict()
    import pprint ; pprint.pprint(od) ;
    #
    if not res:
        errors.append("Pick Evaluation results is False while True")
    if od['max_signal2noise_ratio']['result'] == 38.449994967764013:
        errors.append("max_signal2noise_ratio test differ")
    if od['mean_signal2noise_ratio']['result'] == 30.966984943217639:
        errors.append("mean_signal2noise_ratio test differ")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
