import pprint
from obspy import UTCDateTime
import adapt.utils as QU
from adapt.picks.weight import Weighter, Bootstrap
import numpy as np
import numpy.testing as npt


# ==================================  STANDARDS

nda = (5.008, 4.5, 7.99, 2.5, 4.9, 5.0, 5.3, 10.1)

samples_comp_42 = np.array([
    [5.3, 2.5, 4.9  ,   5.3  ,   7.99 ,  10.1  ,   4.9  , 4.9  ],
    [  5.3  ,   4.5  ,   7.99 ,   5.3  ,   7.99 ,   7.99 ,  10.1  ,4.9  ],
    [  2.5  ,  10.1  ,  10.1  ,   7.99 ,   5.   ,   4.9  ,   4.5  ,        10.1  ],
    [  2.5  ,   5.   ,   5.   ,   4.5  ,  10.1  ,   2.5  ,   4.9  ,       5.008],
    [  2.5  ,   4.5  ,   5.   ,   4.9  ,   2.5  ,   5.008,   5.008,         7.99 ],
    [  7.99 ,   5.3  ,   4.5  ,  10.1  ,   2.5  ,   2.5  ,  10.1  ,      5.3  ],
    [  5.   ,   5.   ,   5.3  ,   5.   ,   7.99 ,   2.5  ,   5.3  ,      2.5  ],
    [ 10.1  ,   5.008,   7.99 ,   4.9  ,   7.99 ,   5.3  ,   4.9  ,      5.008],
    [  5.3  ,   4.5  ,   2.5  ,   5.008,   2.5  ,   5.   ,   4.5  ,      4.5  ],
    [  5.008,   4.5  ,   4.9  ,   4.5  ,   2.5  ,   2.5  ,   5.3  ,      2.5  ]
    ])


replicates_comp_42 = np.array(
 [[5.73625,   5.1,     2.15271885,   0.2,      1.654375,    2.5,   10.1],
  [6.75875,   6.645,   1.88827592,   1.345,    1.75875,     4.5,   10.1],
  [6.89875,   6.495,   2.84249203,   2.8,      2.67375,     2.5,   10.1],
  [4.9385,    4.95,    2.2005626,    0.254,    1.3385,      2.5,   10.1],
  [4.67575,   4.95,    1.6147416,    0.254,    1.1318125,   2.5,   7.99],
  [6.03625,   5.3,     2.8569999,    2.745,    2.5203125,   2.5,   10.1],
  [4.82375,   5.0,     1.63445662,   0.3,      1.161875,    2.5,   7.99],
  [6.3995,    5.154,   1.88187161,   0.254,    1.720375,    4.9,   10.1],
  [4.226,     4.5,     1.03364984,   0.504,    0.863,       2.5,   5.3],
  [3.9635,    4.5,     1.15949375,   0.654,    1.097625,    2.5,   5.3]]
 )


# ==================================  HELPERS

def _compare_arrays(a1, a2, atol=1e-08, rtol=1e-05):
    if not np.isclose(a1, a2, atol=atol, rtol=rtol).all():
        return False
    else:
        return True

# ==================================


def _compare_floats_dictionaries(testdict, refdict, decimal=10):
    """ Use numpy assert_almost_equal for float comparison inside dicts
    """
    asserr = []
    for _k, _v in testdict.items():
        if isinstance(_v, (float, int)):
            try:
                npt.assert_almost_equal(
                  _v, refdict[_k], decimal=decimal)
            except AssertionError:
                asserr.append((_k, _v))
    #
    return asserr


def test_bootstrap_init():
    """
    Create the class and check parameters.
    """
    errors = []

    bs = Bootstrap(nda)

    if not isinstance(bs.get_input_data(), np.ndarray):
        errors.append("Returned type wrong: %s" % type(bs.get_input_data()))

    assert not errors


def test_bootstrap_samples():
    """
    Create the class and check parameters.
    """
    errors = []

    bs = Bootstrap(nda, random_seed=42)

    _ = bs._resample_with_replacement(8, 10)

    if not _compare_arrays(bs.get_samples(), samples_comp_42, atol=1e-12):
        errors.append("Samples are not the same!")

    assert not errors


def test_bootstrap_work():
    """
    Create the class and check parameters.
    """
    errors = []

    bs = Bootstrap(nda, random_seed=42)

    _ = bs.work(n_resamples=10)

    if not _compare_arrays(bs.get_samples(), samples_comp_42, atol=1e-12):
        errors.append("Samples are not the same!")

    if not _compare_arrays(bs.get_replicates(), replicates_comp_42, atol=1e-12):
        errors.append("Samples are not the same!")

    assert not errors


def test_bootstrap_replicaStats():
    """
    Create the class and check parameters.
    """
    errors = []

    bs = Bootstrap(nda, random_seed=42)

    _ = bs.work(n_resamples=10)

    smpl = bs.get_samples()
    rplc = bs.get_replicates()

    if not (
      np.mean(smpl[0, :]) == rplc[0, 0] and
      np.median(smpl[0, :]) == rplc[0, 1] and
      np.std(smpl[0, :]) == rplc[0, 2] and
      np.mean(smpl[-1, :]) == rplc[-1, 0] and
      np.median(smpl[-1, :]) == rplc[-1, 1] and
      np.std(smpl[-1, :]) == rplc[-1, 2]
      ):
        errors.append("Replica's stas are wrong!")

    assert not errors


def test_bootstrap_replicaStats_extraction():
    """
    Create the class and check parameters.
    """
    errors = []

    bs = Bootstrap(nda, random_seed=42)

    _ = bs.work(n_resamples=10)

    rplc = bs.get_replicates()

    _tmp = bs.extract_statistics(replicastat="median", statproc="med")
    if _tmp != np.median(rplc[:, 1]):
        errors.append("Wrong extraction: ReplicaStat MEDIAN -"
                      "StatProc MEDIAN")

    _tmp = bs.extract_statistics(replicastat="mean", statproc="med")
    if _tmp != np.median(rplc[:, 0]):
        errors.append("Wrong extraction: ReplicaStat MEAN -"
                      "StatProc MEDIAN")

    _tmp = bs.extract_statistics(replicastat="mean", statproc="mn")
    if _tmp != np.mean(rplc[:, 0]):
        errors.append("Wrong extraction: ReplicaStat MEAN -"
                      "StatProc MEAN")

    _tmp = bs.extract_statistics(replicastat="med", statproc="mn")
    if _tmp != np.mean(rplc[:, 1]):
        errors.append("Wrong extraction: ReplicaStat MEDIAN -"
                      "StatProc MEAN")

    assert not errors
