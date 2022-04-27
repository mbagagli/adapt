"""ADAPT errors module

This module contains custom error message for ADAPT picking framework.
The initial hint has been given from the `reference`_

.. _reference:
    https://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
"""


class Error(Exception):
    """ Base class for other exceptions """
    pass


class BadConfigurationFile(Error):
    """ Raised when important instance checks are not respected """
    pass


class BadInstance(Error):
    """ Raised when important instance checks are not respected """
    pass


class MissingAttribute(Error):
    """ Raised when a requested class attribute is missing
    """
    pass


class MissingVariable(Error):
    """ Raised when a parameter/value is missing
    """
    pass


class MissingPicks(Error):
    """ Raised when a specific pick is missing
    """
    pass


class InvalidVariable(Error):
    """ Raised when a variable is not used correctly
    """
    pass


class InvalidParameter(Error):
    """ Raised when a function or method parameter isn't set correctly
    """
    pass


class NoisyStation(Error):
    """ Raised when the initial STA/LTA doesn't recognize any seismic
        signal
    """
    pass


class VersionError(Error):
    """ Raised when there's a mismatch between the versions in the I/O
        formats and usage.
    """
    pass


class InvalidType(Error):
    """ Raised when a variable is not used correctly """
    pass


class CheckError(Error):
    """ Raised when the developer is too lazy to think at smth else """
    pass


class NoBaitValidPick(Error):
    """ Raised when no valid picks are found by the BaIt layer """
    pass
