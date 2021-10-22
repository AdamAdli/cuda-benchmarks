import os

_SCRIPT_LOCATION = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT_PATH = os.path.normpath(_SCRIPT_LOCATION + '/../')

_DLMC_KNOWN_LOCATIONS = [
    f'{REPO_ROOT_PATH}/benchmark/dlmc',
    os.path.normpath(f'{REPO_ROOT_PATH}/../dlmc')
]


def find_dlmc():
    """
    Find the sputnik dataset dlmc

    :throws: exception if fails to find dlmc
    """
    for known_location in _DLMC_KNOWN_LOCATIONS:
        if os.path.exists(f'{known_location}/dlmc.csv'):
            return known_location

    raise Exception("Failed to find dlmb.csv in a known location")
