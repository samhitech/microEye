import logging

try:
    from ids_peak import ids_peak
except ImportError as e:
    logging.getLogger(__name__).warning(f'Could not import ids_peak module: {e}')
