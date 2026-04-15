import pandas as pd

from microEye.analysis.tools.photon_cal.models import CalibrationDatasetMeta
from microEye.qt import QtCore


def results_to_dataframe(
    datasets: dict[str, CalibrationDatasetMeta], returned_data: dict
) -> pd.DataFrame:
    records = []
    for name, _dataset in datasets.items():
        params = returned_data[name]
        record = {
            'Dataset': name,
            'Inverse Gain (DN/e-)': 1 / params['gain_e_per_dn'],
            'Gain (e-/DN)': params['gain_e_per_dn'],
            'Read Noise (DN)': params['read_noise_dn'],
            'Read Noise (e-)': params['read_noise_e'],
            'Responsivity (ADU/photon)': params['responsivity'],
            'Responsivity Intercept (ADU)': params['responsivity_intercept'],
            'QE (e/photon)': params['qe'],
            'QE Intercept (e)': params['qe_intercept'],
            'QE (%)': params['responsivity'] * params['gain_e_per_dn'] * 100,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            value = self._df.iat[index.row(), index.column()]
            return str(value)

        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return self._df.columns[section]
            return str(section)
        return None
