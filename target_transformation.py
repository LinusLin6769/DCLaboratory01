from typing import Sequence
import numpy as np

class TargetTransformation:
    def __init__(
                self,
                type: str) -> None:
        self.type = str(type)
        
        self.transformer = {
            'none': self._raw,
            'log-return': self._log_return,
#             'simple-return': self._simple_return
        }

        self.back_transformer = {
            'none': self._back_raw,
            'log-return': self._back_log_return,
#             'simple-return': self._back_simple_return
        }
    
    def transform(self, series: Sequence) -> np.array:
        self.y0 = series[0]
        return self.transformer[self.type](series)

    def back_transform(self, series) -> np.array:
        return self.back_transformer[self.type](series)

    def _raw(self, series) -> np.array:
        return np.array(series)
    
    def _back_raw(self, series) -> np.array:
        return np.array(series)

    def _log_return(self, series) -> np.array:
        transformed = np.diff(np.log(series)).tolist()
        return np.array([0] + transformed)

    def _back_log_return(self, series) -> np.array:
        return np.exp(np.cumsum(series)) * self.y0

    def _simple_return(self, series) -> np.array:
        res = []
        for i, s in enumerate(series):
            if i == 0: continue

            if i == 1: res.append(0)

            res.append((s - series[i-1])/series[i-1])
        
        return np.array(res)

    # @NOTE: Not completed yet. Don't know how to deal with the first value (being 0?)
    def _back_simple_return(self, series) -> np.array:
        return np.exp(np.cumsum(series)) * self.y0
