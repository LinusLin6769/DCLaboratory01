from typing import Dict, Callable, List
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import warnings

class DCTransformer:

    def __init__(self,
                 trade: bool = False,
                 calc_return: str = 'simple') -> None:
        self.STATUS_CODE = {
            3:   'no info',
            0:   'local extreme',
            1:   'bull trend',
            2:   'bull overshoot',
            10:  'bull directional change',
            -1:  'bear trend',
            -2:  'bear overshoot',
            -10: 'bear directional change'}

        # Trading mechanism not available for now.
        if trade: print('Trading mechansims has been activated.')

        # determine the return calculation method used
        self.return_method(calc_return)

        # set-up metadata storages
        self.meta_data()

    def return_method(self, calc_return) -> None:
        # in-built return calculation methods
        self.calc_return_methods: Dict[str, Callable] = {
            'simple': lambda x1, x2: (x2-x1)/x1,
            'log'   : lambda x1, x2: np.log(x2) - np.log(x1)}

        # set-up the return calculation method
        if calc_return in self.calc_return_methods:
            self.calc_return = self.calc_return_methods[calc_return]
        else:
            raise ValueError('Unrecognised return calculation method.')

    def meta_data(self) -> None:
        # The durations for events (directional changes) and overshoots
        # Durations not available at this moment.
        self.durations: Dict[str, List[datetime]] = {
            "DC": [],
            "OS": []}

        # The amount of movements of the recorded events and overshoots
        self.move: Dict[str, List[float]] = {
            "DC": [],
            "OS": []}

        # Status of data points identified by the transformation
        self.status: List[int] = []

    def find_threshold(self, data):
        pass

    def dc_dissect(self) -> None:
        mode = None  # bull or bear, the event currently experiencing

        # Here goes the algorithm
        for i, x in enumerate(self.data):
            # first data point
            if i == 0:
                self.status.append(0)
                # markers[0].append([i] + data_eurgbp[i])
                local_ext = x
                local_ext_ind = i
                continue

            # identify the first trend
            if mode is None:
                r = self.calc_return(local_ext, x)
                if r >= self.DELTA_UP:      # bullish trend confirmed
                    self.status.append(10)    # mark the end of the DC (bullish)
                    self.status[(local_ext_ind+1):i] = [1]*(i-local_ext_ind-1)
                    last_event = i       # update the time of last recorded event
                    mode = "bullish"
                    continue
                elif r <= self.DELTA_DOWN:  # bearish trend confirmed
                    self.status.append(-10)   # mark the end of the DC (bearish)
                    self.status[(local_ext_ind+1):i] = [-1]*(i-local_ext_ind-1)
                    last_event = i       # update the time of last recorded event
                    mode = "bearish"
                    continue
                self.status.append(3)                 # before a trend confirmation, we have nothing
            
            if mode == "bullish":
                # find the new local extreme (maximum)
                if x >= local_ext:
                    local_ext = x
                    local_ext_ind = i
                
                r = self.calc_return(local_ext, x)
                # see if a new directional change has occurred (bearish)
                if r <= self.DELTA_DOWN:
                    self.status.append(-10)    # mark the end of the DC (bearish)
                    mode = "bearish"      # after experiencing the bearish DC, confirm the mode is now bearish

                    # identify the directional changing event (trend)
                    self.status[local_ext_ind] = 0    # mark the newly identified local extreme
                    self.status[(local_ext_ind+1):i] = [-1]*(i-local_ext_ind-1)

                    # log the durations and movements of event and overshoot
                    # durations["DC"].append((arr_eurgbp[i, 0]-arr_eurgbp[local_ext_ind, 0])/timedelta(milliseconds=1))
                    self.move["DC"].append(np.abs(r))
                    # durations["OS"].append((arr_eurgbp[local_ext_ind, 0]-arr_eurgbp[last_event, 0])/timedelta(milliseconds=1))
                    self.move["OS"].append(np.abs(self.calc_return(self.data[last_event], self.data[local_ext_ind])))

                    last_event = i
                    continue
                self.status.append(2)          # extended movements after a DC are considered as overshoot

            elif mode == "bearish":
                # find the new local extreme (minimum)
                if x <= local_ext:
                    local_ext = x
                    local_ext_ind = i

                r = self.calc_return(local_ext, x)
                # see if a new directional change has occurred (bullish)
                if r >= self.DELTA_UP:
                    self.status.append(10)     # mark the end of the DC (bullish)
                    mode = "bullish"      # after experiencing the bullish DC, confirm the mode is now bullish

                    # identify the directional changing event (trend)
                    self.status[local_ext_ind] = 0
                    self.status[(local_ext_ind+1):i] = [1]*(i-local_ext_ind-1)

                    # log the durations and movements of event and overshoot
                    # durations["DC"].append((arr_eurgbp[i, 0]-arr_eurgbp[local_ext_ind, 0])/timedelta(milliseconds=1))
                    self.move["DC"].append(np.abs(r))
                    # durations["OS"].append((arr_eurgbp[local_ext_ind, 0]-arr_eurgbp[last_event, 0])/timedelta(milliseconds=1))
                    self.move["OS"].append(np.abs(self.calc_return(self.data[last_event], self.data[local_ext_ind])))
                    
                    last_event = i
                    continue
                
                self.status.append(-2)  # extended movements after a DC are considered as overshoot

    def marker(self) -> None:
        # I name the collection of marked events as markers
        #
        # get the coordinates of the markers
        self.markers = {k: [] for k in self.STATUS_CODE.keys()}
        self.markers[100] = []  # collection of local extrema and directional changes
        for i, s in enumerate(self.status):
            self.markers[s].append((i, self.data[i]))
            if s in (0, 10, -10):
                self.markers[100].append((i, self.data[i], self.status[i]))

        self.markers = {k: np.array(v).T for k, v in self.markers.items()}

    def interpolation0(self, kind, warn=False) -> None:
        marked_x = np.array(self.markers[0][0]).astype(int)
        y = np.array(self.markers[0][1]).astype(float)
        
        # with inappropriate threshold, the algo might not find any extreme or dc
        if len(y) >= 2:
            index_x = list(np.arange(len(self.data)))

            interpolate_max_range = marked_x[-1]
            index_x_max_range = index_x[-1]
            out_of_range_fill = np.arange(interpolate_max_range+1, index_x_max_range+1)

            # generate the interpolated values
            #
            # in-range interpolation
            inter_y = interp1d(marked_x, y, kind=kind)

            # out-range interpolation (use original data)
            temp = np.arange(interpolate_max_range+1)
            transformed_target = inter_y(temp)
            self.tdata0 = np.append(transformed_target, self.data[out_of_range_fill])
        else:
            self.tdata0 = np.array([y])
            if warn:
                warnings.warn("Cannot found any extreme. Check if threshold is appropriate.")
    
    def interpolation1(self, kind, warn=False) -> None:
        marked_x = np.array(self.markers[100][0]).astype(int)
        y = np.array(self.markers[100][1]).astype(float)

        # with inappropriate threshold, the algo might not find any extreme or dc
        if len(y) >= 2:
            index_x = list(np.arange(len(self.data)))

            interpolate_max_range = marked_x[-1]
            index_x_max_range = index_x[-1]
            out_of_range_fill = np.arange(interpolate_max_range+1, index_x_max_range+1)

            # generate the interpolated values
            #
            # in-range interpolation
            inter_y = interp1d(marked_x, y, kind=kind)

            # out-range interpolation (use original data)
            temp = np.arange(interpolate_max_range+1)
            transformed_target = inter_y(temp)
            self.tdata1 = np.append(transformed_target, self.data[out_of_range_fill])
        else:
            self.tdata1 = np.array([y])
            if warn:
                warnings.warn("Cannot found any directional change. Check if threshold is appropriate.")

    def make_plot(self, marks=False, w_data0=False) -> None:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize = (20, 10))
        plt.plot(list(np.arange(len(self.data))), self.data, 'C0', label='raw data movement')
        if w_data0 :plt.plot(self.tdata0, 'C1', label='local extrema')
        plt.plot(self.tdata1, 'C2', label='local extrema and directional changes')
        if marks:
            plt.plot(np.array(self.markers[0][0]).astype(int), np.array(self.markers[0][1]).astype(float), 'ko', label = 'local extreme')
            plt.plot(np.array(self.markers[10][0]).astype(int), np.array(self.markers[10][1]).astype(float), 'C3D', label = 'bullish directional change')
            plt.plot(np.array(self.markers[-10][0]).astype(int), np.array(self.markers[-10][1]).astype(float), 'C4D', label = 'bearish directional change')
        # plt.plot(np.array(markers['pos close'][0]).astype(int), np.array(markers['pos close'][1]).astype(float), 'gD', label = 'position closed')
        plt.legend()
        plt.title(f'Std. of log-return: {self.sigma}, Thresholds: delta up = {round(self.DELTA_UP*100, 3)}%, delta down = {round(self.DELTA_DOWN*100, 3)}%, number of extrema: {len(self.markers[0][0])}')
        plt.show()

    def transform(self, data, threshold=None, kind='linear') -> None:
        self.data = np.array(data)
        self.sigma = round(np.std(np.diff(np.log(self.data))), 6)  # standard deviation of log-return

        if self.data.ndim != 1: raise ValueError('Data should be of 1 dimension.')

        if threshold == None:
            # self.find_threshold(data) # estimate the appropriate threshold of delta up and delta down (not built yet)
            pass
        else:
            self.DELTA_UP = threshold[0]
            self.DELTA_DOWN = threshold[1]

        self.dc_dissect()
        self.marker()
        # self.interpolation0(kind=kind)    # with extrema only (tdata0)
        self.interpolation1(kind=kind)    # with extrema and directional changes (tdata1)

    def get_raw_data(self) -> np.array:
        """Return the untransformed raw data."""
        return self.data

    def get_tdata0(self) -> np.array:
        """Return the transformed series with local extrema only."""
        return self.tdata0
    
    def get_tdata1(self) -> np.array:
        """Return the transformed series with extrema and directional changes."""
        return self.tdata1

    def get_dc_info(self) -> Dict:
        """Return information obtained from the dissection"""
        return {
            'status': self.status,
            'markers': self.markers,
            'movements': self.move,
            'durations': self.durations}

if __name__ == '__main__':
    import data_prep
    df, frequency, forecast_horizon, missing_values, equal_len = data_prep.convert_tsf_to_dataframe('datasets/m1_monthly_dataset.tsf')
    arr = df.values
    arr = arr.T

    ts = {}
    count = 0
    n_5 = 0
    for s in arr[2]:
        if len(s) >= 100:
            count += 1
            if count <= 5*n_5: continue
            ts[f'T{count-n_5*5}'] = {}
            ts[f'T{count-n_5*5}']['raw'] = s
        if count == 5*(n_5+1): break
    
    t = DCTransformer()
    t.transform(ts['T2']['raw'], threshold=(0.04, -0.04), kind='linear')
    t.make_plot(marks=True, w_data0=False)


    """
    from pandas_datareader import data
    from pprint import pprint
    prices = data.DataReader("^GSPC", "yahoo", "2000-03-17", "2019-02-20")
    arr_prices = prices.values

    tSPX = DCTransformer()
    tSPX.transform(arr_prices[:, 5], threshold=(0.1, -0.1))  # use Adj Close as target column
    tSPX.make_plot()"""
