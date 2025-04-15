import numpy as np
from pathlib import Path
import os
import pickle
import csv
from typing import List, Union
from dataclasses import dataclass

@dataclass
class BenchmarkTestItem:
    BD_RATE = {"bd-rate"}

class Benchmark:
    """
    Benchmark:
        1.calculate BD-Rate for lossy compression
    """

    bd_rate_baseline = dict(
        JPEG=dict(
            bpp=[0.22115325927734372,
                0.32661437988281244,
                0.42312622070312506,
                0.5083855523003472,
                0.5878660413953993,
                0.6601265801323783,
                0.728946261935764,
                0.786026848687066,
                0.8497187296549479,
                0.9060007731119791,
                0.9643800523546006,
                1.0372119479709203,
                1.1273964775933163,
                1.23984612358941,
                1.3688269721137154,
                1.5718070136176217,
                1.8588163587782118,
                2.350199381510416,
                3.4013400607638893],
            psnr=[23.779894921045457,
                26.57723034342358,
                28.042246379237767,
                29.04180914810682,
                29.78473021842612,
                30.378313496658652,
                30.903164761925012,
                31.307827225129476,
                31.70484530103775,
                32.05865273799422,
                32.39929573599792,
                32.789915599755076,
                33.234742421489976,
                33.792594320368266,
                34.39429509119509,
                35.239001425077355,
                36.32886455167303,
                37.9121247026983,
                40.556657112988766],
        )
    )


    def __init__(self, exp_dir: Path, 
                 test_item: Union[BenchmarkTestItem, List[BenchmarkTestItem]] = BenchmarkTestItem.BD_RATE,
                 baseline=None):
        self.exp_dir = exp_dir

        self.exp_metrics = {}

        for root, _, files in os.walk(exp_dir):
            for file in files:
                if file.endswith(".pkl"):
                    with open(os.path.join(root, file), 'rb') as f:
                        data = pickle.load(f)
                        self.exp_metrics[root.split('/')[-1]] = data
        
        self.baseline = baseline if baseline else "JPEG"

        if test_item == BenchmarkTestItem.BD_RATE:
            self.calc_bd_rate()

        # TODO: support more benchmark test items
        for item in test_item:
            pass

    # TODO: support more benchmark test items
    def calc_bd_rate(self):
        """
        Calculate BD-Rate for lossy compression
        Args:
            exp_dir (str): Directory to save the CSV file.
        """
        bd_rates = {}
        for model_name, metrics in self.exp_metrics.items():
            bpp = []
            psnr = []
            for metric in metrics.values(): # multi codec
                if "bpp" in metric and "psnr" in metric:
                    bpp.append(metric["bpp"])
                    psnr.append(metric["psnr"])
            
            # calculate bd-rate
            bd_rate = self._bj_delta(self.bd_rate_baseline[self.baseline]["bpp"], self.bd_rate_baseline[self.baseline]["psnr"], bpp, psnr)
            bd_rates[model_name] = bd_rate

        with open(os.path.join(self.exp_dir, "metrics.csv"), "w") as f:
            # Write the header
            headers = ["name", "bd-rate"]
            writer = csv.writer(f)
            writer.writerow(headers)
            
            # Write the data
            for name, bd_rate in bd_rates.items():
                row = [name, bd_rate]
                writer.writerow(row)



    # https://github.com/Anserw/Bjontegaard_metric
    def _bj_delta(self, R1, PSNR1, R2, PSNR2, mode=1):
        lR1 = np.log(R1)
        lR2 = np.log(R2)

        # find integral
        if mode == 0:
            # least squares polynomial fit
            p1 = np.polyfit(lR1, PSNR1, 3)
            p2 = np.polyfit(lR2, PSNR2, 3)

            # integration interval
            min_int = max(min(lR1), min(lR2))
            max_int = min(max(lR1), max(lR2))

            # indefinite integral of both polynomial curves
            p_int1 = np.polyint(p1)
            p_int2 = np.polyint(p2)

            # evaluates both poly curves at the limits of the integration interval
            # to find the area
            int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
            int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

            # find avg diff between the areas to obtain the final measure
            avg_diff = (int2-int1)/(max_int-min_int)
        else:
            # rate method: sames as previous one but with inverse order
            p1 = np.polyfit(PSNR1, lR1, 3)
            p2 = np.polyfit(PSNR2, lR2, 3)

            # integration interval
            min_int = max(min(PSNR1), min(PSNR2))
            max_int = min(max(PSNR1), max(PSNR2))

            # indefinite interval of both polynomial curves
            p_int1 = np.polyint(p1)
            p_int2 = np.polyint(p2)

            # evaluates both poly curves at the limits of the integration interval
            # to find the area
            int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
            int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

            # find avg diff between the areas to obtain the final measure
            avg_exp_diff = (int2-int1)/(max_int-min_int)
            avg_diff = (np.exp(avg_exp_diff)-1)*100
        return avg_diff
