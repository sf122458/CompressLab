import numpy as np
from pathlib import Path
import os
import pickle
import csv
from typing import List, Union, Optional
from dataclasses import dataclass
from compresslab.utils.config import General
import logging
import matplotlib.pyplot as plt

@dataclass
class BenchmarkTestItem:
    BD_RATE = {"bd-rate"}

class Benchmark:
    """
    Benchmark:
        1. calculate BD-Rate for lossy compression
        2. plot BD-Rate curve
    """

    # from CompressAI
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


    def __init__(self, 
                 exp_dir: Path,
                 config: Optional[Union[General, List[General]]] = None,
                 ):
        """
        Args:
            exp_dir (Path): Directory to save the benchmark files.
            config (General): Benchmark test items defined in the yaml config file.
        """
        
        if config is None:
            return

        self.exp_dir = exp_dir
        if not isinstance(config, list):
            self.config = [config]
        else:
            self.config = config

        self.exp_metrics = {}
        # load all metrics from each model directory
        for root, _, files in os.walk(exp_dir):
            for file in files:
                if file.endswith(".pkl"):
                    with open(os.path.join(root, file), 'rb') as f:
                        data = pickle.load(f)
                        self.exp_metrics[root.split('/')[-1]] = data

        for item in config:
            if item.Key == "BD_RATE":
                self.calc_bd_rate(item.Params)
            elif item.Key == "BD_CURVE":
                self.plot_bd_curve(item.Params)
            else:
                logging.error(f"Unknown benchmark test item: {item.Key}")

    def plot_bd_curve(self, *args, **kwargs):
        plt.figure()
        for model_name, metrics in self.exp_metrics.items():
            bpp = []
            psnr = []
            for metric in metrics.values(): # multi codec
                if "bpp" in metric and "psnr" in metric:
                    bpp.append(metric["bpp"])
                    psnr.append(metric["psnr"])
            plt.plot(bpp, psnr, label=model_name)
        plt.xlabel("Bpp")
        plt.ylabel("PSNR")
        plt.title("BD-Rate Curve")
        plt.legend()
        plt.savefig(os.path.join(self.exp_dir, "bd_curve.png"))


    def calc_bd_rate(self, *args, 
                     baseline: str="JPEG", 
                     mode: int=1, **kwargs):
        """
        Calculate BD-Rate for lossy compression
        Args:
            baseline(str): Baseline model name used in RD-Rate calculation, default is JPEG
            mode(int): 0 compare PSNR at the same Bpp, 1 compare Bpp at the same PSNR
        """
        bd_rates = {}
        for model_name, metrics in self.exp_metrics.items():
            bpp = []
            psnr = []
            for metric in metrics.values(): # multi codec
                if "bpp" in metric and "psnr" in metric:
                    bpp.append(metric["bpp"])
                    psnr.append(metric["psnr"])
            
            if len(bpp) > 0 and len(psnr) > 0:
                # calculate bd-rate
                bd_rate = self._bj_delta(
                    self.bd_rate_baseline[baseline]["bpp"], 
                    self.bd_rate_baseline[baseline]["psnr"], 
                    bpp, psnr,
                    mode=mode
                )
                bd_rates[model_name] = bd_rate

        if len(bd_rates) > 0:
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
