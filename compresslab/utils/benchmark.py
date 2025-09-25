import numpy as np
from pathlib import Path
import os
import pickle
import csv
from typing import List, Union, Optional
from dataclasses import dataclass
from compresslab.utils.config import BenchmarkItem
import logging
import matplotlib.pyplot as plt

class Benchmark:
    
    PRESET_ITEMS = ["BD_RATE", "BD_CURVE"]
    
    def __init__(self, 
                 exp_dir: Path,
                 eval_items: Optional[Union[BenchmarkItem, List[BenchmarkItem]]],
                 ):
        """
        Args:
            exp_dir (Path): Directory to save the benchmark files.
            eval_items (BenchmarkItem): Benchmark test items defined in the yaml config file.
        """
        self.exp_dir = exp_dir

        self.exp_metrics = {}
        # load all metrics from each model directory
        # the metrics is organzied as {Key or Name in the config file: {codec_name: {metric_name: metric_value}}}
        for root, _, files in os.walk(exp_dir):
            for file in files:
                if file == "metrics.pkl":
                    with open(os.path.join(root, file), 'rb') as f:
                        data = pickle.load(f)
                        self.exp_metrics[root.split('/')[-1]] = data

        for item in eval_items:
            if item.Key.upper() not in self.PRESET_ITEMS:
                logging.error(f"Unknown benchmark test item: {item.Key}. Skipping...")
                continue
            getattr(self, item.Key.lower())(**item.Params)

    def bd_curve(self, *args, **kwargs):
        plt.figure()
        for model_name, metrics in self.exp_metrics.items():
            bpp = []
            psnr = []
            for metric in metrics.values(): # multi codec
                if "bpp" in metric and "psnr" in metric:
                    bpp.append(metric["bpp"])
                    psnr.append(metric["psnr"])
            plt.plot(bpp, psnr, label=model_name)
        plt.xlabel("bits per pixel (Bpp)")
        plt.ylabel("PSNR")
        plt.title("BD-Rate Curve")
        plt.legend()
        plt.savefig(os.path.join(self.exp_dir, "bd_curve.pdf"))


    def bd_rate(self, *args, 
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
            print(model_name, metrics)
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
