from typing import Dict
import csv
import torch
import time
import contextlib
import pickle
import os
import numpy as np
import logging

class MetricsLogger:
    """
    A simple logger to log metrics and save them to a CSV file.
    It also provides a context manager to time code blocks.

    Usage:

    1.initialize: 
    .. code-block:: python
        logger = MetricsLogger(save_dir)
            
    2.log the execution time:
    .. code-block:: python
        with logger.timer(name, metric_name):
            # Your code here
            
    3.log metrics:
    .. code-block:: python
        logger.log(name, {"metric_name": value})
            
    4.save metrics as a csv file:
    .. code-block:: python
        logger.save()
        
        
    NOTE: If model name contains '/', it will be treated as 'dataset/model_name'. 
    The result in the CSV file will be sorted by dataset name, bpp, and finally model name.
    """
    def __init__(self, save_dir: str = None, filename: str = "metrics"):
        """
        Args:
            save_dir (str): Directory to save the CSV file.
            filename (str): Name of the CSV file.
        """
        self.metrics = dict()
        self.save_dir = save_dir
        self.filename = filename


    def log(self, name: str, log_dict: Dict[str, float]):
        if self.save_dir is None:
            raise ValueError("Please set the save_dir before saving the metrics.")
        if name not in self.metrics:
            self.metrics[name] = dict()
        for k, v in log_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (np.float32, np.float64, np.float16)):
                v = float(v)
            
            if k not in self.metrics[name]:
                self.metrics[name][k] = list()
            self.metrics[name][k].append(v)
            
    def log_without_avg(self, name: str, log_dict: Dict[str, float]):
        if self.save_dir is None:
            raise ValueError("Please set the save_dir before saving the metrics.")
        if name not in self.metrics:
            self.metrics[name] = dict()
        for k, v in log_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (np.float32, np.float64, np.float16)):
                v = float(v)
            
            self.metrics[name][k] = v
    
    def save(self):
        """
        Save the metrics to a CSV file and a pkl file(used in benchmark test).
        """
        if self.save_dir is None:
            raise ValueError("Please set the save_dir before saving the metrics.")
        # Compute the average of all lists in the metrics
        for name, metrics in self.metrics.items():
            for key, values in metrics.items():
                if isinstance(values, list) and len(values) > 0:
                    if key.endswith("(ms)") or key.endswith("(s)"):
                        # Remove outliers for timing metrics using IQR method
                        if len(values) > 3:  # Only remove outliers if we have enough data points
                            q1 = np.percentile(values, 25)
                            q3 = np.percentile(values, 75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            filtered_values = [v for v in values if lower_bound <= v <= upper_bound]
                            if filtered_values:  # If we still have values after filtering
                                values = filtered_values
                                
                    self.metrics[name][key] = sum(values) / len(values)

        os.makedirs(self.save_dir, exist_ok=True)
        if not self.metrics == {}:
            csv_path = f"{self.save_dir}/{self.filename}.csv"
            existing_data = {}
            headers = ["name"] + list(next(iter(self.metrics.values())).keys())
            
            # Read existing CSV file if it exists
            if os.path.exists(csv_path):
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data[row["name"]] = row
                    # Update headers to include any new columns
                    existing_headers = reader.fieldnames or []
                    headers = list(dict.fromkeys(existing_headers + headers))  # Preserve order, remove duplicates
            
            with open(csv_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                
                # Update existing data with new metrics
                for name, metrics in self.metrics.items():
                    row_data = {"name": name}
                    for key in headers[1:]:
                        if key in metrics:
                            row_data[key] = f"{metrics[key]:.6f}"
                        elif name in existing_data and key in existing_data[name]:
                            row_data[key] = existing_data[name][key]
                        else:
                            row_data[key] = ""
                    existing_data[name] = row_data
                
                # Write all data (existing + new/updated)
                # Sort the data by name with custom logic(dataset name, bpp, model name)
                sorted_data = sorted(existing_data.values(), key=lambda row: (
                    row["name"].split("/")[0] if "/" in row["name"] else "",
                    float(row["bpp"]) if row.get("bpp") and row["bpp"] != "" else "",
                    row["name"].split("/")[1] if "/" in row["name"] else row["name"]
                ))
                
                for row_data in sorted_data:
                    writer.writerow(row_data)
                    
            # Save the metrics as a pickle file
            with open(f"{self.save_dir}/{self.filename}.pkl", "wb") as pkl_file:
                # pickle.dump(self.metrics, pkl_file)
                pickle.dump(existing_data, pkl_file)
        else:
            logging.warning("No metrics to save. The metrics dictionary is empty.")


    @contextlib.contextmanager
    def timer(self, name: str, metric_name: str, cuda_sync=False, unit="ms", avg: int = 1):
        """
        A context manager to time a code block.
        """
        if self.save_dir is None:
            raise ValueError("Please set the save_dir before saving the metrics.")
        if cuda_sync:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
            yield

            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        else:
            start_time = time.time()
            
            yield

            elapsed_time = (time.time() - start_time) * 1000

        if unit == "ms": 
            self.log(name, {metric_name + f"({unit})": elapsed_time / avg})
        elif unit == "s":
            self.log(name, {metric_name + f"({unit})": elapsed_time / avg / 1000})
        else:
            raise ValueError(f"Unsupported unit: {unit}. Supported units are 'ms' and 's'.")
