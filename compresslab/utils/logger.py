from typing import Dict
import csv
import torch
import time
import contextlib
import pickle
import os

class MetricLogger:
    """
    A simple logger to log metrics and save them to a CSV file.
    It also provides a context manager to time code blocks.

    Usage:

    1.initialize: 
    .. code-block:: python
        logger = MetricLogger(save_dir)
            
    2.recode time:
    .. code-block:: python
        with logger.timer(name, metric_name):
            # Your code here
            
    3.log metrics:
    .. code-block:: python
        logger.log(name, {"metric_name": value})
            
    4.save metrics as a csv file:
    .. code-block:: python
        logger.save()  
    """
    def __init__(self, save_dir: str, filename: str = "metrics"):
        """
        Args:
            save_dir (str): Directory to save the CSV file.
            filename (str): Name of the CSV file.
        """
        self.metrics = {}
        self.save_dir = save_dir
        self.filename = filename


    def log(self, name, log_dict: Dict[str, float]):
        if name not in self.metrics:
            self.metrics[name] = {}
        for k, v in log_dict.items():
            # assert isinstance(v, (int, float)), f"Value {v} for key {k} in {name} is not a number."
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.metrics[name]:
                self.metrics[name][k] = []
            self.metrics[name][k].append(v)
    
    def save(self):
        """
        Save the metrics to a CSV file and a pkl file(used in benchmark test).
        """
        # Compute the average of all lists in the metrics
        for name, metrics in self.metrics.items():
            for key, values in metrics.items():
                self.metrics[name][key] = sum(values) / len(values)

        os.makedirs(self.save_dir, exist_ok=True)
        with open(f"{self.save_dir}/{self.filename}.csv", "w") as f:
            # Write the header
            headers = ["name"] + list(next(iter(self.metrics.values())).keys())
            writer = csv.writer(f)
            writer.writerow(headers)
            
            # Write the data
            for name, metrics in self.metrics.items():
                row = [name]
                for key in headers[1:]:
                    row.append(f"{metrics[key]:.6f}" if key in metrics else "")
                writer.writerow(row)
                
        # Save the metrics as a pickle file
        with open(f"{self.save_dir}/{self.filename}.pkl", "wb") as pkl_file:
            pickle.dump(self.metrics, pkl_file)


    @contextlib.contextmanager
    def timer(self, name, metric_name: str, avg: int = 1, cuda_sync=False, unit="ms"):
        """
        A context manager to time a code block.
        """
        # FIXME: the test time maybe not accurate
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
