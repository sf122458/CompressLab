from typing import List, Dict
import csv
import torch
import time
import contextlib
import pickle
import torch.distributed as dist

class MetricLogger():
    """
    A simple logger to log metrics and save them to a CSV file.
    It also provides a context manager to time code blocks.
    Usage:\n
        1.initialize: 
            ```
            logger = MetricLogger(save_dir)
            ```
        2.recode time:
            ```with logger.timer(name, metric_name):
                    # Your code here
            ```
        3.log metrics:
            ```
            logger.log(name, {"metric_name": value})
            ```
        4.save metrics as a csv file:
            ```
            logger.save()
            ```
    """
    def __init__(self, save_dir):
        """
        Args:
            save_dir (str): Directory to save the CSV file.
        """
        self.metrics = {}
        self.save_dir = save_dir
        

    def log(self, name, log_dict: Dict[str, float]):
        if name not in self.metrics:
            self.metrics[name] = {}
        for k, v in log_dict.items():
            assert isinstance(v, (int, float)), f"Value {v} for key {k} in {name} is not a number."
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


        with open(f"{self.save_dir}/metrics.csv", "w") as f:
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
        with open(f"{self.save_dir}/metrics.pkl", "wb") as pkl_file:
            pickle.dump(self.metrics, pkl_file)


    @contextlib.contextmanager
    def timer(self, name, metric_name: str, cuda_sync=False):
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
            
        self.log(name, {metric_name: elapsed_time})
