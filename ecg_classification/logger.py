from typing import Any, Dict, Union, Iterable

import torch
import os
from datetime import datetime


class Logger(object):
    """
    Class to log different metrics.
    """

    def __init__(self,
                 experiment_path: str = os.path.join(os.getcwd(), "my_experiments",
                                                     datetime.now().strftime("%d_%m_%Y__%H_%M_%S")),
                 experiment_path_extension: str = "",
                 path_metrics: str = "metrics",
                 path_models: str = "models") -> None:
        """
        Constructor method
        :param experiment_path: (str) Path to experiment folder
        :param path_metrics: (str) Path to folder in which all metrics are stored
        :param experiment_path_extension: (str) Extension to experiment folder
        :param path_models: (str)  Path to folder in which all models are stored
        """
        experiment_path = experiment_path + experiment_path_extension
        # Save parameters
        self.path_metrics = os.path.join(experiment_path, path_metrics)
        self.path_models = os.path.join(experiment_path, path_models)
        # Init folders
        os.makedirs(self.path_metrics, exist_ok=True)
        os.makedirs(self.path_models, exist_ok=True)
        # Init dicts to store the metrics and hyperparameters
        self.metrics = dict()
        self.temp_metrics = dict()

    def log_metric(self, metric_name: str, value: Any) -> None:
        """
        Method writes a given metric value into a dict including list for every metric.
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(float(value))
        else:
            self.metrics[metric_name] = [float(value)]

    def log_temp_metric(self, metric_name: str, value: Any) -> None:
        """
        Method writes a given metric value into a dict including temporal metrics.
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.temp_metrics:
            self.temp_metrics[metric_name].append(float(value))
        else:
            self.temp_metrics[metric_name] = [float(value)]

    def save_temp_metric(self, metric_name: Union[Iterable[str], str]) -> Dict[str, float]:
        """
        Method writes temporal metrics into the metrics dict by averaging.
        :param metric_name: (Union[Iterable[str], str]) One temporal metric name ore a list of names
        """
        averaged_temp_dict = dict()
        # Case if only one metric is given
        if isinstance(metric_name, str):
            # Calc average
            value = float(torch.tensor(self.temp_metrics[metric_name]).mean())
            # Save metric in log dict
            self.log_metric(metric_name=metric_name, value=value)
            # Put metric also in dict to be returned
            averaged_temp_dict[metric_name] = value
        # Case if multiple metrics are given
        else:
            for name in metric_name:
                # Calc average
                value = float(torch.tensor(self.temp_metrics[name]).mean())
                # Save metric in log dict
                self.log_metric(metric_name=name, value=value)
                # Put metric also in dict to be returned
                averaged_temp_dict[name] = value
        # Reset temp metrics
        self.temp_metrics = dict()
        # Save logs
        self.save()
        return averaged_temp_dict

    def save_model(self, model_sate_dict: Dict, name: str) -> None:
        """
        Saves a given state dict
        :param model_sate_dict: (Dict) State dict to be saved
        :param name: (str) Name of the file
        """
        torch.save(obj=model_sate_dict, f=os.path.join(self.path_models, name + ".pt"))

    def save(self) -> None:
        """
        Method saves all current logs (metrics and hyperparameters). Plots are saved directly.
        """
        # Iterate items in metrics dict
        for metric_name, values in self.metrics.items():
            # Convert list of values to torch tensor to use build in save method from torch
            values = torch.tensor(values)
            # Save values
            torch.save(values, os.path.join(self.path_metrics, '{}.pt'.format(metric_name)))
