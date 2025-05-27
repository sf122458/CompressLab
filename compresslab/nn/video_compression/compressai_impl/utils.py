import math
import torch
import torch.nn as nn
from collections import defaultdict

class RateDistortionLoss(nn.Module):
    def __init__(self, bitdepth=8, return_details=False):
        super().__init__()
        self.bitdepth = bitdepth
        self.return_details = return_details
        self.mse = nn.MSELoss(reduction="none")
        self._scaling_functions = lambda x : x * (2 ** bitdepth - 1) ** 2

    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for frame_likelihoods in likelihoods_list
            for likelihoods in frame_likelihoods
        )
    
    def _get_scaled_distortion(self, x, target):
        if not len(x) == len(target):
            raise RuntimeError(f"len(x)={len(x)} != len(target)={len(target)}")
        
        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError(
                "Number of channels mismatches while computing distortion"
            )
        
        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)

        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)

        # compute metric over each component (y, u, v)
        metric_values = []
        for x0, x1 in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)

        # sum value over the components dimension
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)

        return scaled_metric, metric_value
    
    @staticmethod
    def _check_tensor(x) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) \
            or (isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor))
    
    @classmethod
    def _check_tensors_list(cls, lst):
        if (
            not isinstance(lst, (list, tuple))
            or len(lst) < 1
            or any(not cls._check_tensor(x) for x in lst)
        ):
            raise ValueError(
                "Expected a list of 4D torch.Tensor (or tuples of) as input"
            )
        
    def forward(self, output, target, lmbda=None):
        assert isinstance(target, type(output["x_hat"]))
        assert len(output["x_hat"]) == len(target)

        self._check_tensors_list(target)
        self._check_tensors_list(output["x_hat"])

        _, _, H, W = target[0].size()
        num_frames = len(target)
        out = {}
        num_pixels = H * W * num_frames

        # Get scaled and raw loss distortions for each frame
        scaled_distortions = []
        distortions = []
        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)

            distortions.append(distortion)
            scaled_distortions.append(scaled_distortion)

            if self.return_details:
                out[f"frame{i}.mse_loss"] = distortion
        # aggregate (over batch and frame dimensions).
        out["mse_loss"] = torch.stack(distortions).mean()

        # average scaled_distortions accros the frames
        scaled_distortions = sum(scaled_distortions) / num_frames

        assert isinstance(output["likelihoods"], list)
        likelihoods_list = output.pop("likelihoods")

        # collect bpp info on noisy tensors (estimated differentiable entropy)
        bpp_loss, bpp_info_dict = self.collect_likelihoods_list(likelihoods_list, num_pixels)
        if self.return_details:
            out.update(bpp_info_dict)  # detailed bpp: per frame, per latent, etc...

        # now we either use a fixed lambda or try to balance between 2 lambdas
        # based on a target bpp.
        if lmbda is not None:
            # lambdas = torch.full_like(bpp_loss, lmbda)
            bpp_loss = bpp_loss.mean()
            # out["loss"] = (lambdas * scaled_distortions).mean() + bpp_loss
            out["loss"] = lmbda * out["mse_loss"] + bpp_loss
        else:
            bpp_loss = bpp_loss.mean()

        out["distortion"] = scaled_distortions.mean()
        out["bpp_loss"] = bpp_loss
        out["psnr"] = 10 * torch.log10(1 / out["mse_loss"]).item()
        return out

    def collect_likelihoods_list(self, likelihoods_list, num_pixels):
        bpp_info_dict = defaultdict(int)
        bpp_loss = 0

        for i, frame_likelihoods in enumerate(likelihoods_list):
            frame_bpp = 0
            for label, likelihoods in frame_likelihoods.items():
                label_bpp = 0
                for field, v in likelihoods.items():
                    bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)

                    bpp_loss += bpp
                    frame_bpp += bpp
                    label_bpp += bpp

                    bpp_info_dict[f"bpp_loss.{label}"] += bpp.sum()
                    bpp_info_dict[f"bpp_loss.{label}.{i}.{field}"] = bpp.sum()
                bpp_info_dict[f"bpp_loss.{label}.{i}"] = label_bpp.sum()
            bpp_info_dict[f"bpp_loss.{i}"] = frame_bpp.sum()
        return bpp_loss, bpp_info_dict