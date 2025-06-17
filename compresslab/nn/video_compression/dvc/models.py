# modify from https://github.com/binzzheng/DVC-PyTorch
import torch
import torch.nn as nn
import math
from compresslab.nn.video_compression.dvc.subnet import *
from compresslab.nn.video_compression.abc import *
from compresslab.core.entropy_models import EntropyBottleneck, GaussianConditional

class DVC(VideoCodec):
    def __init__(self, 
                 out_channel_N=64, 
                 out_channel_M=96, 
                 out_channel_mv=128, 
                 optical_level=4,
                 **kwargs):
        super().__init__(**kwargs)
        self.opticFlow = ME_Spynet(optical_level)
        self.mvEncoder = Analysis_mv_net(out_channel_mv)
        self.mvDecoder = Synthesis_mv_net(out_channel_mv)
        self.mvpriorEncoder = Analysis_mvprior_net(out_channel_N, out_channel_M, out_channel_mv)
        self.mvpriorDecoder = Synthesis_mvprior_net(out_channel_N, out_channel_mv)
        self.warpnet = Warp_net()

        self.resEncoder = Analysis_net(out_channel_N, out_channel_M)
        self.resDecoder = Synthesis_net(out_channel_N, out_channel_M)
        self.respriorEncoder = Analysis_prior_net(out_channel_N, out_channel_M)
        self.respriorDecoder = Synthesis_prior_net(out_channel_N, out_channel_M)
        
        self.entropy_hyper_mv = EntropyBottleneck(out_channel_N)
        self.entropy_hyper_res = EntropyBottleneck(out_channel_N)
        self.entropy_bottleneck_mv = GaussianConditional(None)
        self.entropy_bottleneck_res = GaussianConditional(None)

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward_P_frame(self, input: PFrameForwardInput):
        input_image, referframe = input.input_frame, input.refer_frame
        estmv = self.opticFlow(input_image, referframe)
        mv_fea = self.mvEncoder(estmv)

        mv_prior = self.mvpriorEncoder(mv_fea)
        quant_mvprior, mvprior_likelihoods = self.entropy_hyper_mv(mv_prior)
        recon_mv_sigma = self.mvpriorDecoder(quant_mvprior)

        quant_mv, mv_likelihoods = self.entropy_bottleneck_mv(mv_fea, recon_mv_sigma)
        recon_mv = self.mvDecoder(quant_mv)

        prediction, warp_frame = self.motioncompensation(referframe, recon_mv)
        
        res = input_image - prediction
        res_fea = self.resEncoder(res)
        res_prior = self.respriorEncoder(res_fea)
        quant_resprior, resprior_likelihoods = self.entropy_hyper_res(res_prior)
        recon_res_sigma = self.respriorDecoder(quant_resprior)

        quant_res, res_likelihoods = self.entropy_bottleneck_res(res_fea, recon_res_sigma)

        recon_res = self.resDecoder(quant_res)
        recon_image = prediction + recon_res

        clipped_recon_image = recon_image.clamp(0., 1.)

        return PFrameForwardOutput(
            input_frame=input_image,
            recon_frame=clipped_recon_image,
            warp_frame=warp_frame,
            prediction=prediction,
            likelihoods=PFrameLikelihoods(
                y_mv=mv_likelihoods,
                z_mv=mvprior_likelihoods,
                y=res_likelihoods,
                z=resprior_likelihoods
            )
        )

    def update(self, scale_table=None, force=False):

        SCALES_MIN = 0.11
        SCALES_MAX = 256
        SCALES_LEVELS = 64

        def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
            return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.entropy_bottleneck_mv.update_scale_table(scale_table, force=force)
        updated = self.entropy_bottleneck_res.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    
    def compress_P_frame(self, input: PFrameCompressInput):
        input_image, referframe = input.input_frame, input.refer_frame     
        estmv = self.opticFlow(input_image, referframe)
        mv_fea = self.mvEncoder(estmv)

        mv_prior = self.mvpriorEncoder(mv_fea)
        mvprior_strings = self.entropy_hyper_mv.compress(mv_prior)
        quant_mvprior = self.entropy_hyper_mv.decompress(mvprior_strings, mv_prior.size()[-2:])
        recon_mv_sigma = self.mvpriorDecoder(quant_mvprior)
        mv_indexes = self.entropy_bottleneck_mv.build_indexes(recon_mv_sigma)
        mv_strings = self.entropy_bottleneck_mv.compress(mv_fea, mv_indexes)
        quant_mv = self.entropy_bottleneck_mv.decompress(mv_strings, mv_indexes)
        recon_mv = self.mvDecoder(quant_mv)

        prediction, warpframe = self.motioncompensation(referframe, recon_mv)

        res = input_image - prediction
        res_fea = self.resEncoder(res)

        res_prior = self.respriorEncoder(res_fea)
        resprior_strings = self.entropy_hyper_res.compress(res_prior)
        quant_resprior = self.entropy_hyper_res.decompress(resprior_strings, res_prior.size()[-2:])
        
        recon_res_sigma = self.respriorDecoder(quant_resprior)
        res_indexes = self.entropy_bottleneck_res.build_indexes(recon_res_sigma)
        res_strings = self.entropy_bottleneck_res.compress(res_fea, res_indexes)
        quant_res = self.entropy_bottleneck_res.decompress(res_strings, res_indexes)

        recon_res = self.resDecoder(quant_res)
        recon_frame = prediction + recon_res

        clipped_recon_frame = recon_frame.clamp(0., 1.)

        return PFrameCompressOutput(
            input_frame=input_image,
            recon_frame=clipped_recon_frame,
            refer_frame=referframe,
            y_mv_strings=mv_strings,
            z_mv_strings=mvprior_strings,
            y_strings=res_strings,
            z_strings=resprior_strings,
            mv_shape=mv_prior.size()[-2:],
            main_shape=res_prior.size()[-2:]
        )

    def decompress_P_frame(self, input: PFrameCompressOutput):
        mvprior_hat = self.entropy_hyper_mv.decompress(input.z_mv_strings, input.mv_shape)
        recon_mv_sigma = self.mvpriorDecoder(mvprior_hat)
        mv_indexes = self.entropy_bottleneck_mv.build_indexes(recon_mv_sigma)
        mv_hat = self.entropy_bottleneck_mv.decompress(input.y_mv_strings, mv_indexes)
        recon_mv = self.mvDecoder(mv_hat)
        prediction, _ = self.motioncompensation(input.refer_frame, recon_mv)
        
        resprior_hat = self.entropy_hyper_res.decompress(input.z_strings, input.main_shape)
        recon_res_sigma = self.respriorDecoder(resprior_hat)
        res_indexes = self.entropy_bottleneck_res.build_indexes(recon_res_sigma)
        res_hat = self.entropy_bottleneck_res.decompress(input.y_strings, res_indexes)
        recon_res = self.resDecoder(res_hat)
        
        recon_frame = prediction + recon_res
        recon_frame = recon_frame.clamp(0., 1.)
        return PFrameDecompressOutput(
            recon_frame=recon_frame,
        )
