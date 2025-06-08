# modify from https://github.com/binzzheng/DVC-PyTorch
import torch
import torch.nn as nn
import math
from compresslab.nn.video_compression.dvc.subnet import *
from compresslab.nn.video_compression.abc import (
    PFrameCodec, 
    PFrameLikelihoods, 
    PFrameCompressInput,
    PFrameCompressOutput,
    PFrameDecompressOutput,
    PFrameForwardOutput,
    PFrameForwardInput,
    PFrameCodecCompressInput,
    PFrameCodecCompressOutput,
    PFrameCodecDecompressOutput,
    IFrameCompressOutput
)

from compresslab.core.models import CompressionModel
from compresslab.core.entropy_models import EntropyBottleneck, GaussianConditional
# import torchac
import numpy as np

class DVC(CompressionModel, PFrameCodec):
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

    def forward(self, input: PFrameForwardInput):
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
    
    def compress(self, input: PFrameCodecCompressInput) -> PFrameCodecCompressOutput:
        output_list = []
        for i in range(len(input.P_frames)):
            output = self.compress_P_frame(
                PFrameCompressInput(
                    input_frame=input.P_frames[i],
                    refer_frame=input.I_frame if i == 0 else output.recon_frame
                )
            )
            output_list.append(output)

        return PFrameCodecCompressOutput(
            compress_input=input,
            P_frame_compress_output=output_list
        )
    

    def decompress(self, input: PFrameCodecCompressOutput) -> PFrameCodecDecompressOutput:
        recon_frames = []

        for i in range(len(input.P_frame_compress_output)):
            output = self.decompress_P_frame(
                input.P_frame_compress_output[i]
            )
            recon_frames.append(output.recon_frame)

        return PFrameCodecDecompressOutput(recon_frames=recon_frames)




# class DVC_Official(nn.Module, PFrameCodec):
#     def __init__(self,
#                  out_channel_N=64, 
#                  out_channel_M=96, 
#                  out_channel_mv=128, 
#                  optical_level=4,):
#         super().__init__()
        
#         self.out_channel_N = out_channel_N
#         self.out_channel_M = out_channel_M
#         self.out_channel_mv = out_channel_mv
        
#         self.opticFlow = ME_Spynet(optical_level)
#         self.mvEncoder = Analysis_mv_net(out_channel_mv)
#         self.mvDecoder = Synthesis_mv_net(out_channel_mv)
#         self.warpnet = Warp_net()
#         self.resEncoder = Analysis_net(out_channel_N, out_channel_M)
#         self.resDecoder = Synthesis_net()
#         self.respriorEncoder = Analysis_prior_net()
#         self.respriorDecoder = Synthesis_prior_net()
#         self.bitEstimator_z = BitEstimator(out_channel_N)
#         self.bitEstimator_mv = BitEstimator(out_channel_mv)
#         self.warp_weight = 0
#         self.mxrange = 150
#         self.calrealbits = False

#     def motioncompensation(self, ref, mv):
#         warpframe = flow_warp(ref, mv)
#         inputfeature = torch.cat((warpframe, ref), 1)
#         prediction = self.warpnet(inputfeature) + warpframe
#         return prediction, warpframe

#     def forward(self, input_image, referframe):
#         estmv = self.opticFlow(input_image, referframe)
#         mvfeature = self.mvEncoder(estmv)
#         if self.training:
#             quant_mv = mvfeature + torch.empty_like(mvfeature).uniform_(-0.5, 0.5)
#         else:
#             quant_mv = torch.round(mvfeature)
#         quant_mv_upsample = self.mvDecoder(quant_mv)

#         prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

#         input_residual = input_image - prediction

#         feature = self.resEncoder(input_residual)
#         batch_size = feature.size()[0]

#         z = self.respriorEncoder(feature)

#         if self.training:
#             compressed_z = z + torch.empty_like(z).uniform_(-0.5, 0.5)
#         else:
#             compressed_z = torch.round(z)

#         recon_sigma = self.respriorDecoder(compressed_z)

#         feature_renorm = feature

#         if self.training:
#             compressed_feature_renorm = feature_renorm + torch.empty_like(feature_renorm).uniform_(-0.5, 0.5)
#         else:
#             compressed_feature_renorm = torch.round(feature_renorm)

#         recon_res = self.resDecoder(compressed_feature_renorm)
#         recon_image = prediction + recon_res

#         clipped_recon_image = recon_image.clamp(0., 1.)


# # distortion
#         mse_loss = torch.mean((recon_image - input_image).pow(2))

#         # psnr = tf.cond(
#         #     tf.equal(mse_loss, 0), lambda: tf.constant(100, dtype=tf.float32),
#         #     lambda: 10 * (tf.log(1 * 1 / mse_loss) / np.log(10)))

#         warploss = torch.mean((warpframe - input_image).pow(2))
#         interloss = torch.mean((prediction - input_image).pow(2))
        

# # bit per pixel

#         def feature_probs_based_sigma(feature, sigma):
            
#             def getrealbitsg(x, gaussian):
#                 # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
#                 cdfs = []
#                 x = x + self.mxrange
#                 n,c,h,w = x.shape
#                 for i in range(-self.mxrange, self.mxrange):
#                     cdfs.append(gaussian.cdf(i - 0.5).view(n,c,h,w,1))
#                 cdfs = torch.cat(cdfs, 4).cpu().detach()
                
#                 byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

#                 real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

#                 sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

#                 return sym_out - self.mxrange, real_bits


#             mu = torch.zeros_like(sigma)
#             sigma = sigma.clamp(1e-5, 1e10)
#             gaussian = torch.distributions.laplace.Laplace(mu, sigma)
#             probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
#             total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            
#             if self.calrealbits and not self.training:
#                 decodedx, real_bits = getrealbitsg(feature, gaussian)
#                 total_bits = real_bits

#             return total_bits, probs

#         def iclr18_estrate_bits_z(z):
            
#             def getrealbits(x):
#                 cdfs = []
#                 x = x + self.mxrange
#                 n,c,h,w = x.shape
#                 for i in range(-self.mxrange, self.mxrange):
#                     cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
#                 cdfs = torch.cat(cdfs, 4).cpu().detach()
#                 byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

#                 real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

#                 sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

#                 return sym_out - self.mxrange, real_bits

#             prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
#             total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


#             if self.calrealbits and not self.training:
#                 decodedx, real_bits = getrealbits(z)
#                 total_bits = real_bits

#             return total_bits, prob


#         def iclr18_estrate_bits_mv(mv):

#             def getrealbits(x):
#                 cdfs = []
#                 x = x + self.mxrange
#                 n,c,h,w = x.shape
#                 for i in range(-self.mxrange, self.mxrange):
#                     cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
#                 cdfs = torch.cat(cdfs, 4).cpu().detach()
#                 byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

#                 real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

#                 sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
#                 return sym_out - self.mxrange, real_bits

#             prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
#             total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


#             if self.calrealbits and not self.training:
#                 decodedx, real_bits = getrealbits(mv)
#                 total_bits = real_bits

#             return total_bits, prob

#         total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
#         # entropy_context = entropy_context_from_sigma(compressed_feature_renorm, recon_sigma)
#         total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
#         total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

#         im_shape = input_image.size()

#         bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
#         bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
#         bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
#         bpp = bpp_feature + bpp_z + bpp_mv
        
#         # return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp
#         return {
#             "recon_frame": clipped_recon_image,
#             "mse_loss": mse_loss,
#             "warploss": warploss,
#             "interloss": interloss,
#             "prediction": prediction,
#             "bpp": {
#                 "bpp_mv": bpp_mv,
#                 "bpp_feature": bpp_feature,
#                 "bpp_z": bpp_z,
#                 "bpp": bpp
#             },
#         }