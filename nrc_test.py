import drjit as dr
import mitsuba as mi
import numpy as np
import torch
import imageio
import gin
from tqdm import tqdm
from model import configs
# import mitsuba.ad.common.mis_weight 

mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
# from mitsuba.ad.common import mis_weight

mis_weight = mi.ad.common.mis_weight


args = configs.parse_args()

gin.parse_config_files_and_bindings(
    config_files=[args.config],
    bindings=None, #FLAGS.gin_bindings, # additional configs
    skip_unknown=False)

nrc_config = configs.NrcConfig()
render_config = configs.RenderConfig()

render_res = 256

# scene = mi.load_file('mitsuba-tutorials/scenes/cbox.xml')
scene = mi.load_file(render_config.scene_path)

# sensor = mi.load_dict({
#         'type': 'perspective', 
#         'fov': 45, 
#         'to_world': mi.ScalarTransform4f.translate([0, 0, 0]) \
#                                         .rotate([0, 0, 0], 0)   \
#                                         .look_at(target=[0, 0, 0], 
#                                                  origin=[0, 0, 3], 
#                                                  up=[0, 1, 0]),
#         'film': {
#             'type': 'hdrfilm', 
#             'width': render_res, 
#             'height': render_res, 
#             'filter': {'type': 'box'}, 
#             'pixel_format': 'rgba'
#         }
#     })


global RAY

class NRCIntegrator(mi.SamplingIntegrator):
    def __init__(self, 
    nrc_config: configs.NrcConfig, 
    render_config: configs.RenderConfig, 
    props=mi.Properties(), 
    m_max_depth=7,
    m_rr_depth=5, 
    device='cuda'
    ):
        # super().__init__(props)
        super().__init__(props)
        self.m_max_depth = m_max_depth
        self.m_rr_depth = m_rr_depth
        self.device = device
        self.nrc_config = nrc_config
        self.render_config = render_config

    def sample(self, scene, sampler, ray, medium=None,
               active=True, **kwargs):       

        ray = mi.Ray3f(ray)
        prev_si = dr.zeros(mi.Interaction3f)
        bsdf_ctx = mi.BSDFContext()

        B = len(ray.o[0])
        throughput = mi.Spectrum(1)
        # result = np.zeros_like(throughput)
        result = mi.Spectrum(0)
        eta = mi.Float(1)
        depth = mi.UInt32(0)
        active = mi.Bool(active)
        prev_bsdf_delta = mi.Bool(True)
        prev_bsdf_pdf = mi.Float(1.0)

        valid_ray = mi.Bool(True)


        # for depth_i in tqdm(range(self.render_config.short_rendering_bounces), desc='Short renderings'):



        while (dr.any(active)):
            # ---------------------- Direct emission ----------------------
            si = scene.ray_intersect(ray)
            ds = mi.DirectionSample3f(scene, si, prev_si)

            mis_bsdf = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )
            result = throughput * \
                ds.emitter.eval(si, prev_bsdf_pdf > 0.) * mis_bsdf + \
                result

            active_next = (depth + 1 < self.m_max_depth) & si.is_valid()

            # ---------------------- Emitter Sampling ----------------------

            bsdf = si.bsdf(ray)
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            if (dr.any(active_em)):
                ds, em_weight = scene.sample_emitter_direction(
                                    si, sampler.next_2d(), True, active_em)
                active_em &= dr.neq(ds.pdf, 0.)

                wo = si.to_local(ds.d)
                bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

                # Compute the MIS weight
                mis_em = dr.select(ds.delta, 1., mis_weight(ds.pdf, bsdf_pdf))

                result[active_em] = (throughput * bsdf_val * em_weight * mis_em + result)[active_em]

            # ---------------------- BSDF Sampling ----------------------

            sample_1 = sampler.next_1d()
            sample_2 = sampler.next_2d()

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sample_1, sample_2, active_next)
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            # ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight
            eta *= bsdf_sample.eta
            valid_ray = valid_ray | (active & si.is_valid()) & (~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null))

            # Information about the current vertex needed by the next iteration
            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1

            throughput_max = dr.max(mi.unpolarized_spectrum(throughput))

            rr_prob = dr.minimum(throughput_max * dr.sqr(eta), 0.95)
            rr_active = depth >= self.m_rr_depth

            rr_continue = sampler.next_1d() < rr_prob


            active = active_next & ((~rr_active) | rr_continue) & \
                        dr.neq(throughput_max, 0.)

        return dr.select(valid_ray, result, 0.), valid_ray, []


nrc_int = NRCIntegrator(nrc_config, render_config, m_max_depth=3, m_rr_depth=3)
ref_image = mi.render(scene, sensor=scene.sensors()[0], integrator=nrc_int, spp=256)
bmp_img = mi.Bitmap(ref_image)
bmp_img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True).write('test.png')

# imageio.imwrite('test.png', ref_image)
print('done!')



# class NRCIntegrator(mi.SamplingIntegrator):
#     def __init__(self, props=mi.Properties(), m_max_depth=7, m_rr_depth=5, device='cuda'):
#         # super().__init__(props)
#         super().__init__(props)
#         self.m_max_depth = m_max_depth
#         self.m_rr_depth = m_rr_depth
#         self.device = device

#     def sample(self, scene, sampler, ray, medium=None,
#                active=True, **kwargs):       

#         ray = mi.Ray3f(ray)
#         prev_si = dr.zeros(mi.Interaction3f)
#         bsdf_ctx = mi.BSDFContext()

#         B = len(ray.o[0])
#         throughput = np.ones([B, 3], dtype=np.float32)
#         result = np.zeros_like(throughput)
#         eta = np.ones([B], dtype=np.float32)
#         depth = np.zeros([B], dtype=np.int32)
#         active = np.ones([B], dtype=bool)
#         prev_bsdf_delta = np.ones([B], dtype=bool)
#         prev_bsdf_pdf = np.ones([B], dtype=np.float32)


#         # # FIXME: why always False?
#         # valid_ray = mi.Bool(dr.neq(scene.environment(), None))
#         valid_ray = np.ones([B], dtype=bool)


#         while (active.any()):
#             # ---------------------- Direct emission ----------------------
#             si = scene.ray_intersect(ray)
#             ds = mi.DirectionSample3f(scene, si, prev_si)

#             em_pdf = np.zeros_like(prev_bsdf_pdf)
#             em_pdf[~prev_bsdf_delta] = np.array(scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta))[~prev_bsdf_delta]

#             mis_bsdf = np.array(mi.ad.common.mis_weight(mi.Float(prev_bsdf_pdf), mi.Float(em_pdf)))
#             result = throughput * \
#                 np.array(ds.emitter.eval(si, prev_bsdf_pdf > 0.)) * mis_bsdf[:, None] + \
#                 result

#             active_next = (depth + 1 < self.m_max_depth) & si.is_valid()


#             if not active_next.any():
#                 break


#             # ---------------------- Emitter Sampling ----------------------

#             bsdf = si.bsdf(ray)
#             active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

#             if (active_em.any()):
#                 ds, em_weight = scene.sample_emitter_direction(
#                                     si, sampler.next_2d(), True, active_em)
#                 active_em &= ds.pdf != 0.

#                 wo = si.to_local(ds.d)
#                 bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
#                 bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

#                 # Compute the MIS weight
#                 mis_em = np.where(ds.delta, 1., mi.ad.common.mis_weight(ds.pdf, bsdf_pdf))

#                 result[active_em] = (throughput * bsdf_val * em_weight * mis_em[:, None] + result)[active_em]

#             # ---------------------- BSDF Sampling ----------------------

#             sample_1 = sampler.next_1d()
#             sample_2 = sampler.next_2d()

#             bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sample_1, sample_2, active_next)
#             bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

#             ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

#             # ------ Update loop variables based on current interaction ------

#             throughput *= bsdf_weight
#             eta *= bsdf_sample.eta
#             valid_ray = valid_ray | (active & si.is_valid()) & (~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null))

#             # Information about the current vertex needed by the next iteration
#             prev_si = si
#             prev_bsdf_pdf = bsdf_sample.pdf
#             prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

#             # -------------------- Stopping criterion ---------------------

#             depth[si.is_valid()] += 1

#             throughput_max = np.max(mi.unpolarized_spectrum(throughput), axis=-1)

#             rr_prob = np.minimum(throughput_max * np.square(eta), 0.95)
#             rr_active = depth >= self.m_rr_depth

#             rr_continue = sampler.next_1d() < rr_prob


#             active = active_next & ((~rr_active) | rr_continue) & \
#                         (throughput_max != 0.)

#         return mi.Spectrum(np.where(valid_ray[:, None], result, np.zeros_like(result))), mi.Bool(valid_ray), []




        # loop = mi.Loop(name=f"NRC loop",
        #                state=lambda: (sampler, ray, throughput, result, eta, depth, valid_ray,
        #                               prev_si, active, prev_bsdf_delta, prev_bsdf_pdf))

        # loop.set_max_iterations(self.m_max_depth)


        # while(loop(active)):

        # # use python loop
        # # convert to numpy

        
        #     # ---------------------- Direct emission ----------------------

        #     # dr::Loop implicitly masks all code in the loop using the 'active'
        #     # flag, so there is no need to pass it to every function 
        #     si = scene.ray_intersect(ray)
        #     ds = mi.DirectionSample3f(scene, si, prev_si)

        #     em_pdf = mi.Float(0.)
        #     if not loop(prev_bsdf_delta): # loop(~prev_bsdf_delta) ?
        #                 em_pdf = scene.pdf_emitter_direction(prev_si, ds,
        #                                                         not prev_bsdf_delta)

        #     mis_bsdf = mi.ad.common.mis_weight(prev_bsdf_pdf, em_pdf)
        #     result = throughput * \
        #         ds.emitter.eval(si, prev_bsdf_pdf > 0.) * mis_bsdf + \
        #         result

        #     active_next = (depth + 1 < self.m_max_depth) & si.is_valid()


        #     if not dr.any(loop(active_next)):
        #         break


        #     # ---------------------- Emitter Sampling ----------------------

        #     bsdf = si.bsdf(ray)
        #     active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        #     if (dr.any(active_em)):
        #         ds, em_weight = scene.sample_emitter_direction(
        #                             si, sampler.next_2d(), True, active_em)
        #         active_em &= dr.neq(ds.pdf, 0.)

        #         wo = si.to_local(ds.d)
        #         bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        #         # bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

        #         # Compute the MIS weight
        #         # mis_em = dr.select(ds.delta, 1., mi.ad.common.mis_weight(ds.pdf, bsdf_pdf))

        #         # result[active_em] = throughput * bsdf_val * em_weight * mis_em + result

        #     # ---------------------- BSDF Sampling ----------------------

        #     sample_1 = sampler.next_1d()
        #     sample_2 = sampler.next_2d()

        #     bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sample_1, sample_2, active_next)
        #     bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

        #     ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        #     # ------ Update loop variables based on current interaction ------

        #     throughput *= bsdf_weight
        #     eta *= bsdf_sample.eta
        #     valid_ray = valid_ray | (active & si.is_valid()) & (~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null))

        #     # Information about the current vertex needed by the next iteration
        #     prev_si = si
        #     prev_bsdf_pdf = bsdf_sample.pdf
        #     prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        #     # -------------------- Stopping criterion ---------------------

        #     # dr.masked(depth, si.is_valid()) += 1
        #     depth[si.is_valid()] += 1

        #     throughput_max = dr.max(mi.unpolarized_spectrum(throughput))

        #     rr_prob = dr.minimum(throughput_max * dr.sqr(eta), 0.95)
        #     rr_active = depth >= self.m_rr_depth

        #     rr_continue = sampler.next_1d() < rr_prob


        #     active = active_next & ((~rr_active) | rr_continue) & \
        #                 dr.neq(throughput_max, 0.)

        # return dr.select(valid_ray, result, 0), valid_ray, dr.zeros(mi.Float)


# nrc_int = NRCIntegrator(m_max_depth=3, m_rr_depth=3)
# ref_image = mi.render(scene, sensor=sensor, integrator=nrc_int, spp=256)






