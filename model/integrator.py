import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
from mitsuba.cuda_ad_rgb import Color3f
import numpy as np
import torch
import imageio
import gin
from model.model import NRC_MLP
from tqdm import tqdm
from model import configs
# import mitsuba.ad.common.mis_weight 
from model.utils import BounceInfo, ReflectanceInfo, vertex_spread_dr, primary_vertex_spread_dr
import os

# from mitsuba.ad.common import mis_weight

mis_weight = mi.ad.common.mis_weight


class NRCIntegrator(mi.SamplingIntegrator):
    def __init__(self,
    nrc_net: NRC_MLP,
    nrc_config: configs.NrcConfig, 
    render_config: configs.RenderConfig,
    props=mi.Properties(), 
    m_max_depth=7,
    m_rr_depth=5
    ):
        # super().__init__(props)
        super().__init__(props)
        self.nrc_config = nrc_config
        self.render_config = render_config
        self.nrc_net = nrc_net
        self.m_max_depth = m_max_depth
        self.m_rr_depth = m_rr_depth
        self.device = nrc_net.device
        self.optimizer = torch.optim.Adam(params=nrc_net.parameters(), lr=nrc_config.optim_lr)

    def sample(self, scene, sampler, ray, medium=None,
               active=True, **kwargs):       

        ray = mi.Ray3f(ray)
        prev_si = dr.zeros(mi.Interaction3f)
        bsdf_ctx = mi.BSDFContext()

        B = len(ray.o[0])
        throughput = mi.Spectrum(1)
        # result = np.zeros_like(throughput)
        L = mi.Spectrum(0)
        eta = mi.Float(1)
        depth = mi.UInt32(0)
        active = mi.Bool(active)
        active_next = active # TODO: is copy needed?
        prev_bsdf_delta = mi.Bool(True)
        prev_bsdf_pdf = mi.Float(1.0)
        terminated_paths = ~active

        bounce_info = BounceInfo()


        # ------------------ Short Depth Rendering ---------------------
        for depth_i in tqdm(range(self.render_config.short_depth), desc=f'Frame {self.nrc_net.num_rendered_frame} Short Renderings'):
            # while (dr.any(active)):
            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))
            bsdf = si.bsdf(ray)
            # Create reflectance dictionary using scene.xml
            idx_array = dr.reinterpret_array_v(mi.UInt32, si.bsdf())
            reflectance_info = ReflectanceInfo(self.render_config.scene_path, idx_array, bsdf)

            # Should we continue tracing to reach one more vertex?
            active_next &= si.is_valid()

            # ---------------------- Direct emission ----------------------
            ds = mi.DirectionSample3f(scene, si, prev_si)

            mis_bsdf = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            inst_emitter_rad = mis_bsdf * ds.emitter.eval(si, active=active_next)
            inst_emitter_rad = dr.select(active_next, inst_emitter_rad, 0) # Maybe unnecessary
            rew_emitter_rad = throughput * inst_emitter_rad


            # ------------------ Detached BSDF sampling -------------------
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)


            # ------------------ NRC  inference----------------------------
            # NRC # 2 # Path Termination: terminate the ray if the spread is sufficiently large (r>t)
            if depth_i == 0:
                # spread at the primary vertex
                a_0 = primary_vertex_spread_dr(x0=ray.o, x1=si.p, wi=si.to_world(bsdf_sample.wo), n=si.n)
                a_sum = 0
            else:
                # spread at subsequent vertexes
                a_curr = vertex_spread_dr(x_i_1=ray.o, x_i=si.p, pdf_bsdf=bsdf_sample.pdf, wi=si.to_world(bsdf_sample.wo), n=si.n)
                a_sum = a_sum + a_curr
                a_subpath = (a_sum)**2

                # path termination condition
                terminated_paths = a_subpath > self.render_config.path_termination_c * a_0

                print(f'Ratio terminated paths: {dr.count(terminated_paths)/dr.count(active_next)}')
                t_paths = terminated_paths.numpy()

                if depth_i == 1:
                    # # NRC # 6 # selection of specific number of the rays (choosen among the rays which are terminated by the max_bounce metric)
                    random_ind = np.random.choice(B,
                                                size=self.nrc_config.training_samples,
                                                replace=False, 
                                                p=t_paths/t_paths.sum())
                    extend_active_np = np.zeros(B)
                    extend_active_np[random_ind] = 1
                    extend_active = mi.Bool(extend_active_np)

                    # Disable the terminated paths that we will extend, so NRC is not computed and we keep tracing them.
                    terminated_paths = terminated_paths & ~extend_active


            # NRC # 3 # Input data collection: collect all the necessary data to create the NRC input (position, scattered direction, surface normal/roughness, diffuse/specular reflectance)
            # Get diffuse and specular reflectance and roughness
            diffuse_reflectance, specular_reflectance, roughness = reflectance_info.get_properties(idx_array)
            if self.nrc_config.reflectance_factorisation:
                specular_reflectance = (bsdf_weight+rew_emitter_rad)

            if self.render_config.visualise_primary and depth_i == 0:
                print('NRC inference at primary vertex')
                inference_results = self.nrc_net.inference_np(
                    si.p.numpy(), si.wi.numpy(), si.n.numpy(), 
                    diffuse_reflectance, specular_reflectance, roughness)
                inference_results = np.clip(inference_results, 0, None)
                nrc_results       = Color3f(inference_results)
                filtered_nrc_results = dr.select(terminated_paths, Color3f(inference_results),0)
                LprimaryNRC = nrc_results
                    
            # NRC # 4 # NRC query: for all the terminated rays, query the neural network.
            if self.nrc_net.num_rendered_frame > 0 and depth_i > 0:
                print('NRC inference')
                # Select the path to be continued here
                
                active_NRC_paths = active_next & terminated_paths
                
                # TODO keep indices and only provide the non_zero values to the network
                inference_results = self.nrc_net.inference_np(
                    dr.select(active_NRC_paths, si.p, 0).numpy(),
                    dr.select(active_NRC_paths, si.wi, 0).numpy(),
                    dr.select(active_NRC_paths, si.n, 0).numpy(),
                    diffuse_reflectance, specular_reflectance, roughness)

                inference_results = np.clip(inference_results, 0, None)
                filtered_nrc_results = dr.select(active_NRC_paths, Color3f(inference_results),0)
                if self.nrc_config.activate_nrc:
                    L = L + throughput*filtered_nrc_results

                # Update path activation arrays after querying NRC
                active_next &= ~terminated_paths
            else:
                # if total_rendered_frames==0 and idx==1 we don't want to filter out the terimnated paths
                # because we will not do inference for them anyway! For the terminated paths we will do
                # direct and emitter sampling instead. This is why I add the following if statement:
                if depth_i == 0:
                    active_next &= ~terminated_paths

            
            print(f'Frame {self.nrc_net.num_rendered_frame}, Bounce id {depth_i}, Next active rays {dr.count(active_next)}')

            # ---------------------- Emitter sampling ----------------------

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)
            # Evaluate BSDF * cos(theta)
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            mis_em = dr.select(ds.delta, 1., mis_weight(ds.pdf, bsdf_pdf_em))
            inst_bsdf_rad = mis_em * bsdf_value_em * em_weight
            # inst_bsdf_rad = dr.select(active_next, inst_bsdf_rad, 0) # unnecessary
            rew_bsdf_rad = throughput * inst_bsdf_rad

            
            # ------------------ Add radiances ----------------------------
            if not self.render_config.output_only_nrc:
                L = (L + rew_emitter_rad + rew_bsdf_rad) 

            # ------------------ NRC  training ----------------------------
            if depth_i == 0 or self.nrc_net.num_rendered_frame == 0:
                bounce_info.log_data(depth_i, si, active_next, bsdf_weight, inst_emitter_rad + inst_bsdf_rad, 0,
                                    diffuse_reflectance, specular_reflectance, roughness)
            else:
                ## Include NRC output in the training data
                bounce_info.log_data(depth_i, si, active_NRC_paths, bsdf_weight, 
                                    inst_emitter_rad + inst_bsdf_rad, filtered_nrc_results.numpy(),
                                    diffuse_reflectance, specular_reflectance, roughness)

            if self.nrc_config.first_frame_training and self.nrc_net.num_rendered_frame == 0:
                ## First frame ==> Full training 1 epoch
                if depth_i == 1:
                    nrc_inputs, nrc_outputs = bounce_info.gather_training_data()
                    print('Start full epoch training')

                    loss = self.nrc_net.nrc_train(nrc_inputs, nrc_outputs, 
                        self.optimizer, num_step=self.nrc_config.steps_per_frame)
                    print('Final training Loss: {}'.format(loss))

            # ---- Update loop variables based on current interaction -----
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            eta *= bsdf_sample.eta
            throughput *= bsdf_weight

            # Information about the current vertex needed by the next iteration
            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)


            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1

            throughput_max = dr.max(mi.unpolarized_spectrum(throughput))

            rr_prob = dr.minimum(throughput_max * dr.sqr(eta), 0.95)
            rr_active = depth >= self.m_rr_depth
            
            # TODO: ?
            throughput[rr_active] *= dr.rcp(rr_prob) # rcp = 1/x
            rr_continue = sampler.next_1d() < rr_prob


            active = active_next & ((~rr_active) | rr_continue) & \
                        dr.neq(throughput_max, 0.)



        # ------------------ Short Depth Rendering ---------------------

        # # NRC # 6 # selection of specific number of the rays (choosen among the rays which are terminated by the max_bounce metric)
        active_next = extend_active
        
        for depth_i in tqdm(range(self.render_config.short_depth, self.render_config.long_depth), desc=f'Frame {self.nrc_net.num_rendered_frame} Long Renderings'):
            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))
            bsdf = si.bsdf(ray)
            # Create reflectance dictionary using scene.xml
            idx_array = dr.reinterpret_array_v(mi.UInt32, si.bsdf())
            reflectance_info = ReflectanceInfo(self.render_config.scene_path, idx_array, bsdf)

            # Should we continue tracing to reach one more vertex?
            active_next &= si.is_valid()

            # ---------------------- Direct emission ----------------------
            ds = mi.DirectionSample3f(scene, si, prev_si)

            mis_bsdf = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            inst_emitter_rad = mis_bsdf * ds.emitter.eval(si, active=active_next)
            inst_emitter_rad = dr.select(active_next, inst_emitter_rad, 0) # Maybe unnecessary
            rew_emitter_rad = throughput * inst_emitter_rad

            # ------------------ Detached BSDF sampling -------------------
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)
            
            # ------------------ NRC  inference----------------------------
            # NRC # 2 # Path Termination: terminate the ray if the spread is sufficiently large (r>t)
            if depth_i == self.render_config.short_depth:
                # spread at the primary vertex
                a_0 = primary_vertex_spread_dr(x0=ray.o, x1=si.p, wi=si.to_world(bsdf_sample.wo), n=si.n)
                a_sum = 0
            else:
                # spread at subsequent vertexes
                a_curr = vertex_spread_dr(x_i_1=ray.o, x_i=si.p, pdf_bsdf=bsdf_sample.pdf, wi=si.to_world(bsdf_sample.wo), n=si.n)
                a_sum = a_sum + a_curr
                a_subpath = (a_sum)**2

                # path termination condition
                terminated_paths = a_subpath > self.render_config.path_termination_c * a_0

                # TODO: this can give > 1!
                print(f'Ratio terminated paths: {dr.count(terminated_paths)/dr.count(active_next)}')


            # NRC # 3 # Input data collection: collect all the necessary data to create the NRC input (position, scattered direction, surface normal/roughness, diffuse/specular reflectance)
            # Get diffuse and specular reflectance and roughness
            diffuse_reflectance, specular_reflectance, roughness = reflectance_info.get_properties(idx_array)
            if self.nrc_config.reflectance_factorisation:
                specular_reflectance = (bsdf_weight+rew_emitter_rad)

            # NRC # 4 # NRC query: for all the terminated rays, query the neural network.
            if depth_i > self.render_config.short_depth and self.nrc_net.num_rendered_frame > 0:
                print('NRC inference')
                # Select the path to be continued here
                
                active_NRC_paths = active_next & terminated_paths
                
                # TODO keep indices and only provide the non_zero values to the network
                inference_results = self.nrc_net.inference_np(
                    dr.select(active_NRC_paths, si.p, 0).numpy(),
                    dr.select(active_NRC_paths, si.wi, 0).numpy(),
                    dr.select(active_NRC_paths, si.n, 0).numpy(),
                    diffuse_reflectance, specular_reflectance, roughness)

                inference_results = np.clip(inference_results, 0, None)
                filtered_nrc_results = dr.select(active_NRC_paths, Color3f(inference_results),0)
                if self.nrc_config.activate_nrc:
                    L = L + throughput*filtered_nrc_results

                # Update path activation arrays after querying NRC
                active_next &= ~terminated_paths

            print(f'Frame {self.nrc_net.num_rendered_frame}, Bounce id {depth_i}, Next active rays {dr.count(active_next)}')
            # ---------------------- Emitter sampling ----------------------

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)
            # Evaluate BSDF * cos(theta)
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            mis_em = dr.select(ds.delta, 1., mis_weight(ds.pdf, bsdf_pdf_em))
            inst_bsdf_rad = mis_em * bsdf_value_em * em_weight
            # inst_bsdf_rad = dr.select(active_next, inst_bsdf_rad, 0) # unnecessary
            rew_bsdf_rad = throughput * inst_bsdf_rad

            # ------------------ Add radiances ----------------------------
            if not self.render_config.output_only_nrc:
                L = (L + rew_emitter_rad + rew_bsdf_rad) 

            # ------------------ NRC  training ----------------------------
            if depth_i > self.render_config.short_depth and self.nrc_net.num_rendered_frame > 0:
                bounce_info.log_data(depth_i, si, active_NRC_paths, bsdf_weight, 
                                    inst_emitter_rad + inst_bsdf_rad, filtered_nrc_results.numpy(),
                                    diffuse_reflectance, specular_reflectance, roughness)
            else:
                bounce_info.log_data(depth_i, si, active_next, bsdf_weight, 
                    inst_emitter_rad + inst_bsdf_rad, 0,
                    diffuse_reflectance, specular_reflectance, roughness)


            ## First frame ==> Full training 1 epoch
            if depth_i == self.render_config.long_depth -1:
                nrc_inputs, nrc_outputs = bounce_info.gather_training_data()

                # Select only a fixed amount training samples uniformly to keep training budget constant
                # TODO perform this before getting training features. Consider decreasing number of extended rays.
                if nrc_outputs.shape[0] > self.nrc_config.training_samples:
                    random_ind = np.random.choice(nrc_outputs.shape[0],
                                    size=self.nrc_config.training_samples,
                                    replace=False)
                    nrc_inputs = [nrc_inputs[i][random_ind] for i in range(len(nrc_inputs))]
                    nrc_outputs = nrc_outputs[random_ind]
                    
                print('Mean radiance',nrc_outputs.mean())
                print('Start extended rays training', nrc_outputs.shape)
                loss = self.nrc_net.nrc_train(nrc_inputs, nrc_outputs, 
                    self.optimizer, num_step=self.nrc_config.steps_per_frame)
                print('Final training Loss: {}'.format(loss))



            # ---- Update loop variables based on current interaction -----
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            eta *= bsdf_sample.eta
            throughput *= bsdf_weight # TODO: bsdf_weight is 0 everywhere!

            # Information about the current vertex needed by the next iteration
            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1

            throughput_max = dr.max(mi.unpolarized_spectrum(throughput))

            rr_prob = dr.minimum(throughput_max * dr.sqr(eta), 0.95)
            rr_active = depth >= self.m_rr_depth
            
            # TODO: ?
            throughput[rr_active] *= dr.rcp(dr.maximum(rr_prob, 1e-10)) # rcp = 1/x
            rr_continue = sampler.next_1d() < rr_prob


            active = active_next & ((~rr_active) | rr_continue) & \
                        dr.neq(throughput_max, 0.)

        if self.render_config.visualise_primary:
            L = LprimaryNRC

        self.nrc_net.num_rendered_frame += 1
        return L, dr.neq(depth, 0), []
