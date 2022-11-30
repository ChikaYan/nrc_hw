import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
from mitsuba.cuda_ad_rgb import Color3f
import numpy as np
import torch
import imageio
import gin
from model.model import NRC_MLP
from model import configs
# import mitsuba.ad.common.mis_weight 
import os
from model.integrator import NRCIntegrator


args = configs.parse_args()

gin.parse_config_files_and_bindings(
    config_files=[args.config],
    bindings=None, #FLAGS.gin_bindings, # additional configs
    skip_unknown=False)

nrc_config = configs.NrcConfig()
render_config = configs.RenderConfig()

# scene = mi.load_file('mitsuba-tutorials/scenes/cbox.xml')
scene = mi.load_file(render_config.scene_path)


# B = 128
# device = 'cuda'
# nrc_mlp = NRC_MLP().to(device)

# pos = torch.rand([B, 3], device=device).double()
# w = torch.rand([B, 3], device=device).double()
# n = torch.rand([B, 3], device=device).double()
# r = torch.rand([B, 1], device=device).double()
# alpha = torch.rand([B, 3], device=device).double()
# beta = torch.rand([B, 3], device=device).double()

# out = nrc_mlp(pos, w, n, alpha, beta, r)
# print(out)



nrc_net = NRC_MLP().to('cuda')
nrc_int = NRCIntegrator(nrc_net, nrc_config, render_config, m_max_depth=3, m_rr_depth=3)
sensor = scene.sensors()[0]
ref_image = mi.render(scene, sensor=sensor, integrator=nrc_int, spp=1)
bmp_img = mi.Bitmap(ref_image)
bmp_img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True).write('test.png')

# imageio.imwrite('test.png', ref_image)
print('done!')


