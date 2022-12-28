import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
import numpy as np
import torch
import imageio
import gin
from model.model import NRC_MLP
from model import configs
# import mitsuba.ad.common.mis_weight 
import os
from model.integrator import NRCIntegrator
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


args = configs.parse_args()

gin.parse_config_files_and_bindings(
    config_files=[args.config],
    bindings=None, #FLAGS.gin_bindings, # additional configs
    skip_unknown=False)

nrc_config = configs.NrcConfig()
render_config = configs.RenderConfig()

# scene = mi.load_file('mitsuba-tutorials/scenes/cbox.xml')
scene = mi.load_file(render_config.scene_path)

log_path = Path(render_config.log_dir)
log_path.mkdir(exist_ok=True, parents=True)


nrc_net = NRC_MLP()
writer = SummaryWriter(str(log_path / f'tb'))
nrc_int = NRCIntegrator(nrc_net, nrc_config, render_config, m_max_depth=3, m_rr_depth=3, writer=writer)
sensor = scene.sensors()[0]

for i in range(100):
    ref_image = mi.render(scene, sensor=sensor, integrator=nrc_int, spp=render_config.spp)
    bmp_img = mi.Bitmap(ref_image)
    bmp_img = bmp_img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
    bmp_img.write(str(log_path / f'render_{i:04d}.png'))
    writer.add_image('Rendering', np.array(bmp_img).transpose([2,0,1]), i)

# imageio.imwrite('test.png', ref_image)
print('done!')







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

# torch.autograd.set_detect_anomaly(True)