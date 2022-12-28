import os
import torch
from torch import nn
from model import encoder
from model.utils import to_sphe_coords
import numpy as np
from copy import deepcopy

class NRC_Core(nn.Module):
    def __init__(self, input_dim=64, ref_factor=True, dtype=torch.float32):
        '''
            Note that the actual input dim is 62
            We pad two colums of 1 to allow implicit bias

            ira_factor: Reflectance factorization
        '''
        super(NRC_Core, self).__init__()
        self.ref_factor = ref_factor
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 3, bias=False).to(dtype=dtype),
        )

        self.freq_embed = encoder.FreqEmbed()
        self.blob_embd = encoder.OneBlobEmbed()


    @property
    def device(self):
        return next(self.mlp[0].parameters()).device

    def forward(self, p, wi, n, alpha, beta, r):
        '''
        p: [B, 3] position
        wi: [B, 3] scattered dir
        n: [B, 3] surface normal
        alpha: [B, 3] diffuse reflectance
        beta: [B, 3] specular reflectance
        r: [B, 1] surface roughness
        '''

        input = torch.concat(
            [
                self.freq_embed.embed(p),
                self.blob_embd.embed(to_sphe_coords(wi)),
                self.blob_embd.embed(to_sphe_coords(n)),
                self.blob_embd.embed(1.-torch.exp(-r)),
                alpha,
                beta,
                torch.ones_like(p[:,:2])
            ],
            axis=-1
        )


        out = self.mlp(input)

        if self.ref_factor:
            out = out * (alpha + beta)
        return out

class NRC_MLP:
    def __init__(self, input_dim=64, ref_factor=True, ma_alpha=0.99, dtype=torch.float32, device='cuda'):
        '''
            Note that the actual input dim is 62
            We pad two colums of 1 to allow implicit bias

            ref_factor: Reflectance factorization
            ma_alpha: moving average alpha
        '''

        self.train_mlp = NRC_Core(input_dim, ref_factor, dtype).to(device)
        self.inference_mlp = deepcopy(self.train_mlp)

        # moving average alpha
        self.ma_alpha = ma_alpha
        self.t = 0

        self.dtype = dtype
        self.num_rendered_frame = 0
        self.device = device



    def _to_tensor(self, x):
        return torch.tensor(x, device=self.device, dtype=self.dtype)


    def inference_np(self, p, wi, n, alpha, beta, r):
        '''
        Forward function for np input
        '''

        with torch.no_grad():
            ret = self.inference_mlp(
            self._to_tensor(p),
            self._to_tensor(wi),
            self._to_tensor(n),
            self._to_tensor(alpha),
            self._to_tensor(beta),
            self._to_tensor(r),
            ).detach().cpu().numpy()

        return ret

    def nrc_train(self, inputs, outputs, optimizer, batch_num=4, relative_loss=True):
        '''
        batch_num: slipt data into several batches and perform multiple training steps
        relative_loss: needed when expected outputs are noisy. Paper Sec 5
        '''
        mean_loss = 0.

        outputs = self._to_tensor(outputs)
        p, wi, n, alpha, beta, r = inputs

        # assert outputs.shape[0] % batch_num == 0, 'Batch slipt uneven!'

        ids = np.array_split(np.random.permutation(outputs.shape[0]), batch_num)
        
        # multiple steps per frame
        for i in range(batch_num):
            optimizer.zero_grad()
            pred = self.train_mlp(
                self._to_tensor(p[ids[i]]),
                self._to_tensor(wi[ids[i]]),
                self._to_tensor(n[ids[i]]),
                self._to_tensor(alpha[ids[i]]),
                self._to_tensor(beta[ids[i]]),
                self._to_tensor(r[ids[i]]),
            )
            loss = (outputs[ids[i]] - pred)**2
            if relative_loss:
                loss = loss / (pred.detach().clone()**2 + 0.01)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

            # update inference mlp
            self.t += 1
            state_wt = self.train_mlp.state_dict()
            state_inf = self.inference_mlp.state_dict()
            for key in state_inf:
                state_inf[key] = (1 - self.ma_alpha) / (1-(self.ma_alpha**self.t)) * state_wt[key] \
                                + self.ma_alpha * (1-(self.ma_alpha**(self.t-1))) * state_inf[key]


            mean_loss += loss

        return mean_loss / batch_num

        


    # def 



# B = 64
# device = 'cuda'
# nrc_mlp = NRC_MLP().to(device)

# pos = torch.rand([B, 3], device=device)
# w = torch.rand([B, 3], device=device)
# n = torch.rand([B, 3], device=device)
# r = torch.rand([B, 1], device=device)
# alpha = torch.rand([B, 3], device=device)
# beta = torch.rand([B, 3], device=device)

# out = nrc_mlp(pos, w, n, alpha, beta, r)


# print(out)