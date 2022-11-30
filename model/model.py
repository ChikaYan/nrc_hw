import os
import torch
from torch import nn
from model import encoder
from model.utils import to_sphe_coords

class NRC_MLP(nn.Module):
    def __init__(self, input_dim=64, ref_factor=True, dtype=torch.float32):
        '''
            Note that the actual input dim is 62
            We pad two colums of 1 to allow implicit bias

            ira_factor: Reflectance factorization
        '''
        super(NRC_MLP, self).__init__()
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
        self.dtype = dtype
        self.num_rendered_frame = 0


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

    def to_tensor(self, x):
        return torch.tensor(x, device=self.device, dtype=self.dtype)


    def inference_np(self, p, wi, n, alpha, beta, r):
        '''
        Forward function for np input
        '''

        with torch.no_grad:
            ret = self(
            self.to_tensor(p),
            self.to_tensor(wi),
            self.to_tensor(n),
            self.to_tensor(alpha),
            self.to_tensor(beta),
            self.to_tensor(r),
            ).detach().cpu().numpy()

        return ret

    def nrc_train(self, inputs, outputs, optimizer, num_step=4):
        mean_loss = 0.
        loss_fn = torch.nn.MSELoss()

        outputs = self.to_tensor(outputs)

        p, wi, n, alpha, beta, r = inputs
        
        # multiple steps per frame
        for _ in range(num_step):
            optimizer.zero_grad()
            pred = self(
                self.to_tensor(p),
                self.to_tensor(wi),
                self.to_tensor(n),
                self.to_tensor(alpha),
                self.to_tensor(beta),
                self.to_tensor(r),
            )
            loss = loss_fn(outputs, pred)
            loss.backward()
            optimizer.step()
            mean_loss += loss

        return mean_loss / num_step

        


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