import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GLN(nn.Module):
    def __init__(self, embed_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 1e-05
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.ones(self.embed_dim, 1))
        self.bias = nn.Parameter(torch.zeros(self.embed_dim, 1))
    
    def forward(self, x):
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.var(x, (1, 2), unbiased=False, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias



class TCN(nn.Module):
    def __init__(self, in_channels=256, out_channels=512, kernel_size=3, dilation=1, speaker=0, *args, **kwargs) -> None:  
        super().__init__(*args, **kwargs)
        self.conv_1 = nn.Conv1d(in_channels + speaker, out_channels, kernel_size=1)
        self.conv_2 = nn.Conv1d(out_channels, in_channels, kernel_size=1)

        self.activation_1 = nn.PReLU()
        self.activation_2 = nn.PReLU()

        self.de_conv = nn.Conv1d(out_channels, 
                                 out_channels, 
                                 kernel_size=kernel_size, 
                                 groups=out_channels, 
                                 dilation=dilation, 
                                 padding=(kernel_size - 1) * dilation // 2)
        
        self.norm_1 = GLN(embed_dim=out_channels)
        self.norm_2 = GLN(embed_dim=out_channels)

        self.has_speaker = speaker != 0


    def forward(self, inp):
        x = inp[0]
        speaker = inp[1]
        if self.has_speaker:
            last_shape = x.shape[-1]
            speaker = speaker.unsqueeze(-1)
            speaker = speaker.repeat(1, 1, last_shape)
            new_x = torch.cat((x, speaker), 1)
            new_x = self.conv_1(new_x)
        else:
            new_x = self.conv_1(x)

        new_x = self.activation_1(new_x)
        new_x = self.norm_1(new_x)
        new_x = self.de_conv(new_x)
        new_x = self.activation_2(new_x)
        new_x = self.norm_2(new_x)
        new_x = self.conv_2(new_x)
        return new_x + x, speaker

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        self.conv_eq = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.activation_1 = nn.PReLU()
        self.activation_2 = nn.PReLU()

        self.norm_1 = nn.BatchNorm1d(out_channels)
        self.norm_2 = nn.BatchNorm1d(out_channels)

        self.max_pool = nn.MaxPool1d(3)

        self.need_proj = (in_channels != out_channels)
    
    def forward(self, x):
        new_x = self.conv_1(x)
        new_x = self.norm_1(new_x)
        new_x = self.activation_1(new_x)
        new_x = self.conv_2(new_x)
        new_x = self.norm_2(new_x)
        new_x = new_x + self.conv_eq(x) if self.need_proj else new_x + x
        new_x = self.activation_2(new_x)
        return self.max_pool(new_x)
    

class SpEx(nn.Module):
    def __init__(self,
                 short_length=20, 
                 middle_length=80,
                 long_length=160,
                 in_channels=256,
                 stack_size=8,
                 num_stacks=4,
                 kernel_size=3,
                 proj_channels=256,
                 out_channels=512, 
                 n_speakers=101,
                 speaker_dim=256,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder_short = nn.Sequential(
            nn.Conv1d(1, in_channels, kernel_size=short_length, stride=short_length // 2),
            nn.ReLU()
        )
        self.encoder_middle = nn.Sequential(
            nn.ConstantPad1d((0, middle_length - short_length), 0),
            nn.Conv1d(1, in_channels, kernel_size=middle_length, stride=short_length // 2), 
            nn.ReLU()
        )
        self.encoder_long = nn.Sequential(
            nn.ConstantPad1d((0, long_length - short_length), 0),
            nn.Conv1d(1, in_channels, kernel_size=long_length, stride=short_length // 2), 
            nn.ReLU()
        )

        self.layer_norm = nn.LayerNorm(in_channels * 3)

        self.projector = nn.Conv1d(in_channels * 3, proj_channels, 1)

        self.stacks = nn.ModuleList()

        for num in range(num_stacks):
            stack = nn.Sequential(
                *[TCN(proj_channels, out_channels, kernel_size, dilation=(2**i), speaker=((i == 0) * speaker_dim)) for i in range(stack_size)]
            )
            self.stacks.add_module(f"stack_{num + 1}", stack)
        
        self.short_mask = nn.Sequential(
            nn.Conv1d(proj_channels, in_channels, 1),
            nn.ReLU()
        )
        self.middle_mask = nn.Sequential(
            nn.Conv1d(proj_channels, in_channels, 1),
            nn.ReLU()
        )
        self.long_mask = nn.Sequential(
            nn.Conv1d(proj_channels, in_channels, 1),
            nn.ReLU()
        )

        self.decoder_short = nn.ConvTranspose1d(in_channels, 1, kernel_size=short_length, stride=short_length // 2)
        self.decoder_middle = nn.ConvTranspose1d(in_channels, 1, kernel_size=middle_length, stride=short_length // 2)
        self.decoder_long = nn.ConvTranspose1d(in_channels, 1, kernel_size=long_length, stride=short_length // 2)

        self.encoder_norm = nn.LayerNorm(in_channels * 3)
        self.encoder_projector = nn.Conv1d(in_channels * 3, proj_channels, 1)

        self.resnet = nn.Sequential(
            *[ResNet(in_channels=proj_channels, out_channels=proj_channels),
              ResNet(in_channels=proj_channels, out_channels=out_channels),
              ResNet(in_channels=out_channels, out_channels=out_channels)]
        )

        self.conv_speaker = nn.Conv1d(out_channels, speaker_dim, 1)

        self.mean_pooling = nn.AvgPool1d(1)

        self.cls = nn.Linear(speaker_dim, n_speakers)

        self.short_length = short_length
        self.middle_length = middle_length
        self.long_length = long_length
        self.n_speakers = n_speakers
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, mixes, references, references_length, **kwargs):
        ###  SPEAKER ENCODER  ###
        enc_spk_short = self.encoder_short(references)
        enc_spk_middle = self.encoder_middle(references)
        enc_spk_long = self.encoder_long(references)

        enc_spk_out = torch.cat((enc_spk_short, enc_spk_middle, enc_spk_long), 1)
        new_speaker = self.encoder_norm(enc_spk_out.transpose(1, 2)).transpose(1, 2)
        new_speaker = self.encoder_projector(new_speaker)
        new_speaker = self.resnet(new_speaker)
        new_speaker = self.conv_speaker(new_speaker)
        speaker_emb = new_speaker.sum(dim=-1)

        new_length = (references_length - self.short_length) // (self.short_length // 2) + 1
        new_length = ((new_length // 3) // 3) // 3

        speaker_emb = speaker_emb / new_length.reshape(-1, 1)

        speaker_pred = self.cls(speaker_emb)
        ###  SPEAKER ENCODER  ###

        ### SPEAKER EXTRACTOR ###
        enc_short = self.encoder_short(mixes)
        enc_middle = self.encoder_middle(mixes)
        enc_long = self.encoder_long(mixes)

        short_len = mixes.shape[-1]

        enc_out = torch.cat((enc_short, enc_middle, enc_long), 1)

        new_x = self.layer_norm(enc_out.transpose(1, 2)).transpose(1, 2)
        new_x = self.projector(new_x)

        for module in self.stacks:
            new_x, _ = module((new_x, speaker_emb))
        
        s_mask = self.short_mask(new_x)
        m_mask = self.middle_mask(new_x)
        l_mask = self.long_mask(new_x)

        
        s_s = self.decoder_short(s_mask * enc_short)
        decoder_len = s_s.shape[-1]
        s_s = F.pad(s_s, (0, short_len - decoder_len))
        s_m = self.decoder_middle(m_mask * enc_middle)[:, :, :short_len]
        s_l = self.decoder_long(l_mask * enc_long)[:, :, :short_len]
        ### SPEAKER EXTRACTOR ###

        return {
            "short": s_s, 
            "middle": s_m, 
            "long": s_l, 
            "logits": speaker_pred
        }
