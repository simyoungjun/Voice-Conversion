import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append('/home/sim/VoiceConversion/FreeVC')
import commons
import modules_v9 as modules_v9

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

sys.path.append('/home/sim/VoiceConversion/Codebook')
from codebook_utils import spk_emb_corr


class ResidualCouplingBlock(nn.Module):
  '''
  Normalizing Flow
'''
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules_v9.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules_v9.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class Encoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0, vq_codebook_size=None):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.vq_codebook_size = vq_codebook_size

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules_v9.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    #Vector Quantization
    if vq_codebook_size != None:
        self.codebook = modules_v9.VQEmbeddingEMA(vq_codebook_size, hidden_channels)

  def forward(self, x, x_lengths, g=None):
    '''
    z : (N, out_channels, T)
    '''
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)    # Bottleneck Extractor
    
    if self.vq_codebook_size==None: 
        # stats = self.proj(x) * x_mask    # d dim -> d*2 dim
        pass
    else: # Prior Encoder
        x, x_quan, perplexity = self.codebook(x)
        # stats = self.proj(x_quan) * x_mask    # d dim -> d*2 dim
    
    # #Sampling in multivariate gaussian (VAE process)
    # m, logs = torch.split(stats, self.out_channels, dim=1) # 192씩 2개로 -> mu, sigma
    # z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask # Gaussian Distribution 
    
    if self.vq_codebook_size == None:
        return x
    else:
        return x, x_quan, perplexity 
    


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        # self.lin_pre = nn.Linear(1024, 512)
        #적용안함: V9_1024, V9_VQ1024_res_slice_cond_2
        # self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        
        # V9_VQ1024_res_slice_concat
        self.conv_pre_1 = Conv1d(initial_channel, upsample_initial_channel-8, 3, 1, padding='same')
        
        resblock = modules_v9.ResBlock1 if resblock == '1' else modules_v9.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        # concat 부분 떄문에 바뀜
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel-8, 1)
            # self.cond_res = nn.Conv1d(gin_channels, 1, 1)
            # concat 부분 떄문에 바뀜
            self.cond_res = nn.Conv1d(gin_channels, 8, 1)
    def forward(self, x, g=None, res=None):
        # import pdb; pdb.set_trace()
        # x = self.lin_pre(x)
        
        #적용안함: VQ_1024, V9_VQ1024_res_slice_cond_2
        x = self.conv_pre_1(x)
        
        res = self.cond_res(res)
        # print(x.size())
        # print(g.size())
        if g is not None:
            spk_ = self.cond(g)
            x = x + spk_
            
            # print('x.size', x.size())
        x = F.leaky_relu(x)
        x = torch.cat((x, res), dim=1)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules_v9.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules_v9.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules_v9.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
        
        
class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=192, model_embedding_size=192):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames-partial_frames, partial_hop):
            mel_range = torch.arange(i, i+partial_frames)
            mel_slices.append(mel_range)
            
        return mel_slices
    
    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:,-partial_frames:]
        
        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:,s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)
            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            #embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)
        
        return embed


class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    gin_channels,
    ssl_dim,
    use_spk,
    vq_codebook_size,
    **kwargs):

    super().__init__()
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.gin_channels = gin_channels
    self.ssl_dim = ssl_dim
    self.use_spk = use_spk
    self.vq_codebook_size = vq_codebook_size
    
    #Bottleneck Extractor
    # self.enc_p = Encoder(ssl_dim, inter_channels, hidden_channels, 5, 1, 16, vq_codebook_size=vq_codebook_size)
    #Decoder
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    #Posterior Encoder
    # self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels, vq_codebook_size=None) 
    # #Normalizing Flow
    # self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
    
    #Vector Quantization
    if vq_codebook_size != None:
        self.codebook = modules_v9.VQEmbeddingEMA(vq_codebook_size, hidden_channels)
        
    # if not self.use_spk:
    #   self.enc_spk = SpeakerEncoder(model_hidden_size=gin_channels, model_embedding_size=gin_channels)
    self.spk_emb = None
    
    #V9_1024에서 사용.
    # self.gn = nn.GroupNorm(1, ssl_dim, eps=1e-08)

  def forward(self, c, spec, g=None, mels=None, c_lengths=None, spec_lengths=None):

    if c_lengths == None:
      c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    if spec_lengths == None:
      spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)
      
    # if not self.use_spk:
    #   g = self.enc_spk(mel.transpose(1,2))
    #   g_H = self.enc_spk(mel_H.transpose(1,2))

    # g, g_H = g.unsqueeze(-1), g_H.unsqueeze(-1)
    
    #enc_p : content encoder, enc_q: speaker encoder, flow: normalizing flow
    # emb, emb_quantized, perplexity = self.enc_p(c, c_lengths) # _p: prior  
    # 
    
    # Quantization
    quantized, commitment_loss, perplexity = self.codebook(c)
    if quantized.size(1) != c.size(1):
        quantized = quantized.permute(0, 2, 1)
    # speaker emb
    speaker_emb = c - quantized
    # residual_emb_avg = torch.mean(residual_emb, dim=1, keepdim=True)
    
    # residual_emb_IN = instance_norm(residual_emb, -1)
    # residual_emb_LN = self.gn(residual_emb)
    
    # z = quantized + residual_emb_avg # d: (B, D, T)
    z = quantized # d: (B, D, T)
    
    z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)
    spk_slice = commons.slice_segments(speaker_emb, ids_slice, self.segment_size)
    
    spk_slice_avg = torch.mean(spk_slice, dim=-1, keepdim=True)
    res_slice = spk_slice - spk_slice_avg
    # z_prior_slice, ids_slice = commons.rand_slice_segments(z_prior, spec_lengths, self.segment_size)
    

        
    o = self.dec(z_slice, g=spk_slice_avg, res=res_slice)
    # o = self.dec(z_slice)
    
    
    return o, ids_slice, (commitment_loss, perplexity)
    # return o, ids_slice, (None, None)

  def infer(self, c, g=None, mels=None, c_lengths=None):
    mel = mels
    if c_lengths == None:
      c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    # if not self.use_spk:
    #   g = self.enc_spk.embed_utterance(mel.transpose(1,2))
    # g = g.unsqueeze(-1)

    quantized, commitment_loss, perplexity = self.codebook(c)
    
    # fig = spk_emb_corr(c.permute(0, 2, 1)[:10,:20], quantized[:10,:20])
    fig = None
    if quantized.size(1) != c.size(1):
        quantized = quantized.permute(0, 2, 1)
    # speaker emb
    speaker_emb = c - quantized
    spk_emb_avg = torch.mean(speaker_emb, dim=-1, keepdim=True)
    residual_emb = speaker_emb - spk_emb_avg
    # residual_emb_avg = torch.mean(residual_emb, dim=1, keepdim=True)
    
    # z = quantized + residual_emb_avg
    z = quantized
    
    o = self.dec(z, g=spk_emb_avg, res=residual_emb)
    
    return o, fig

  def convert(self, src_c, tgt_c, c_lengths=None):


    quantized_src, commitment_loss, perplexity = self.codebook(src_c)
    quantized_tgt, commitment_loss, perplexity = self.codebook(tgt_c)
    
    if quantized_src.size(1) != src_c.size(1):
        quantized_src = quantized_src.permute(0, 2, 1)
        quantized_tgt = quantized_tgt.permute(0, 2, 1)
        
    speaker_emb_tgt = tgt_c - quantized_tgt
    speaker_emb_src = src_c - quantized_src
    
    speaker_emb_avg_tgt = torch.mean(speaker_emb_tgt, dim=-1, keepdim=True)
    speaker_emb_avg_src = torch.mean(speaker_emb_src, dim=-1, keepdim=True)
    
    residual_emb_src = speaker_emb_src - speaker_emb_avg_src
    
    z_src = quantized_src
    
    o = self.dec(z_src, g=speaker_emb_avg_tgt, res=residual_emb_src)
    
    return o

def instance_norm(x, dim, epsilon=1e-5):
    mu = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)

    z = (x - mu) / (std + epsilon)
    return z

