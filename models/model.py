from base import BaseModel
from models.modules.fusion import MFB
from models.modules.sincnet.dnn_models import *
from models.dann_functions import ReverseLayerF


# waveform feature extraction branch
class WaveformExtractor(nn.Module):
    def __init__(self,
                 n_in_channels=1,
                 non_linearity='LeakyReLU',
                 latent_size=1024,
                 parameterization='normal',
                 kernel_0_size=251,
                 cn_feature_n=[32, 64, 128, 256],
                 max_pool_kernel=8,
                 kernel_size=7,
                 layernorm_fusion=False):

        super(WaveformExtractor, self).__init__()

        # test_input_waveform = torch.zeros(3, n_in_channels, input_length)
        cn = []
        for ilb, n_out in enumerate(cn_feature_n):
            if ilb == 0:
                if parameterization == 'normal':
                    cn.append(nn.Conv1d(n_in_channels, n_out, kernel_size=kernel_0_size, padding=kernel_0_size // 2))

                elif parameterization == 'sinc':
                    cn.append(
                        SincConv_fast(n_out, kernel_size=kernel_0_size, sample_rate=16000, padding=kernel_0_size // 2))

            else:
                cn.append(nn.Conv1d(cn_feature_n[ilb - 1], n_out, kernel_size=kernel_size, padding=kernel_size // 2))

            cn.append(nn.BatchNorm1d(n_out))
            cn.append(getattr(nn, non_linearity)())
            cn.append(nn.MaxPool1d(kernel_size=max_pool_kernel))

        cn.append(nn.AdaptiveAvgPool1d(latent_size // cn_feature_n[-1]))

        if layernorm_fusion:
            cn.append(nn.Flatten())
            cn.append(nn.LayerNorm(latent_size))

        self.wave_branch = nn.Sequential(*cn)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.wave_branch(x)
        return x


# handcraft feature extraction branch
class HandcraftExtractor(nn.Module):
    def __init__(self,
                 embed_size=64,
                 hidden_size=75,
                 num_layers=2,
                 dropout_p=0.3,
                 batch_first=True,
                 bidirectional=True,
                 latent_size=1024,
                 layernorm_fusion=False):
        super(HandcraftExtractor, self).__init__()

        # test_input_spectrogram = torch.zeros(3, 70, 64)
        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_p,
                          batch_first=batch_first, bidirectional=bidirectional)

        cn = [nn.AdaptiveAvgPool1d(latent_size)]
        if layernorm_fusion:
            cn.append(nn.Flatten())
            cn.append(nn.LayerNorm(latent_size))

        self.post_process = nn.Sequential(*cn)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # x should be: (batch, time_step, input_size)

        # None is for initial hidden state
        r_out, ht = self.rnn(x, None)  # r_out shape (batch, time_step, output_size)

        last_out = r_out[:, -1, :]  # last output

        x_out = self.post_process(last_out)
        return x_out


# spectrogram feature extraction branch
class SpectrogramExtractor(nn.Module):
    def __init__(self,
                 n_in_channels=1,
                 non_linearity='LeakyReLU',
                 latent_size=1024,
                 cn_feature_n=[32, 64, 128, 256],
                 kernel_size=3,
                 max_pool_kernel=(2, 2),
                 layernorm_fusion=False):
        super(SpectrogramExtractor, self).__init__()

        # test_input_spectrogram = torch.zeros(3, n_in_channels, n_bins, n_frames)
        cn = []
        for ilb, n_out in enumerate(cn_feature_n):
            n_in = n_in_channels if ilb == 0 else cn_feature_n[ilb - 1]
            cn.append(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=kernel_size // 2))
            cn.append(nn.BatchNorm2d(n_out))
            cn.append(getattr(nn, non_linearity)())
            cn.append(nn.MaxPool2d(kernel_size=max_pool_kernel))

        cn.append(nn.AdaptiveAvgPool2d((1, latent_size // cn_feature_n[-1])))

        if layernorm_fusion:
            cn.append(nn.Flatten())
            cn.append(nn.LayerNorm(latent_size))

        self.spec_branch = nn.Sequential(*cn)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.spec_branch(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
# AFPINet baseline
class AFPINetMultiModalFusionClassifier(BaseModel):
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)

    def __init__(self,
                 input_length,
                 n_bins,
                 n_frames,
                 num_classes,
                 fusion_method='sum',
                 parameterization='normal',
                 non_linearity='LeakyReLU',
                 layernorm_fusion=False,
                 dropout=0.3,
                 latent_size=1024,
                 fc_layer_n=[512, 256],
                 kernel_0_size=251):
        super().__init__()

        self.fusion_method = fusion_method
        self.layernorm_fusion = layernorm_fusion

        # waveform feature extraction branch
        self.wave_branch = WaveformExtractor(
            n_in_channels=1,
            non_linearity='LeakyReLU',
            latent_size=latent_size,
            parameterization=parameterization,
            kernel_0_size=kernel_0_size,
            cn_feature_n=[32, 64, 128, 256],
            max_pool_kernel=8,
            kernel_size=7,
            layernorm_fusion=layernorm_fusion)

        # handcraft feature extraction branch
        self.hcraft_branch = HandcraftExtractor(
            embed_size=64,
            hidden_size=50,
            num_layers=2,
            dropout_p=0.3,
            batch_first=True,
            bidirectional=True,
            latent_size=latent_size,
            layernorm_fusion=layernorm_fusion)

        # spectrogram feature extraction branch
        self.spec_branch = SpectrogramExtractor(
            n_in_channels=1,
            non_linearity='LeakyReLU',
            latent_size=latent_size,
            cn_feature_n=[32, 64, 128, 256],
            kernel_size=3,
            max_pool_kernel=(2, 2),
            layernorm_fusion=layernorm_fusion)

        # test_input_waveform = torch.zeros(3, input_length)
        # test_input_handcraft = torch.zeros(3, 70, 64)
        # test_input_spectrogram = torch.zeros(3, n_bins, n_frames)

        self.flat = nn.Flatten()
        # fc layer network
        fc = [nn.Dropout(dropout)]

        if fusion_method == 'concat':
            fc_fea_in = latent_size + latent_size + latent_size
        else:
            fc_fea_in = latent_size

        # this need be modified to compatible for three feature branches fusion
        # TODO
        if fusion_method == 'mfb':
            self.mfb = MFB(fc_fea_in, latent_size, MFB_O=fc_fea_in, MFB_K=3)

        if fusion_method == 'sum-attention-noinit':
            self.attention = nn.Sequential(
                nn.Linear(fc_fea_in * 3, fc_fea_in * 4),
                getattr(nn, non_linearity)(),
                nn.Linear(fc_fea_in * 4, fc_fea_in // 2),
                getattr(nn, non_linearity)(),
                nn.Linear(fc_fea_in // 2, 3),
                nn.Softmax(dim=1)
            )
        if fusion_method == 'sum-attention-init':
            self.attention = nn.Sequential(
                nn.Linear(fc_fea_in * 3, fc_fea_in * 4),
                getattr(nn, non_linearity)(),
                nn.Linear(fc_fea_in * 4, fc_fea_in // 2),
                getattr(nn, non_linearity)(),
                nn.Linear(fc_fea_in // 2, 3),
                nn.Softmax(dim=1)
            )
            self.init_weights(self.attention)

        for il, n_out in enumerate(fc_layer_n):
            n_in = fc_fea_in if il == 0 else fc_layer_n[il - 1]

            fc.append(nn.Linear(n_in, n_out))
            fc.append(getattr(nn, non_linearity)())
            fc.append(nn.BatchNorm1d(n_out))
            fc.append(nn.Dropout(dropout))

        fc.append(nn.Linear(fc_layer_n[-1], num_classes))

        self.tail = nn.Sequential(*fc)

    def forward(self, x):

        wave = x['wave']
        hcraft = x['hcraft']
        spec = x['spec']
        wave = self.wave_branch(wave)
        hcraft = self.hcraft_branch(hcraft)
        spec = self.spec_branch(spec)

        if not self.layernorm_fusion:
            w_flat = self.flat(wave)
            h_flat = self.flat(hcraft)
            s_flat = self.flat(spec)
        else:
            w_flat = wave
            h_flat = hcraft
            s_flat = spec

        # Select fusion method
        if self.fusion_method == 'sum':
            combined_features = w_flat.add(h_flat)
            combined_features = combined_features.add(s_flat)

        elif self.fusion_method == 'concat':
            w_flat = F.normalize(w_flat, p=2.0, dim=1, eps=1e-12)
            h_flat = F.normalize(h_flat, p=2.0, dim=1, eps=1e-12)
            s_flat = F.normalize(s_flat, p=2.0, dim=1, eps=1e-12)
            combined_features = torch.cat((w_flat, h_flat, s_flat), dim=1)

        # TODO
        elif self.fusion_method == 'mfb':
            combined_features, _ = self.mfb(w_flat.unsqueeze(1), s_flat.unsqueeze(1))
            combined_features = combined_features.squeeze(1)

        elif self.fusion_method == 'sum-attention-noinit' or self.fusion_method == 'sum-attention-init':
            concat_features = torch.cat((w_flat, h_flat, s_flat), dim=1)
            att = self.attention(concat_features)
            att_1, att_2, att_3 = torch.split(att, 1, dim=1)
            combined_features = (w_flat * att_1).add((h_flat * att_2)).add((s_flat * att_3))

        res = self.tail(combined_features)

        return res


# ----------------------------------------------------------------------------------------------------------------------
# waveform + mel-spectrogram 'Waveforms and spectrograms: enhancing acoustic scene classification
# using multimodal feature fusion'
# modified for ablation experiments
class WaveSpecFusionClassifier(BaseModel):
    @staticmethod
    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)

    def __init__(self,
                 input_length,
                 n_bins,
                 n_frames,
                 num_classes,
                 fusion_method='sum',
                 parameterization='normal',
                 non_linearity='LeakyReLU',
                 layernorm_fusion=False,
                 dropout=0.3,
                 latent_size=1024,
                 fc_layer_n=[512, 256],
                 kernel_0_size=251):
        super().__init__()

        self.fusion_method = fusion_method
        self.layernorm_fusion = layernorm_fusion

        # waveform feature extraction branch
        self.wave_branch = WaveformExtractor(
            n_in_channels=1,
            non_linearity='LeakyReLU',
            latent_size=latent_size,
            parameterization=parameterization,
            kernel_0_size=kernel_0_size,
            cn_feature_n=[32, 64, 128, 256],
            max_pool_kernel=8,
            kernel_size=7,
            layernorm_fusion=layernorm_fusion)

        # handcraft feature extraction branch
        self.hcraft_branch = HandcraftExtractor(
            embed_size=64,
            hidden_size=50,
            num_layers=2,
            dropout_p=0.3,
            batch_first=True,
            bidirectional=True,
            latent_size=latent_size,
            layernorm_fusion=layernorm_fusion)

        # spectrogram feature extraction branch
        self.spec_branch = SpectrogramExtractor(
            n_in_channels=1,
            non_linearity='LeakyReLU',
            latent_size=latent_size,
            cn_feature_n=[32, 64, 128, 256],
            kernel_size=3,
            max_pool_kernel=(2, 2),
            layernorm_fusion=layernorm_fusion)

        # test_input_waveform = torch.zeros(3, input_length)
        # test_input_handcraft = torch.zeros(3, 70, 64)
        # test_input_spectrogram = torch.zeros(3, n_bins, n_frames)

        self.flat = nn.Flatten()
        # fc layer network
        fc = [nn.Dropout(dropout)]

        if fusion_method == 'concat':
            fc_fea_in = latent_size + latent_size + latent_size
        else:
            fc_fea_in = latent_size

        for il, n_out in enumerate(fc_layer_n):
            n_in = fc_fea_in if il == 0 else fc_layer_n[il - 1]

            fc.append(nn.Linear(n_in, n_out))
            fc.append(getattr(nn, non_linearity)())
            fc.append(nn.BatchNorm1d(n_out))
            fc.append(nn.Dropout(dropout))

        fc.append(nn.Linear(fc_layer_n[-1], num_classes))

        self.tail = nn.Sequential(*fc)

    def forward(self, x):
        # waveform + handcraft feature
        # wave = x['wave']
        # hcraft = x['hcraft']
        # x_fea1 = self.wave_branch(wave)
        # x_fea2 = self.hcraft_branch(hcraft)

        # waveform + spectrogram
        wave = x['wave']
        spec = x['spec']
        x_fea1 = self.wave_branch(wave)
        x_fea2 = self.spec_branch(spec)

        # handcraft + spectrogram
        # hcraft = x['hcraft']
        # spec = x['spec']
        # x_fea1 = self.hcraft_branch(hcraft)
        # x_fea2 = self.spec_branch(spec)

        if not self.layernorm_fusion:
            fea1_flat = self.flat(x_fea1)
            fea2_flat = self.flat(x_fea2)
        else:
            fea1_flat = x_fea1
            fea2_flat = x_fea2

        # Select fusion method
        if self.fusion_method == 'sum':
            combined_features = fea1_flat.add(fea2_flat)
            # combined_features = w_flat

        elif self.fusion_method == 'concat':
            fea1_flat = F.normalize(fea1_flat, p=2.0, dim=1, eps=1e-12)
            fea2_flat = F.normalize(fea2_flat, p=2.0, dim=1, eps=1e-12)
            combined_features = torch.cat((fea1_flat, fea2_flat), dim=1)

        res = self.tail(combined_features)

        return res


# for Ablation experiments
class WaveSingleClassifier(BaseModel):
    @staticmethod
    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)

    def __init__(self,
                 input_length,
                 n_bins,
                 n_frames,
                 num_classes,
                 fusion_method='sum',
                 parameterization='normal',
                 non_linearity='LeakyReLU',
                 layernorm_fusion=False,
                 dropout=0.3,
                 latent_size=1024,
                 fc_layer_n=[512, 256],
                 kernel_0_size=251):
        super().__init__()

        self.fusion_method = fusion_method
        self.layernorm_fusion = layernorm_fusion

        # waveform feature extraction branch
        self.wave_branch = WaveformExtractor(
            n_in_channels=1,
            non_linearity='LeakyReLU',
            latent_size=latent_size,
            parameterization=parameterization,
            kernel_0_size=kernel_0_size,
            cn_feature_n=[32, 64, 128, 256],
            max_pool_kernel=8,
            kernel_size=7,
            layernorm_fusion=layernorm_fusion)

        # handcraft feature extraction branch
        self.hcraft_branch = HandcraftExtractor(
            embed_size=64,
            hidden_size=50,
            num_layers=2,
            dropout_p=0.3,
            batch_first=True,
            bidirectional=True,
            latent_size=latent_size,
            layernorm_fusion=layernorm_fusion)

        # spectrogram feature extraction branch
        self.spec_branch = SpectrogramExtractor(
            n_in_channels=1,
            non_linearity='LeakyReLU',
            latent_size=latent_size,
            cn_feature_n=[32, 64, 128, 256],
            kernel_size=3,
            max_pool_kernel=(2, 2),
            layernorm_fusion=layernorm_fusion)

        # test_input_waveform = torch.zeros(3, input_length)
        # test_input_handcraft = torch.zeros(3, 70, 64)
        # test_input_spectrogram = torch.zeros(3, n_bins, n_frames)

        self.flat = nn.Flatten()
        # fc layer network
        fc = [nn.Dropout(dropout)]

        if fusion_method == 'concat':
            fc_fea_in = latent_size + latent_size + latent_size
        else:
            fc_fea_in = latent_size

        for il, n_out in enumerate(fc_layer_n):
            n_in = fc_fea_in if il == 0 else fc_layer_n[il - 1]

            fc.append(nn.Linear(n_in, n_out))
            fc.append(getattr(nn, non_linearity)())
            fc.append(nn.BatchNorm1d(n_out))
            fc.append(nn.Dropout(dropout))

        fc.append(nn.Linear(fc_layer_n[-1], num_classes))

        self.tail = nn.Sequential(*fc)

    def forward(self, x):
        # waveform
        # x_fea = x['wave']
        # x_fea = self.wave_branch(x_fea)

        # handcraft audio feature
        x_fea = x['hcraft']
        x_fea = self.hcraft_branch(x_fea)

        # spectrogram
        # x_fea = x['spec']
        # x_fea = self.spec_branch(x_fea)

        if not self.layernorm_fusion:
            x_flat = self.flat(x_fea)
        else:
            x_flat = x_fea

        combined_features = x_flat

        res = self.tail(combined_features)

        return res


# ----------------------------------------------------------------------------------------------------------------------
# re-implementation of 'Person Identification by Footstep Sound Using Convolutional Neural Networks'
class CnnClassifier(BaseModel):
    def __init__(self,
                 input_length,
                 n_bins,
                 n_frames,
                 num_classes,
                 fusion_method='sum',
                 parameterization='normal',
                 non_linearity='LeakyReLU'):
        super().__init__()
        # test_input = torch.zeros(3, 3, 64, 70)  # batch_size, channels, height, width
        self.cn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 16, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )

        self.flat = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(16 * 17 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 13),
        )

    def forward(self, x):
        spec = x['spec']
        spec = torch.unsqueeze(spec, dim=1)
        spec_x = torch.cat([spec, spec, spec], dim=1)
        spec_fe = self.cn(spec_x)

        spec_fe = self.flat(spec_fe)
        res = self.fc(spec_fe)

        return res


# ----------------------------------------------------------------------------------------------------------------------
# AFPI-Net combined with Domain adversarial neural network (DANN).
class AFPINet_DANN(BaseModel):
    @staticmethod
    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)

    def __init__(self,
                 input_length,
                 n_bins,
                 n_frames,
                 num_classes,
                 fusion_method='sum',
                 parameterization='normal',
                 non_linearity='LeakyReLU',
                 layernorm_fusion=False,
                 dropout=0.3,
                 latent_size=1024,
                 fc_layer_n=[512, 256],
                 kernel_0_size=251):
        super().__init__()

        self.fusion_method = fusion_method
        self.layernorm_fusion = layernorm_fusion

        # waveform feature extraction branch
        self.wave_branch = WaveformExtractor(
            n_in_channels=1,
            non_linearity='LeakyReLU',
            latent_size=latent_size,
            parameterization=parameterization,
            kernel_0_size=kernel_0_size,
            cn_feature_n=[32, 64, 128, 256],
            # max_pool_kernel=8,
            max_pool_kernel=6,  # adjust for AFPID_FE1 dataset
            kernel_size=7,
            layernorm_fusion=layernorm_fusion)

        # handcraft feature extraction branch
        self.hcraft_branch = HandcraftExtractor(
            embed_size=64,
            hidden_size=50,
            num_layers=2,
            dropout_p=0.3,
            batch_first=True,
            bidirectional=True,
            latent_size=latent_size,
            layernorm_fusion=layernorm_fusion)

        # spectrogram feature extraction branch
        self.spec_branch = SpectrogramExtractor(
            n_in_channels=1,
            non_linearity='LeakyReLU',
            latent_size=latent_size,
            cn_feature_n=[32, 64, 128, 256],
            kernel_size=3,
            max_pool_kernel=(2, 2),
            layernorm_fusion=layernorm_fusion)

        # test_input_waveform = torch.zeros(3, input_length)
        # test_input_handcraft = torch.zeros(3, 70, 64)
        # test_input_spectrogram = torch.zeros(3, n_bins, n_frames)

        self.flat = nn.Flatten()
        # fc layer network
        fc = [nn.Dropout(dropout)]

        if fusion_method == 'concat':
            fc_fea_in = latent_size + latent_size + latent_size
        else:
            fc_fea_in = latent_size

        # this need be modified to compatible for three feature branchs fusion
        # TODO
        if fusion_method == 'mfb':
            self.mfb = MFB(fc_fea_in, latent_size, MFB_O=fc_fea_in, MFB_K=3)

        if fusion_method == 'sum-attention-noinit':
            self.attention = nn.Sequential(
                nn.Linear(fc_fea_in * 3, fc_fea_in * 4),
                getattr(nn, non_linearity)(),
                nn.Linear(fc_fea_in * 4, fc_fea_in // 2),
                getattr(nn, non_linearity)(),
                nn.Linear(fc_fea_in // 2, 2),
                nn.Softmax(dim=1)
            )
        if fusion_method == 'sum-attention-init':
            self.attention = nn.Sequential(
                nn.Linear(fc_fea_in * 3, fc_fea_in * 4),
                getattr(nn, non_linearity)(),
                nn.Linear(fc_fea_in * 4, fc_fea_in // 2),
                getattr(nn, non_linearity)(),
                nn.Linear(fc_fea_in // 2, 2),
                nn.Softmax(dim=1)
            )
            self.init_weights(self.attention)

        for il, n_out in enumerate(fc_layer_n):
            n_in = fc_fea_in if il == 0 else fc_layer_n[il - 1]

            fc.append(nn.Linear(n_in, n_out))
            fc.append(getattr(nn, non_linearity)())
            fc.append(nn.BatchNorm1d(n_out))
            fc.append(nn.Dropout(dropout))

        fc.append(nn.Linear(fc_layer_n[-1], num_classes))
        fc.append(nn.LogSoftmax(dim=1))

        self.class_classifier = nn.Sequential(*fc)

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(1024, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x, alpha):

        wave = x['wave']
        hcraft = x['hcraft']
        spec = x['spec']
        wave = self.wave_branch(wave)
        hcraft = self.hcraft_branch(hcraft)
        spec = self.spec_branch(spec)

        if not self.layernorm_fusion:
            w_flat = self.flat(wave)
            h_flat = self.flat(hcraft)
            s_flat = self.flat(spec)
        else:
            w_flat = wave
            h_flat = hcraft
            s_flat = spec

        # Select fusion method
        if self.fusion_method == 'sum':
            combined_features = w_flat.add(h_flat)
            combined_features = combined_features.add(s_flat)
            # combined_features = w_flat

        elif self.fusion_method == 'concat':
            w_flat = F.normalize(w_flat, p=2.0, dim=1, eps=1e-12)
            h_flat = F.normalize(h_flat, p=2.0, dim=1, eps=1e-12)
            s_flat = F.normalize(s_flat, p=2.0, dim=1, eps=1e-12)
            combined_features = torch.cat((w_flat, h_flat, s_flat), dim=1)

        # TODO
        elif self.fusion_method == 'mfb':
            combined_features, _ = self.mfb(w_flat.unsqueeze(1), s_flat.unsqueeze(1))
            combined_features = combined_features.squeeze(1)

        elif self.fusion_method == 'sum-attention-noinit' or self.fusion_method == 'sum-attention-init':
            concat_features = torch.cat((w_flat, h_flat, s_flat), dim=1)
            att = self.attention(concat_features)
            att_1, att_2, att_3 = torch.split(att, 1, dim=1)
            combined_features = (w_flat * att_1).add((h_flat * att_2)).add((s_flat * att_3))

        # features generated from the AFPI-Net feature extraction branch: combined_features
        # res = self.tail(combined_features)

        reverse_feature = ReverseLayerF.apply(combined_features, alpha)
        # reverse_feature = ReverseLayerF.apply(combined_features, 1)
        class_output = self.class_classifier(combined_features)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

