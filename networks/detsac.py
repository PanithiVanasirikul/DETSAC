import torch
from torch import nn
import torch.nn.functional as F

class DETSAC(nn.Module):

    def __init__(self, input_dim, output_dim, encoder_blocks=3, decoder_blocks=3, nheads=4, dim_feedforward=128, batch_norm=True):

        super(DETSAC, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.p_in = nn.Conv1d(self.input_dim, 128, 1, 1, 0)

        self.backbone = []

        self.batch_norm = batch_norm

        for i in range(0, encoder_blocks):
            if batch_norm:
                self.backbone.append((
                    nn.Conv1d(128, 128, 1, 1, 0),
                    nn.BatchNorm1d(128),
                    nn.Conv1d(128, 128, 1, 1, 0),
                    nn.BatchNorm1d(128),
                ))
            else:
                self.backbone.append((
                    nn.Conv1d(128, 128, 1, 1, 0),
                    nn.Conv1d(128, 128, 1, 1, 0),
                ))

        for i, r in enumerate(self.backbone):
            super(DETSAC, self).add_module(str(i) + 's0', r[0])
            super(DETSAC, self).add_module(str(i) + 's1', r[1])
            if batch_norm:
                super(DETSAC, self).add_module(str(i) + 's2', r[2])
                super(DETSAC, self).add_module(str(i) + 's3', r[3])
        
        hidden_dim = 128
        
        self.object_queries = nn.Parameter(torch.rand(output_dim, hidden_dim))
        nn.init.normal_(self.object_queries, mean=0, std=0.01)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            batch_first=True,
            dim_feedforward=dim_feedforward
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_blocks
        )
        
        self.param_hnet1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim+1)
            )

        nn.init.kaiming_normal_(self.param_hnet1[-1].weight, a=0.0, nonlinearity='relu', mode='fan_out')
        self.param_hnet1[-1].weight.data *= 1e-1
        
        self.latent_code_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, inputs, mask):
        '''
        Forward pass.

        inputs -- 3D data tensor (BxNxC)
        '''
        inputs_ = torch.transpose(inputs, 1, 2)
        x = inputs_[:, 0:self.input_dim]
        x = F.relu(self.p_in(x))

        for r in self.backbone:
                    res = x
                    if self.batch_norm:
                        x = F.relu(r[1](F.instance_norm(r[0](x))))
                        x = F.relu(r[3](F.instance_norm(r[2](x))))
                    else:
                        x = F.relu(F.instance_norm(r[0](x)))
                        x = F.relu(F.instance_norm(r[1](x)))
                    x = x + res
        
        object_queries_output = self.transformer(self.object_queries[None].repeat(x.shape[0], 1, 1), x.transpose(1, 2), memory_key_padding_mask=~(mask.to(torch.bool)))
        
        B, O, D_query = object_queries_output.shape

        hnet_out1 = self.param_hnet1(object_queries_output.view(B*O, D_query))
        p_out = hnet_out1.view(B, O, D_query+1)
        dynamic_linear_weights, dynamic_linear_bias = p_out[..., :-1], p_out[..., -1:]
        point_output = torch.bmm(dynamic_linear_weights, x) + dynamic_linear_bias
        
        latent_code_classifier_output = self.latent_code_mlp(object_queries_output.view(-1, D_query)).reshape(B, O, 1)
        return latent_code_classifier_output, point_output
