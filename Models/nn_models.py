
import torch
import torch.nn as nn
import numpy as np
import einops
from einops import rearrange, repeat
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with varied size tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binary mask
    output = x.div(keep_prob) * random_tensor
    return output

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads,dropout_attn=0.1,dropout_proj=0.1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        self.dropout_attn = nn.Dropout(dropout_attn)
        self.dropout_proj = nn.Dropout(dropout_proj)
        self.values = nn.Linear(embed_size, embed_size,bias=False) # bias = False?
        self.keys = nn.Linear(embed_size, embed_size,bias=False)
        self.queries = nn.Linear(embed_size, embed_size,bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask,return_attention=False):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)



        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])


        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        attention = self.dropout_attn(attention)
        # attention shape: (N, heads, query_len, key_len/value)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        if return_attention:
            return self.dropout_proj(out), attention

        else:
            return self.dropout_proj(out)



class ProbeLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ProbeLayer, self).__init__()
        # This is a simple linear layer that acts as a probe.
        self.probe = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Apply the probe to the input tensor x
        # x shape: (batch_size, seq_length, input_dim)
        logits = self.probe(x)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout_attn, dropout_proj, dropout_embed, drop_path_rate, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads,dropout_attn,dropout_proj)
        self.prenorm1 = nn.LayerNorm(embed_size)
        self.prenorm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Dropout(dropout_embed),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.drop_path_rate = drop_path_rate
        #self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask,return_attention=False):
        
        value = self.prenorm2(value)
        key = self.prenorm2(key)
        query = self.prenorm1(query)
        if return_attention:
            attention,SA = self.attention(value, key, query, mask,return_attention=return_attention)
        else: 
            attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        #x = self.norm1(drop_path(attention, self.drop_path_rate, self.training) + query)
        x = drop_path(attention, self.drop_path_rate, self.training) + query
        
        forward = self.feed_forward(x)

        #out = self.norm2(drop_path(forward, self.drop_path_rate, self.training) + x)
        out = drop_path(forward, self.drop_path_rate, self.training) + x
        if return_attention:
            return out,SA
        else:
            return out


class LearnablePELayer(nn.Module):

    def __init__(self, shape, init_mode='learnable', **kwargs):
        super(LearnablePELayer, self).__init__(**kwargs)
        self.init_mode = init_mode
        if init_mode == 'learnable':
            self.pe = nn.Parameter(torch.randn(shape) * 0.02)
            #print(self.pe.shape)
        elif init_mode == 'sincos':
            self.pe = nn.Parameter(self.get_2d_sincos_pos_embed(shape[-1], int(shape[0]**0.5), cls_token=False), requires_grad=True)
        else:
            raise ValueError("Invalid initialization mode")

    def _init_weights(self):
        """
        Initialize the weights of the LearnablePELayer.
        """
        # Apply custom weights initialization for the learnable positional encoding
        if self.init_mode == 'learnable':
            torch.nn.init.normal_(self.pe, std=.02)
        # If there are other layers in this class that need initialization, 
        # they should be handled here.

    def initialize_weights(self):
        """
        Public method to initialize weights.
        """
        if self.init_mode == 'learnable':
            torch.nn.init.normal_(self.pe, std=.02)

        #self.apply(self._init_weights)

    def forward(self, inputs):
        if self.init_mode == 'learnable':
            b = inputs.size(0)
            batched_pe = self.pe #einops.repeat(self.pe, '... -> b ...', b=b)

        elif self.init_mode == 'sincos':
            batched_pe = self.pe.unsqueeze(0).expand(inputs.size(0), -1, -1)
        return batched_pe

    #@staticmethod
    def get_1d_sincos_pos_embed_from_grid(self,embed_dim, pos):
        # Implementation of 1D sin-cos positional embedding from grid
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.)
        omega = 1. / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb
    
    #@staticmethod
    def get_2d_sincos_pos_embed(self,embed_dim, grid_size, cls_token=False):
        # Implementation of 2D sin-cos positional embedding
        pos = np.arange(grid_size, dtype=np.float32)
        pos_embed = self.get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return torch.from_numpy(pos_embed).float()



class ConcatPoolBlock(nn.Module):
    def __init__(self,num_features,pool_dim = 1):
        super(ConcatPoolBlock, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(pool_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(2*num_features,num_features),
            nn.GELU()
        )

    def forward(self, x):
        # x: input tensor of shape (batch_size, num_features, N)
        # input should be shape (bacth,num_groups,group_size,num_features)
        
        b,g,n,f = x.shape
        # Local Max Pooling
        # Reshape x for group-wise pooling using einops
        x = rearrange(x, 'b g n f -> (b g) f n') # shape: (batch_size, num_features, group_size)
        lm_pool = self.maxpool(x).squeeze(-1)  # shape: (batch_size*num_groups, num_features)
        y = rearrange(lm_pool, '(b g) f -> b g f', g=g)
        return y


class ConcatPoolBlock_org(nn.Module):
    def __init__(self,num_features,pool_dim = 1):
        super(ConcatPoolBlock, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(pool_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(2*num_features,num_features),
            nn.GELU()
        )

    def forward(self, x):
        # x: input tensor of shape (batch_size, num_features, N)
        # input should be shape (bacth,num_groups,group_size,num_features)
        
        b,g,n,f = x.shape
        # Local Max Pooling
        # Reshape x for group-wise pooling using einops
        x = rearrange(x, 'b g n f -> (b g) f n') # shape: (batch_size, num_features, group_size)
        lm_pool = self.maxpool(x).squeeze(-1)  # shape: (batch_size*num_groups, num_features)
    

        # Concatenate each local max pool feature with global feature
        y = torch.cat([x, lm_pool.unsqueeze(-1).expand_as(x)], dim=1) # shape: (batch_size*num_groups, num_features*2,group_size)

        # Reshape back to the original format using einops ## rearrange back
        y = rearrange(y, '(b g) f n -> b g n f', g=g) # shape: (batch_size, num_groups, 2*num_features)
        
        y = self.fc(y)

        return y


class GroupedPoolBlock(nn.Module):
    def __init__(self,pool_dim = 1):
        super(GroupedPoolBlock, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(pool_dim)

    def forward(self, x):
        # x : input tensor of shape (bacth,num_groups,group_size,num_features)
        # Global Max Pooling
        b,g,n,f = x.shape
        x = rearrange(x, 'b g n f -> b f (g n)')
        gmpool = self.maxpool(x).squeeze(-1)  # shape: (batch_size, num_features)
        x = rearrange(x, 'b f (g n) -> (b g) f n', g=g) # shape: (batch_size, num_features, group_size)
        # Local Max Pooling
        lm_pool = self.maxpool(x).squeeze(-1)  # shape: (batch_size*num_groups, num_features)
        lm_pool = rearrange(lm_pool, '(b g) f -> b f g', g=g)  # shape: (batch_size, num_features, num_groups)

        # Concatenate each local max pool feature with global feature
        y = torch.cat([lm_pool, gmpool.unsqueeze(-1).expand_as(lm_pool)], dim=1)

        # Reshape back to the original format using einops ## rearrange back
        y = rearrange(y, 'b out_f g -> b g out_f') # shape: (batch_size, num_groups, 2*num_features)

        return y

class TemPoseIII(nn.Module):
    def __init__(
            self,
            channels,
            poses_numbers,
            dim,
            dim_joints,
            depth_temp,
            depth_spat,
            heads,
            heads_spat,
            forward_expansion,
            dropout_attn,
            dropout_proj,
            dropout_embed,
            drop_path_rate,
            time_steps,
            num_people=2,
            playerDB = 27,

        ):
        super().__init__()
        
        self.heads = heads
        self.time_sequence = time_steps
        self.num_key = poses_numbers
        self.people=num_people

    
        self.to_patch_embedding = nn.Linear(channels, dim_joints)
   

        self.spat_embedding = LearnablePELayer((1, self.num_key, dim_joints), init_mode='learnable')
        inc_drop_rates = np.linspace(0, drop_path_rate, depth_spat)
        self.Transformer_spat = nn.ModuleList(
            [
                TransformerBlock(
                    dim_joints,
                    heads_spat,
                    dropout_attn,
                    dropout_proj,
                    dropout_embed,
                    inc_drop_rates[i],
                    forward_expansion=forward_expansion,
                )
                for i in range(depth_spat)
            ]
        )

        self.Transformer_spat_cross = nn.ModuleList(
            [
                TransformerBlock(
                    dim_joints,
                    heads_spat,
                    dropout_attn,
                    dropout_proj,
                    dropout_embed,
                    inc_drop_rates[i],
                    forward_expansion=forward_expansion,
                )
                for i in range(depth_spat)
            ]
        )
        ## player table
        self.numplayer_DB = playerDB # 27 when shuttleset #35 when it is shuttleset22
        self.player_emb = nn.Embedding(self.numplayer_DB, 1)

        ### GPB

        self.GPB = GroupedPoolBlock()
        self.spat_to_temp = nn.Linear(dim_joints*3,dim) 
    

        self.temporal_embedding = LearnablePELayer((1, time_steps, dim), init_mode='learnable')
        inc_drop_rates = np.linspace(0, drop_path_rate, depth_temp)
        self.Transformer_temp = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    dropout_attn,
                    dropout_proj,
                    dropout_embed,
                    inc_drop_rates[i],
                    forward_expansion=forward_expansion,
                )
                for i in range(depth_temp)
            ]
        )
        self.Transformer_temp_cross = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    dropout_attn,
                    dropout_proj,
                    dropout_embed,
                    inc_drop_rates[i],
                    forward_expansion=forward_expansion,
                )
                for i in range(depth_temp)
            ]
        )



        
        self.temp_to_dec = nn.Linear(2*dim,dim)
        
      
        
        ### TCN block
        self.num_channels = [dim_joints//2,dim_joints]
        self.kernel_size = 5
        input_size = 4

        # define temporal convolutional layers
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
           dilation_size = (2 * i) + 1
           in_channels = input_size if i == 0 else self.num_channels[i-1]
           out_channels = self.num_channels[i]
           padding = (self.kernel_size - 1) * dilation_size // 2
           layers += [nn.Conv1d(in_channels, out_channels, self.kernel_size, dilation=dilation_size, padding=padding),
                      nn.BatchNorm1d(out_channels),
                      nn.GELU(),
                      nn.Dropout(dropout_embed)]
        self.tcn1 = nn.Sequential(*layers)

        # define temporal convolutional layers
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
           dilation_size = (2 * i) + 1
           in_channels = 8 if i == 0 else self.num_channels[i-1]
           out_channels = self.num_channels[i]
           padding = (self.kernel_size - 1) * dilation_size // 2
           layers += [nn.Conv1d(in_channels, out_channels, self.kernel_size, dilation=dilation_size, padding=padding),
                      nn.BatchNorm1d(out_channels),
                      nn.GELU(),
                      nn.Dropout(dropout_embed)]
        self.tcn2 = nn.Sequential(*layers)


        ## 
        self.lastpool = ConcatPoolBlock(dim)
        self.proj_enc_out = nn.Linear(2*dim,dim)

        self.initialize_weights()
        
    def initialize_weights(self):

        self.spat_embedding.initialize_weights()
        self.temporal_embedding.initialize_weights()



        print('weights initialized')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
       # elif isinstance(m, LearnablePELayer):
       #     m.initialize_weights()


    def forward(self, x,ID,mask,return_attention=False):#,t_pad,n_pad
        
        
        N, ral_length, pep, seq_length,_ = x.shape
        ##add ID infomation 
        x = rearrange(x,'N s p t (j d) -> N s p t j d',j=self.num_key)
        ID = ID.unsqueeze(-2).unsqueeze(-2)  # Add dimensions for T and J at the end, A shape becomes (B, S, N, 1, 1, 2)
        ID = ID.expand(-1, -1, -1, seq_length, self.num_key, -1)  # Expand A to match the size of B in T and J dimensions
        ID_1 = self.player_emb(ID[:,:,:,:,:,0]) ## player descriptions instead... 
        # Last: Concatenate A_expanded and B along the last dimension

        x = torch.cat((x, ID_1,ID[:,:,:,:,:,1].unsqueeze(-1)), dim=-1)  # Concatenate along the last dimension
        ##
        x_pos = rearrange(x,'N s p t j d -> (N s p) d t j',j=self.num_key)[:,:,:,-1]


        x_active = rearrange(x,'N s p t j d -> (N s t) p j d',j=self.num_key)[:,:,:,-1].unsqueeze(-1)
        x = self.to_patch_embedding(x)


        x_cross  = rearrange(x,'N s p t j d -> (N s t) p j d',j=self.num_key) ## can be improved (ALOT)
        x = rearrange(x,'N s p t j d -> (N s p t) j d',j=self.num_key)
    
        x_cross= x_cross[:,0]*x_active[:,0,:,0].unsqueeze(-1) + x_cross[:,1]*x_active[:,1,:,0].unsqueeze(-1)

        
        ##TCN
        x_pos_mu = rearrange(self.tcn2(rearrange(x_pos,'(N s p) d t -> (N s) (p d) t',p=pep,s=ral_length)),'N d t-> N t d').unsqueeze(1)
        x_pos = rearrange(self.tcn1(x_pos),'(N s p) d t-> (N s) p t d',N=N,p=pep,s=ral_length)
        x_pos = torch.cat((x_pos_mu,x_pos),dim=1)


        x = x + self.spat_embedding(x)


        #x = self.dropout(x)
        
        if return_attention:
            for (layer1,layer2) in zip(self.Transformer_spat,self.Transformer_spat_cross):
                x,KA = layer1(x, x, x, mask=None,return_attention=return_attention)
                x_react = rearrange(x, '(b s n t) j d -> (b s t) n j d', b=N, s=ral_length, n=pep, t=seq_length)
                x_react = x_react[:,0]*(1-x_active[:,0,:,0]).unsqueeze(-1) + x_react[:,1]*(1-x_active[:,1,:,0]).unsqueeze(-1)
                x_cross,_ = layer2(x_cross, x_cross, x_react, mask=None,return_attention=return_attention)
        else:
            for (layer1,layer2) in zip(self.Transformer_spat,self.Transformer_spat_cross):
                x_react = rearrange(x, '(b s n t) j d -> (b s t) n j d', b=N, s=ral_length, n=pep, t=seq_length)
                x = layer1(x, x, x, mask=None)
                #x_react = rearrange(x, '(b s n t) j d -> (b s t) n j d', b=N, s=ral_length, n=pep, t=seq_length) potentially could be used after 1st attention
                x_react = x_react[:,0]*(1-x_active[:,0,:,0]).unsqueeze(-1) + x_react[:,1]*(1-x_active[:,1,:,0]).unsqueeze(-1)
                x_cross = layer2(x_cross, x_cross, x_react, mask=None)
        

        #x = x.mean(dim=1)
        x1 = rearrange(x, '(b s n t) j d -> (b s n) t j d', b=N, s=ral_length, n=pep, t=seq_length)                              
        x = torch.cat((rearrange(x, '(b s n t) j d -> (b s t) n j d', b=N, s=ral_length, n=pep, t=seq_length),x_cross.unsqueeze(1)),dim=1)
        x = rearrange(x, '(b s t) n j d -> (b s) t (n j) d', b=N, s=ral_length, n=pep+1, t=seq_length)
        x = self.GPB(x).unsqueeze(1)    
        x1 = self.GPB(x1) 
        x1 = rearrange(x1, '(b s n) t d -> (b s) n t d', b=N, s=ral_length, n=pep, t=seq_length)  
        x = torch.cat((x,x1),dim=1)

        x = torch.cat((x,x_pos),dim=-1)                   
        x = rearrange(x, '(b s) n t d -> (b s n) t d', b=N, s=ral_length, n=pep+1, t=seq_length)

        x = self.spat_to_temp(x)
        x1 = rearrange(x, '(b s n) t d -> (b s) n t d', b=N, s=ral_length, n=pep+1, t=seq_length)[:,1:]

        x_active = rearrange(x_active[:,:,-1,:],'(b s t) n d -> (b s) n t d',b=N, s=ral_length, n=pep, t=seq_length)
        

        x_cross= x1[:,0]*x_active[:,0,:,0].unsqueeze(-1) + x1[:,1]*x_active[:,1,:,0].unsqueeze(-1)
    
        x = x + self.temporal_embedding(x)

        if return_attention:
            for (layer1,layer2) in zip(self.Transformer_temp,self.Transformer_temp_cross):
                x_react = rearrange(x, '(b s n) t d -> (b s) n t d', b=N, s=ral_length, n=pep, t=seq_length)[:,1:]
                x,TA = layer1(x, x, x, mask,return_attention=return_attention)
                x_react = x_react[:,0]*(1-x_active[:,0,:,0]).unsqueeze(-1) + x_react[:,1]*(1-x_active[:,1,:,0]).unsqueeze(-1)
                x,_ = layer2(x_cross, x_cross, x_react, mask,return_attention=return_attention)
        else:    
            for (layer1,layer2) in zip(self.Transformer_temp,self.Transformer_temp_cross):

                x_react = rearrange(x, '(b s n) t d -> (b s) n t d', b=N, s=ral_length, n=pep+1, t=seq_length)[:,1:]
                x = layer1(x, x, x, mask)
                x_react = x_react[:,0]*(1-x_active[:,0,:,0]).unsqueeze(-1) + x_react[:,1]*(1-x_active[:,1,:,0]).unsqueeze(-1)
                x_cross = layer2(x_cross, x_cross, x_react, rearrange(mask, '(b s p) y x t -> (b s) p y x t',b=N, s=ral_length, p=pep+1)[:,0])
                

        x = torch.cat((rearrange(x, '(b s n) t d -> (b s) n t d', b=N, s=ral_length, n=pep+1, t=seq_length),x_cross.unsqueeze(1)),dim=1)
        x_id = x[:,1:-1]

        x = self.GPB(rearrange(x, 'B n t d -> B (n t) d', B=N*ral_length, n=pep+2, t=seq_length).unsqueeze(1)) # shape (batch*seqlen,1,2*dim)  
    
        x_id = self.lastpool(x_id)
     

        out = self.proj_enc_out(x)

        out = torch.cat((out,x_id),dim=1) 
        out = rearrange(out,'(N s) p d -> N s p d',N=N,s=ral_length)   
        if return_attention:
            return out,KA,TA        
        else:        
            return out
    
class TemPoseII(nn.Module):
    def __init__(
            self,
            channels,
            poses_numbers,
            #num_classes,
            dim,
            dim_joints,
            #kernel_size=5,
            depth_temp,
            depth_spat,
            heads,
            heads_spat,
            forward_expansion,
            #mlp_dim=512,
            #pool = 'cls',
            dropout_attn,
            dropout_proj,
            dropout_embed,
            drop_path_rate,
            time_steps,
            num_people=2,
            #emb_dropout = 0.3,
            #dataset= 'OL'
        ):
        super().__init__()
        
        self.heads = heads
        self.time_sequence = time_steps
        self.num_key = poses_numbers

        self.people=num_people

        

        self.to_patch_embedding = nn.Linear(channels, dim_joints)

        self.spat_embedding = LearnablePELayer((1, self.num_key, dim_joints), init_mode='learnable')
        inc_drop_rates = np.linspace(0, drop_path_rate, depth_spat)
        self.Transformer_spat = nn.ModuleList(
            [
                TransformerBlock(
                    dim_joints,
                    heads_spat,
                    dropout_attn,
                    dropout_proj,
                    dropout_embed,
                    inc_drop_rates[i],
                    forward_expansion=forward_expansion,
                )
                for i in range(depth_spat)
            ]
        )
        ## player table
        self.numplayer_DB = 35
        self.player_emb = nn.Embedding(self.numplayer_DB, 1)

        ### GPB

        self.GPB = GroupedPoolBlock()
        self.spat_to_temp = nn.Linear(dim_joints*3,dim) 

        self.temporal_embedding = LearnablePELayer((1, time_steps, dim), init_mode='learnable')#self.temporal_embedding = LearnablePELayer((1, self.people, time_steps, dim), init_mode='learnable')
        

        inc_drop_rates = np.linspace(0, drop_path_rate, depth_temp)
        self.Transformer_temp = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    dropout_attn,
                    dropout_proj,
                    dropout_embed,
                    inc_drop_rates[i],
                    forward_expansion=forward_expansion,
                )
                for i in range(depth_temp)
            ]
        )


        
        self.temp_to_dec = nn.Linear(2*dim,dim)
        
        
        ### TCN block
        self.num_channels = [dim_joints//2,dim_joints]
        self.kernel_size = 5
        input_size = 4

        # define temporal convolutional layers
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
           dilation_size = (2 * i) + 1
           in_channels = input_size if i == 0 else self.num_channels[i-1]
           out_channels = self.num_channels[i]
           padding = (self.kernel_size - 1) * dilation_size // 2
           layers += [nn.Conv1d(in_channels, out_channels, self.kernel_size, dilation=dilation_size, padding=padding),
                      nn.BatchNorm1d(out_channels),
                      nn.GELU(),
                      nn.Dropout(dropout_embed)]
        self.tcn1 = nn.Sequential(*layers)

        # define temporal convolutional layers
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
           dilation_size = (2 * i) + 1
           in_channels = 8 if i == 0 else self.num_channels[i-1]
           out_channels = self.num_channels[i]
           padding = (self.kernel_size - 1) * dilation_size // 2
           layers += [nn.Conv1d(in_channels, out_channels, self.kernel_size, dilation=dilation_size, padding=padding),
                      nn.BatchNorm1d(out_channels),
                      nn.GELU(),
                      nn.Dropout(dropout_embed)]
        self.tcn2 = nn.Sequential(*layers)


        ## 
        self.lastpool = ConcatPoolBlock(dim)
        self.proj_enc_out = nn.Linear(2*dim,dim)


        
        
        self.initialize_weights()
        
    def initialize_weights(self):

        self.spat_embedding.initialize_weights()
        self.temporal_embedding.initialize_weights()

        print('weights initialized')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
       # elif isinstance(m, LearnablePELayer):
       #     m.initialize_weights()


    def forward(self, x,ID,mask,return_attention=False):#,t_pad,n_pad
        
        
        N, ral_length, pep, seq_length,_ = x.shape
        ##add ID infomation 
        x = rearrange(x,'N s p t (j d) -> N s p t j d',j=self.num_key)
        ID = ID.unsqueeze(-2).unsqueeze(-2)  # Add dimensions for T and J at the end, A shape becomes (B, S, N, 1, 1, 2)
        ID = ID.expand(-1, -1, -1, seq_length, self.num_key, -1)  # Expand A to match the size of B in T and J dimensions
        ID_1 = self.player_emb(ID[:,:,:,:,:,0])
        # Last: Concatenate A_expanded and B along the last dimension

        x = torch.cat((x, ID_1,ID[:,:,:,:,:,1].unsqueeze(-1)), dim=-1)  # Concatenate along the last dimension
        ##
        x_pos = rearrange(x,'N s p t j d -> (N s p) d t j',j=self.num_key)[:,:,:,-1]
    
        x = rearrange(x,'N s p t j d -> (N s p t) j d',j=self.num_key)
        
        ##TCN
        x_pos_mu = rearrange(self.tcn2(rearrange(x_pos,'(N s p) d t -> (N s) (p d) t',p=pep,s=ral_length)),'N d t-> N t d').unsqueeze(1)
        x_pos = rearrange(self.tcn1(x_pos),'(N s p) d t-> (N s) p t d',N=N,p=pep,s=ral_length)
        x_pos = torch.cat((x_pos_mu,x_pos),dim=1)

        x = self.to_patch_embedding(x)


        x = x + self.spat_embedding(x)


        #x = self.dropout(x)
        
        if return_attention:
            for layer in self.Transformer_spat: 
                x,KA = layer(x, x, x, mask=None,return_attention=return_attention)
        else:
            for layer in self.Transformer_spat: 
                x = layer(x, x, x, mask=None)
            
        #x = x.mean(dim=1)
                                      
        #x = rearrange(x, '(b s n t) j d -> (b s) (n t) j d', b=N, s=ral_length, n=pep, t=seq_length)
        x = rearrange(x, '(b s n t) j d -> (b s) t (n j) d', b=N, s=ral_length, n=pep, t=seq_length)
        x1 = rearrange(x, '(b s) t (n j) d -> (b s n) t j d', b=N, s=ral_length, n=pep, t=seq_length)
        x = self.GPB(x).unsqueeze(1)    
        x1 = self.GPB(x1) 
        x1 = rearrange(x1, '(b s n) t d -> (b s) n t d', b=N, s=ral_length, n=pep, t=seq_length)  
        x = torch.cat((x,x1),dim=1)

        x = torch.cat((x,x_pos),dim=-1)                   
        x = rearrange(x, '(b s) n t d -> (b s n) t d', b=N, s=ral_length, n=pep+1, t=seq_length)

        x = self.spat_to_temp(x)
    
        x = x + self.temporal_embedding(x)

        if return_attention:
            for layer in self.Transformer_temp:
                x,TA = layer(x, x, x, mask,return_attention=return_attention)
        else:    
            for layer in self.Transformer_temp:
                x = layer(x, x, x, mask)

        x = rearrange(x, '(b s n) t d -> (b s) n t d', b=N, s=ral_length, n=pep+1, t=seq_length)
        x_id = x[:,1:]
        x = self.GPB(rearrange(x, 'B n t d -> B (n t) d', B=N*ral_length, n=pep+1, t=seq_length).unsqueeze(1)) # shape (batch*seqlen,1,2*dim)  
    
        x_id = self.lastpool(x_id)
     
        out = self.proj_enc_out(x)

        out = torch.cat((out,x_id),dim=1) 
        out = rearrange(out,'(N s) p d -> N s p d',N=N,s=ral_length)   
        if return_attention:
            return out,KA,TA        
        else:        
            return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout_attn, dropout_proj, dropout_embed, drop_path_rate=0.15):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads, dropout_attn, dropout_proj)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout_attn, dropout_proj, dropout_embed, drop_path_rate, forward_expansion)
        self.dropout = nn.Dropout(dropout_embed)
        self.drop_path_rate = drop_path_rate

    def forward(self, x, value, key, src_mask, trg_mask,return_attention=False):
        if return_attention:     
            x = self.norm(x)                   
            attention,SA = self.attention(x, x, x, trg_mask,return_attention=return_attention)
            query = drop_path(attention, self.drop_path_rate, self.training) + x
            
            
            out,QA = self.transformer_block(value, key, query, trg_mask,return_attention=return_attention) 
            return out,SA,QA
        else:
            x = self.norm(x)
            attention = self.attention(x, x, x, trg_mask)
            query = drop_path(attention, self.drop_path_rate, self.training) + x
            out = self.transformer_block(value, key, query, trg_mask) 


            return out

### Could have droppath for indivudal self-attention
class DecoderBlockV2(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout_attn, dropout_proj, dropout_embed, drop_path_rate=0.15):
        super(DecoderBlockV2, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)  
        #self.norm = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads, dropout_attn, dropout_proj)
        self.cross_attention = SelfAttention(embed_size, heads, dropout_attn, dropout_proj)  # Cross-Attention
        self.reverse_cross_attention = SelfAttention(embed_size, heads, dropout_attn, dropout_proj)  # Reverse Cross-Attention
        self.transformer_block = TransformerBlock(embed_size, heads, dropout_attn, dropout_proj, dropout_embed, drop_path_rate, forward_expansion)
        self.dropout = nn.Dropout(dropout_embed)
        self.drop_path_rate = drop_path_rate


        self.adaptive_fusion = nn.Linear(embed_size * 2, embed_size)  # Adaptive fusion layer

    def forward(self, x,placeholder, enc_out, src_mask, trg_mask,return_attention=False):# placeholder to make input the same for now
        # Cross-Attention: enc_out as query, x as key and value

        x1 = self.norm1(x)
        enc_out = self.norm2(enc_out)
        if return_attention:      
            x1,SA = self.attention(x1, x1, x1, trg_mask,return_attention=return_attention)
            x1 = self.norm3(x1 + x) # self.norm3(drop_path(x1, self.drop_path_rate, self.training) + x) # self.norm3(x1 + x)

            cross_attn_output,QA = self.cross_attention(enc_out, enc_out, x1, trg_mask,return_attention=return_attention)

            reverse_cross_attn_output,RQA = self.reverse_cross_attention(x1, x1, enc_out, trg_mask,return_attention=return_attention)

            fused_output = torch.cat((cross_attn_output, reverse_cross_attn_output), dim=-1)
            fused_output = self.dropout(self.adaptive_fusion(fused_output))
            out = self.transformer_block(fused_output, fused_output, fused_output, trg_mask)
            return out + x#drop_path(out, self.drop_path_rate, self.training) + x,SA, (QA,RQA)
        else:
            x1 = self.attention(x1, x1, x1, trg_mask)
            x1 = self.norm3(drop_path(x1, self.drop_path_rate, self.training) + x) #self.norm3(x1 + x)
            # Reverse Cross-Attention: att as query, enc_out as key and value
            cross_attn_output = self.cross_attention(enc_out, enc_out, x1, trg_mask)
            #cross_attn_output = self.dropout(cross_attn_output)

            # Reverse Cross-Attention: enc_out as query, att as key and value
            reverse_cross_attn_output = self.reverse_cross_attention(x1, x1, enc_out, trg_mask)
            #reverse_cross_attn_output = self.dropout(reverse_cross_attn_output)
            # Adaptive Fusion
            fused_output = torch.cat((cross_attn_output, reverse_cross_attn_output), dim=-1)
            fused_output = self.dropout(self.adaptive_fusion(fused_output))

            # Passing through the transformer block

            out = self.transformer_block(fused_output, fused_output, fused_output, trg_mask)

            return out + x #drop_path(out, self.drop_path_rate, self.training) + x#self.dropout(drop_path(out, self.drop_path_rate, self.training) + x)
from transformers import BertModel, BertTokenizer
class BertEmbedder(nn.Module):
    def __init__(self, bert_model_name, embed_dim):
        super(BertEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        #tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, embed_dim)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            # Extract last layer hidden states from BERT
            input_ids = input_ids.squeeze(0) ### works only for batch size 1, rearrange can be used in that case instead.
            attention_mask = attention_mask.squeeze(0)
            #input_ids = rearrange(input_ids,'b s d -> (b s) d')
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output  # Use the pooled output for classification tasks
        pooled_output = pooled_output.unsqueeze(0)
        # Pass the BERT embeddings to the custom classifier layer
        logits = self.classifier(pooled_output)
        return logits


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout_attn,
        dropout_proj,
        dropout_embed,
        drop_path_rate,
        max_length_ral=10,
        bert_model_name = 'bert-base-uncased'
    ):
        super(Decoder, self).__init__()
        if bert_model_name is None:
            self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
            self.bert_on = False
        else:
            self.word_embedding = BertEmbedder(bert_model_name, embed_size)#.to(device)
            self.bert_on = True

        
        self.player_embed = AlternatingEmbeddingAdder(embed_size)
        
        self.position_embedding = nn.Embedding(max_length_ral, embed_size)
        
        inc_drop_rates = np.linspace(0, drop_path_rate, num_layers)
        self.layers = nn.ModuleList(
            [
                DecoderBlockV2(embed_size, heads, forward_expansion, dropout_attn, dropout_proj, dropout_embed, inc_drop_rates[i])
                for i in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out,id, src_mask, trg_mask,return_attention=False,bert_Amask=None): ## add seq-embedding to encoding tokens layers
     
        #N, seq_length = x.shape
        N, seq_length,_ = x.shape

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        if self.bert_on:
            #bert_ids = x#.to(device)
            bert_mask = bert_Amask#.to(device)
            x = self.word_embedding(x, bert_mask) +  self.position_embedding(positions)
        else:
            x = self.word_embedding(x) + self.position_embedding(positions)### word_embedding 
        enc_player = enc_out[:,:,1:]
        enc_player = enc_player[:,:,0]*id[:,:,0,1].unsqueeze(-1) + enc_player[:,:,1]*id[:,:,1,1].unsqueeze(-1)
        enc_out = enc_out[:,:,0]
        #enc_out = self.player_embed(enc_out,id)

        x = x + enc_player # self.player_embed(x,id) + enc_player
        if return_attention:
            for layer in self.layers:
                x,SA,CA = layer(x, enc_out, enc_out, src_mask, trg_mask,return_attention=return_attention) #org

            return SA,CA
        else:
            for layer in self.layers:
                x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)


        return out


class AlternatingEmbeddingAdder(nn.Module):
    def __init__(self, D, player_db = 28):
        super(AlternatingEmbeddingAdder, self).__init__()
        # Initialize the binary embeddings as learnable parameters
        self.players = player_db
        self.player_embeddings = nn.Embedding(self.players, D,padding_idx=0,scale_grad_by_freq=True)


    def forward(self, sequence,id):
        N, L, D = sequence.shape
        player1 = self.player_embeddings(id[:,:,0,0])
        player2 = self.player_embeddings(id[:,:,1,0])
        id_embedding = player1*id[:,:,0,1].unsqueeze(-1) + player2*id[:,:,1,1].unsqueeze(-1) ## make use of turnbased sport (i.e. badminton)

        return sequence + id_embedding

class RallyTempose(nn.Module):
    def __init__(
        self,
        embed_size,
        embed_size_spat,
        channels,
        num_joints,
        trg_vocab_size,
        trg_pad_idx,
        heads=8,
        heads_spat=4,
        num_encoder_layers=6,
        num_encspat_layers=6,
        num_decoder_layers=8,
        forward_expansion=4,
        dropout_attn=0.,
        dropout_proj=0.,
        dropout_embed=0.,
        drop_path_rate=0.,
        max_length=100,
        ral_length=60,
        prob_bool = False,
        get_latent = False,
    ): 

        super(RallyTempose, self).__init__()
        num_people=2
        self.prob_bool = prob_bool
        self.encoder =  TemPoseIII(
            channels,
            num_joints,
            embed_size,
            embed_size_spat,
            num_encoder_layers,
            num_encspat_layers,
            heads,
            heads_spat,
            forward_expansion,
            dropout_attn,
            dropout_proj,
            dropout_embed,
            drop_path_rate,
            max_length)

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_decoder_layers,
            heads,
            forward_expansion,
            dropout_attn,
            dropout_proj,
            dropout_embed,
            drop_path_rate,
            ral_length)

        self.trg_pad_idx = trg_pad_idx
        #self.trg_pad_idx = trg_pad_idx

        #self.to_enc_src = nn.Linear(embed_size*num_people,embed_size) ## can scale down here 
        self.linear_probe = ProbeLayer(embed_size,trg_vocab_size)
        self.get_latent = get_latent
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def make_temp_padding_mask(self, temp, src):
        N, sq_len = temp.shape

        _, _, people, src_len, embed_dim = src.shape
        
        # Initialize src_mask with zeros
        src_mask = torch.zeros(N, sq_len, people+1, 1, 1, src_len, dtype=torch.long, device=src.device) ## people +1 for the mean / rally embedding channel
        
        # Create a 1D tensor `mask_indices` of shape (src_len,) 
        # with values from 0 to src_len-1
        mask_indices = torch.arange(src_len, device=src.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Expand `temp` and `mask_indices` to have the same shape for broadcasting
        expanded_temp = temp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, -1, src_len)
        mask_indices = mask_indices.expand(N, sq_len, people+1, -1, -1, -1) ## pepole+1 for the mean / rally embedding channel
        
        # Create a mask based on where the index is smaller than temp
        src_mask = (mask_indices < expanded_temp).type(torch.long)
        # Re-arrange src_mask if needed
        src_mask = rearrange(src_mask, 'b s p y x t -> (b s p) y x t').type(torch.long)
        return src_mask

    def make_trg_padding_mask(self, trg):
        N,sq_len = trg.shape
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        return trg_mask.to(trg.device)
    def make_src_ones_mask(self, src):
        N,sq_len,people,src_len,embed_dim = src.shape
        src_mask = torch.ones(N*sq_len*people,1,1,src_len).type(torch.LongTensor)
 
        return src_mask.to(src.device)

    def make_trg_mask(self, trg):
            #N, trg_len = trg.shape
            N, trg_len, _ = trg.shape
            trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
                N, 1, trg_len, trg_len
            )
    
            return trg_mask.to(trg.device)
    def make_trg_mask2(self, trg): ### mask for assymetrical input
            #N, trg_len = trg.shape
            N, trg_len,_ = trg.shape
            trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
                N, 1, trg_len, trg_len
            )
            trg_mask = torch.cat((trg_mask,torch.ones((N,1,1,trg_len))),dim=2)
    
            return trg_mask.to(trg.device)

    def make_src_mask(self, src):
        N, sq_len, src_len, _ = src.shape
        src_mask = torch.tril(torch.ones((src_len, src_len))).expand(
            N, 1, src_len, src_len
        )
        return src_mask.to(src.device)
    

    def extract_attention(self, src, trg,temp=None,bert_mask = None):
        KA,TA,SA,QA= self.forward(src,trg,temp,return_attention=True,bert_mask=bert_mask)
        return KA, TA, SA, QA
            
    def forward(self, src, trg, ID,temp=None,return_attention=False,bert_mask=None):

        if temp is not None:
            src_padding = self.make_temp_padding_mask(temp,src)
        else: 
            src_padding = self.make_src_padding_mask(src)

        #trg_padding = self.make_trg_padding_mask(trg)

            
        #one_pad_src = self.make_src_ones_mask(src)    
        #src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        trg_mask2 = self.make_trg_mask2(trg)

        full_trg_mask = trg_mask#*trg_padding

        if return_attention:
            enc_src,KA,TA = self.encoder(src, ID, src_padding,return_attention=return_attention)
        else:
            enc_src = self.encoder(src, ID, src_padding)
        if self.get_latent:
            return enc_src
        if self.prob_bool:
            aux_out = self.linear_probe(enc_src[:,:,0])
        #enc_src = self.to_enc_src(enc_src)

        if return_attention:
             SA,QA = self.decoder(trg,enc_src, ID, src_padding, full_trg_mask,return_attention=return_attention,bert_Amask=bert_mask)
             return KA,TA,SA,QA
        else:  
            #out = self.decoder(trg,enc_src, src_padding, trg_mask)
            out = self.decoder(trg,enc_src, ID, trg_mask2, full_trg_mask,bert_Amask=bert_mask)
        if self.prob_bool:
            return out,aux_out
        else:
            return out
    def predict(self,src,trg, ID,temp=None,bert_mask=None):
        #Apply softmax to output. 
        #b,t = trg.shape
        b,t,_ = trg.shape

        if self.prob_bool:
            out,aux_out= self.forward(src,trg,ID,temp,bert_mask=bert_mask)
            aux_pred = F.softmax(rearrange(aux_out,'b t d-> (b t) d'),dim=1).max(1)[1]
            aux_pred = rearrange(aux_pred,'(b t)-> b t',t=t)
            pred = F.softmax(rearrange(out,'b t d-> (b t) d'),dim=1).max(1)[1]
            pred = rearrange(pred,'(b t)-> b t',t=t)
            return pred,aux_pred
        else:
            out= self.forward(src,trg,ID,temp,bert_mask=bert_mask)
            pred = F.softmax(rearrange(out,'b t d-> (b t) d'),dim=1).max(1)[1]
            pred = rearrange(pred,'(b t)-> b t',t=t)
            return pred
    def predict_sample(self, src, trg, ID, temp=None,bert_mask=None):
        b, t = trg.shape

        # Forward pass
        if self.prob_bool:
            out, _ = self.forward(src, trg, ID, temp,bert_mask=bert_mask)
        else:
            out = self.forward(src, trg, ID, temp,bert_mask=bert_mask)

        # Softmax to get probabilities
        out_prob = F.softmax(rearrange(out, 'b t d -> (b t) d'), dim=1)

        # Sampling from the probability distribution
        pred = torch.multinomial(out_prob, 1).squeeze(1) # This randomly samples indices based on the given probabilities
        pred = rearrange(pred, '(b t) -> b t', t=t)

        return pred
    def get_probs(self,src,trg, ID,temp=None,bert_mask=None):
        #Apply softmax to output. 
        b,t = trg.shape
        pred = F.softmax(rearrange(self.forward(src,trg, ID,temp,bert_mask=bert_mask),'b t d-> (b t) d'),dim=1)
        #pred = rearrange(pred,'(b t)-> b t',t=t)
        return pred


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(device)
    x = torch.randn(2,5,65,10).type(torch.FloatTensor)
    trg = torch.randint(0,9,(2,5)).type(torch.LongTensor)
    print(trg)

    x = x.to(device)
    trg = trg.to(device)
    embed_size=512
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
