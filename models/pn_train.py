import torch.nn as nn
import torch

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, detect=False):
        batch_size, num_nodes, tgt_length, _ = query.shape
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        if detect:
            query_activate = query.sum(dim=(0, 1, 2)).reshape(-1)
            key_activate = key.sum(dim=(0, 1, 2)).reshape(-1)
            value_activate = value.sum(dim=(0, 1, 2)).reshape(-1)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)

        attn_score = (query @ key) / self.head_dim ** 0.5

        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)

        out = attn_score @ value

        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)

        out = self.out_proj(out)

        if detect:
            out_activate = out.sum(dim=(0, 1, 2)).reshape(-1)
            return out, (query_activate, key_activate, value_activate, out_activate)
        else:
            return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False, id=None
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)

        self.feed_forward_up = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
        )

        self.feed_forward_down = nn.Linear(feed_forward_dim, model_dim)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mask = mask
        # self.id = id

    def forward(self, x, dim=-2, detect=False):

        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x

        if detect:
            out, attn_activate = self.attn(x, x, x, detect=detect)  # (batch_size, ..., length, model_dim)
        else:
            out = self.attn(x, x, x)

        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out

        out = self.feed_forward_up(out)

        if detect:
            up_activate = out.sum(dim=(0, 1, 2)).reshape(-1)

        out = self.feed_forward_down(out)

        if detect:
            down_activate = out.sum(dim=(0, 1, 2)).reshape(-1)

        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)

        if detect:
            return out, attn_activate, (up_activate, down_activate)
        else:
            return out

class STAEformer(nn.Module):
    def __init__(
            self,
            args,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=1,
            output_dim=1,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = args.num_nodes
        self.in_steps = args.seq_len
        self.out_steps = args.pred_len
        self.steps_per_day = args.time_of_day_size
        self.time_interval = args.time_interval
        self.output_dim = output_dim
        self.input_embedding_dim = args.input_embedding_dim
        self.tod_embedding_dim = args.tod_embedding_dim
        self.dow_embedding_dim = args.dow_embedding_dim
        self.spatial_embedding_dim = args.spatial_embedding_dim
        self.adaptive_embedding_dim = args.adaptive_embedding_dim

        self.feed_forward_dim = feed_forward_dim

        self.model_dim = (
                self.input_embedding_dim
                + self.tod_embedding_dim
                + self.dow_embedding_dim
                + self.spatial_embedding_dim
                + self.adaptive_embedding_dim
        )

        self.num_heads = num_heads
        self.num_layers = args.num_layers
        self.use_mixed_proj = use_mixed_proj

        if self.tod_embedding_dim > 0:
            input_dim += 1
            self.tod_embedding = nn.Embedding(steps_per_day, self.tod_embedding_dim)

        if self.dow_embedding_dim > 0:
            input_dim += 1
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)

        self.input_dim = input_dim
        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)

        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)

        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, self.num_nodes, self.adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for i in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, batch_x_mark, detect=False):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)

        if self.tod_embedding_dim > 0:
            tod = ((batch_x_mark[..., 1] * 60 + batch_x_mark[..., 0]) // self.time_interval) / self.steps_per_day
            tod = tod.reshape(batch_size, self.in_steps, 1, 1)
            tod = tod.repeat(1, 1, self.num_nodes, 1).to(x.device)
            x = torch.cat([x, tod], dim=-1)

        if self.dow_embedding_dim > 0:
            dow = batch_x_mark[..., 2]
            dow = dow.reshape(batch_size, self.in_steps, 1, 1)
            dow = dow.repeat(1, 1, self.num_nodes, 1).to(x.device)
            x = torch.cat([x, dow], dim=-1)

        x = self.input_proj(x.float())

        features = [x]

        if self.tod_embedding_dim > 0:
            tod = tod.squeeze(-1)
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
            features.append(tod_emb)

        if self.dow_embedding_dim > 0:
            dow = dow.squeeze(-1)
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)

        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)

        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)

        if detect:
            temporal_q, temporal_k, temporal_vo, temporal_out, temporal_ffn_up, temporal_ffn_down = {}, {}, {}, {}, {}, {}
            spatio_q, spatio_k, spatio_vo, spatio_out, spatio_ffn_up, spatio_ffn_down = {}, {}, {}, {}, {}, {}

            temporal_q_save, temporal_k_save, temporal_vo_save, temporal_out_save, temporal_ffn_up_save, temporal_ffn_down_save = {}, {}, {}, {}, {}, {}
            spatio_q_save, spatio_k_save, spatio_vo_save, spatio_out_save, spatio_ffn_up_save, spatio_ffn_down_save = {}, {}, {}, {}, {}, {}

        for idx, attn in enumerate(self.attn_layers_t):
            if detect:
                x, (query_activate, key_activate, value_activate, out_activate), (up_activate, down_activate) = attn(x, dim=1, detect=detect)
                temporal_q[idx] = query_activate
                temporal_k[idx] = key_activate
                temporal_vo[idx] = value_activate
                temporal_out[idx] = out_activate
                temporal_ffn_up[idx] = up_activate
                temporal_ffn_down[idx] = down_activate

                temporal_q_save[idx] = query_activate.detach().cpu().numpy()
                temporal_k_save[idx] = key_activate.detach().cpu().numpy()
                temporal_vo_save[idx] = value_activate.detach().cpu().numpy()
                temporal_out_save[idx] = out_activate.detach().cpu().numpy()
                temporal_ffn_up_save[idx] = up_activate.detach().cpu().numpy()
                temporal_ffn_down_save[idx] = down_activate.detach().cpu().numpy()
            else:
                x = attn(x, dim=1)

        for idx, attn in enumerate(self.attn_layers_s):
            if detect:
                x, (query_activate, key_activate, value_activate, out_activate), (up_activate, down_activate) = attn(x, dim=2, detect=detect)
                spatio_q[idx] = query_activate
                spatio_k[idx] = key_activate
                spatio_vo[idx] = value_activate
                spatio_out[idx] = out_activate
                spatio_ffn_up[idx] = up_activate
                spatio_ffn_down[idx] = down_activate

                spatio_q_save[idx] = query_activate.detach().cpu().numpy()
                spatio_k_save[idx] = key_activate.detach().cpu().numpy()
                spatio_vo_save[idx] = value_activate.detach().cpu().numpy()
                spatio_out_save[idx] = out_activate.detach().cpu().numpy()
                spatio_ffn_up_save[idx] = up_activate.detach().cpu().numpy()
                spatio_ffn_down_save[idx] = down_activate.detach().cpu().numpy()
            else:
                x = attn(x, dim=2)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(out)  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(out.transpose(1, 3))  # (batch_size, out_steps, num_nodes, output_dim)
        if detect:
            return (out.squeeze(-1), (temporal_q, temporal_k, temporal_vo, temporal_out, temporal_ffn_up, temporal_ffn_down), (spatio_q, spatio_k, spatio_vo, spatio_out, spatio_ffn_up, spatio_ffn_down),
                    (temporal_q_save, temporal_k_save, temporal_vo_save, temporal_out_save, temporal_ffn_up_save, temporal_ffn_down_save),
                    (spatio_q_save, spatio_k_save, spatio_vo_save, spatio_out_save, spatio_ffn_up_save, spatio_ffn_down_save))
        else:
            return out.squeeze(-1)

