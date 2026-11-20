from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_out: int,
        circular: bool,
        segment_count: int,
        transformer_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.circular = circular
        self.d_model = d_model
        self.token_embedding = nn.Embedding(5, d_model, padding_idx=4)
        self.segment_embedding = (
            nn.Embedding(segment_count, d_model) if circular else None
        )
        self.local_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=5,
            padding=2,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )

    def _forward_impl(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.circular:
            batch, segments, length = tokens.shape
            x = self.token_embedding(tokens)
            if self.segment_embedding is not None:
                segment_ids = torch.arange(segments, device=tokens.device)
                x = x + self.segment_embedding(segment_ids)[None, :, None, :]
            x = x.reshape(batch, segments * length, self.d_model)
            flat_mask = mask.reshape(batch, segments * length)
        else:
            x = self.token_embedding(tokens)
            flat_mask = mask

        x = x.transpose(1, 2)
        x = self.local_conv(x)
        x = x.transpose(1, 2)
        x = self.encoder(x, src_key_padding_mask=~flat_mask)

        weights = flat_mask.unsqueeze(-1).to(dtype=x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output(pooled)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        chunk_size = 1024
        if torch.is_grad_enabled() or tokens.size(0) <= chunk_size:
            return self._forward_impl(tokens, mask)
        chunks = [
            self._forward_impl(
                tokens[start : start + chunk_size],
                mask[start : start + chunk_size],
            )
            for start in range(0, tokens.size(0), chunk_size)
        ]
        return torch.cat(chunks, dim=0)


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_out: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features)


class HybridSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_out: int,
        circular: bool,
        segment_count: int,
        transformer_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_encoder = FeatureEncoder(
            input_dim=input_dim,
            d_out=d_out,
            dropout=dropout,
        )
        self.token_encoder = SequenceEncoder(
            d_model=64,
            d_out=d_out,
            circular=circular,
            segment_count=segment_count,
            transformer_layers=transformer_layers,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_out * 2),
            nn.Linear(d_out * 2, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )

    def forward(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        feature_embedding = self.feature_encoder(features)
        token_embedding = self.token_encoder(tokens, mask)
        return self.fusion(torch.cat([feature_embedding, token_embedding], dim=-1))


class GraphEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.message_weights = nn.ModuleList(
            [
                nn.Linear(d_model, d_model)
                for _ in range(layers)
            ]
        )
        self.interaction_weights = nn.ModuleList(
            [
                nn.Linear(d_model, d_model)
                for _ in range(layers)
            ]
        )
        self.layer_gates = nn.ModuleList(
            [nn.Linear(d_model * 2, d_model) for _ in range(layers)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model * (layers + 1)),
            nn.Linear(d_model * (layers + 1), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        all_layers = [x]
        current = x
        for layer in range(self.layers):
            neighbors = torch.sparse.mm(norm_adj, current)
            direct = self.message_weights[layer](neighbors)
            interaction = self.interaction_weights[layer](current * neighbors)
            message = F.leaky_relu(direct + interaction, negative_slope=0.2)
            gate = torch.sigmoid(
                self.layer_gates[layer](torch.cat([current, message], dim=-1))
            )
            current = current + self.dropout(gate * message)
            current = F.normalize(current, p=2, dim=-1)
            all_layers.append(current)
        return self.output(torch.cat(all_layers, dim=-1))


class FusionBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        sequence_embedding: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:
        gamma = torch.tanh(self.gamma(graph_embedding))
        beta = self.beta(graph_embedding)
        gate = torch.sigmoid(
            self.gate(torch.cat([sequence_embedding, graph_embedding], dim=-1))
        )
        fused = gate * sequence_embedding + (1.0 - gate) * (gamma * graph_embedding + beta)
        return F.normalize(self.output(fused), p=2, dim=-1)


class InteractionDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        structure_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bilinear = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 4 + structure_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        circ_embedding: torch.Tensor,
        mir_embedding: torch.Tensor,
        structure_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bilinear = (self.bilinear(circ_embedding) * mir_embedding).sum(dim=-1)
        bilinear = bilinear / math.sqrt(circ_embedding.size(-1))
        features = torch.cat(
            [
                circ_embedding,
                mir_embedding,
                circ_embedding * mir_embedding,
                torch.abs(circ_embedding - mir_embedding),
            ],
            dim=-1,
        )
        if structure_features is not None:
            features = torch.cat([features, structure_features], dim=-1)
        return bilinear + self.mlp(features).squeeze(-1)


class CIRC_PU(nn.Module):
    def __init__(
        self,
        mir_tokens: torch.Tensor | None,
        mir_mask: torch.Tensor | None,
        circ_tokens: torch.Tensor | None,
        circ_mask: torch.Tensor | None,
        n_nodes: int,
        mir_features: torch.Tensor | None = None,
        circ_features: torch.Tensor | None = None,
        sequence_encoder: str = "transformer",
        structure_dim: int = 0,
        circ_segments: int = 4,
        d_model: int = 128,
        sequence_layers: int = 1,
        graph_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_encoder = sequence_encoder
        if sequence_encoder == "transformer":
            if mir_tokens is None or mir_mask is None:
                raise ValueError("mir_tokens and mir_mask are required")
            if circ_tokens is None or circ_mask is None:
                raise ValueError("circ_tokens and circ_mask are required")
            self.register_buffer("mir_tokens", mir_tokens, persistent=False)
            self.register_buffer("mir_mask", mir_mask, persistent=False)
            self.register_buffer("circ_tokens", circ_tokens, persistent=False)
            self.register_buffer("circ_mask", circ_mask, persistent=False)
            self.mir_encoder = SequenceEncoder(
                d_model=64,
                d_out=d_model,
                circular=False,
                segment_count=1,
                transformer_layers=sequence_layers,
                dropout=dropout,
            )
            self.circ_encoder = SequenceEncoder(
                d_model=64,
                d_out=d_model,
                circular=True,
                segment_count=circ_segments,
                transformer_layers=sequence_layers,
                dropout=dropout,
            )
        elif sequence_encoder == "kmer":
            if mir_features is None or circ_features is None:
                raise ValueError("mir_features and circ_features are required")
            self.register_buffer("mir_features", mir_features, persistent=False)
            self.register_buffer("circ_features", circ_features, persistent=False)
            self.mir_encoder = FeatureEncoder(
                input_dim=mir_features.size(1),
                d_out=d_model,
                dropout=dropout,
            )
            self.circ_encoder = FeatureEncoder(
                input_dim=circ_features.size(1),
                d_out=d_model,
                dropout=dropout,
            )
        elif sequence_encoder == "hybrid":
            if mir_tokens is None or mir_mask is None:
                raise ValueError("mir_tokens and mir_mask are required")
            if circ_tokens is None or circ_mask is None:
                raise ValueError("circ_tokens and circ_mask are required")
            if mir_features is None or circ_features is None:
                raise ValueError("mir_features and circ_features are required")
            self.register_buffer("mir_tokens", mir_tokens, persistent=False)
            self.register_buffer("mir_mask", mir_mask, persistent=False)
            self.register_buffer("circ_tokens", circ_tokens, persistent=False)
            self.register_buffer("circ_mask", circ_mask, persistent=False)
            self.register_buffer("mir_features", mir_features, persistent=False)
            self.register_buffer("circ_features", circ_features, persistent=False)
            self.mir_encoder = HybridSequenceEncoder(
                input_dim=mir_features.size(1),
                d_out=d_model,
                circular=False,
                segment_count=1,
                transformer_layers=sequence_layers,
                dropout=dropout,
            )
            self.circ_encoder = HybridSequenceEncoder(
                input_dim=circ_features.size(1),
                d_out=d_model,
                circular=True,
                segment_count=circ_segments,
                transformer_layers=sequence_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown sequence encoder: {sequence_encoder}")

        self.graph_encoder = GraphEncoder(
            d_model=d_model,
            layers=graph_layers,
            dropout=dropout,
        )
        self.node_embedding = nn.Embedding(n_nodes, d_model)
        self.type_embedding = nn.Embedding(2, d_model)
        nn.init.normal_(self.node_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)
        self.mir_fusion = FusionBlock(d_model, dropout=dropout)
        self.circ_fusion = FusionBlock(d_model, dropout=dropout)
        self.decoder = InteractionDecoder(
            d_model,
            structure_dim=structure_dim,
            dropout=dropout,
        )
        self.alignment_temperature = nn.Parameter(torch.tensor(0.1))

    def encode_nodes(
        self,
        norm_adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sequence_encoder == "transformer":
            mir_sequence = self.mir_encoder(self.mir_tokens, self.mir_mask)
            circ_sequence = self.circ_encoder(self.circ_tokens, self.circ_mask)
        elif self.sequence_encoder == "kmer":
            mir_sequence = self.mir_encoder(self.mir_features)
            circ_sequence = self.circ_encoder(self.circ_features)
        else:
            mir_sequence = self.mir_encoder(
                self.mir_features,
                self.mir_tokens,
                self.mir_mask,
            )
            circ_sequence = self.circ_encoder(
                self.circ_features,
                self.circ_tokens,
                self.circ_mask,
            )
        n_mir = mir_sequence.size(0)
        n_nodes = n_mir + circ_sequence.size(0)
        node_ids = torch.arange(n_nodes, device=mir_sequence.device)
        type_ids = torch.cat(
            [
                torch.zeros(n_mir, dtype=torch.long, device=mir_sequence.device),
                torch.ones(
                    circ_sequence.size(0),
                    dtype=torch.long,
                    device=mir_sequence.device,
                ),
            ]
        )
        initial = (
            torch.cat([mir_sequence, circ_sequence], dim=0)
            + self.node_embedding(node_ids)
            + self.type_embedding(type_ids)
        )
        graph_embedding = self.graph_encoder(initial, norm_adj)

        graph_mir = graph_embedding[: mir_sequence.size(0)]
        graph_circ = graph_embedding[mir_sequence.size(0) :]
        fused_mir = self.mir_fusion(mir_sequence, graph_mir)
        fused_circ = self.circ_fusion(circ_sequence, graph_circ)
        return fused_mir, fused_circ, mir_sequence, circ_sequence

    def forward(
        self,
        mir_index: torch.Tensor,
        circ_index: torch.Tensor,
        norm_adj: torch.Tensor,
        structure_features: torch.Tensor | None = None,
        return_alignment: bool = False,
    ):
        fused_mir, fused_circ, mir_sequence, circ_sequence = self.encode_nodes(
            norm_adj
        )
        scores = self.decoder(
            fused_circ[circ_index],
            fused_mir[mir_index],
            structure_features,
        )
        if not return_alignment:
            return scores

        temperature = self.alignment_temperature.clamp_min(0.02)
        mir_sequence_norm = F.normalize(mir_sequence, p=2, dim=-1)
        circ_sequence_norm = F.normalize(circ_sequence, p=2, dim=-1)
        graph_mir_norm = F.normalize(fused_mir, p=2, dim=-1)
        graph_circ_norm = F.normalize(fused_circ, p=2, dim=-1)
        mir_alignment = (mir_sequence_norm @ graph_mir_norm.T) / temperature
        circ_alignment = (circ_sequence_norm @ graph_circ_norm.T) / temperature
        return scores, mir_alignment, circ_alignment


class NGCFLite(nn.Module):
    def __init__(
        self,
        n_mir: int,
        n_circ: int,
        n_nodes: int,
        d_model: int = 128,
        graph_layers: int = 3,
        dropout: float = 0.1,
        structure_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_mir = n_mir
        self.n_circ = n_circ
        self.d_model = d_model
        self.graph_layers = graph_layers
        self.dropout = nn.Dropout(dropout)
        self.node_embedding = nn.Embedding(n_nodes, d_model)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        self.message_weights = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(graph_layers)]
        )
        self.interaction_weights = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(graph_layers)]
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.structure_projection = (
            nn.Linear(structure_dim, 1) if structure_dim > 0 else None
        )
        if self.structure_projection is not None:
            nn.init.zeros_(self.structure_projection.weight)
            nn.init.zeros_(self.structure_projection.bias)

    def encode_nodes(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.node_embedding.weight
        all_layers = [current]
        for layer in range(self.graph_layers):
            neighbors = torch.sparse.mm(norm_adj, current)
            direct = self.message_weights[layer](neighbors)
            interaction = self.interaction_weights[layer](current * neighbors)
            current = F.leaky_relu(direct + interaction, negative_slope=0.2)
            current = self.dropout(current)
            current = F.normalize(current, p=2, dim=-1)
            all_layers.append(current)
        combined = torch.cat(all_layers, dim=-1)
        combined = combined / math.sqrt(self.graph_layers + 1)
        mir = combined[: self.n_mir]
        circ = combined[self.n_mir :]
        return F.normalize(mir, p=2, dim=-1), F.normalize(circ, p=2, dim=-1)

    def forward(
        self,
        mir_index: torch.Tensor,
        circ_index: torch.Tensor,
        norm_adj: torch.Tensor,
        structure_features: torch.Tensor | None = None,
        return_alignment: bool = False,
    ):
        mir_embedding, circ_embedding = self.encode_nodes(norm_adj)
        scores = (
            mir_embedding[mir_index] * circ_embedding[circ_index]
        ).sum(dim=-1) + self.bias
        if self.structure_projection is not None and structure_features is not None:
            scores = scores + self.structure_projection(structure_features).squeeze(-1)
        if return_alignment:
            return scores, None, None
        return scores


def info_nce(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(0) < 2:
        return logits.new_zeros(())
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def non_negative_pu_risk(
    positive_scores: torch.Tensor,
    unlabeled_scores: torch.Tensor,
    positive_prior: float,
) -> torch.Tensor:
    if positive_scores.numel() == 0 or unlabeled_scores.numel() == 0:
        return positive_scores.new_zeros(())
    positive_prior_tensor = positive_scores.new_tensor(
        max(min(positive_prior, 0.5), 1e-5)
    )
    positive_risk = F.softplus(-positive_scores).mean()
    unlabeled_risk = F.softplus(unlabeled_scores).mean()
    positive_as_negative_risk = F.softplus(positive_scores).mean()
    corrected = unlabeled_risk - positive_prior_tensor * positive_as_negative_risk
    return positive_prior_tensor * positive_risk + torch.relu(corrected)
