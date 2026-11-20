"""Explicit CIRC-PU ablations; historical model defaults are not edited."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from codes.model import CIRC_PU

VARIANTS = ('full', 'sequence_only', 'graph_only', 'fixed_mean_fusion',
            'no_alignment', 'no_pu', 'no_node_id')

class ExperimentalCIRCPU(CIRC_PU):
    def __init__(self, *args, variant='full', training_node_mask=None,
                 active_id_mask=None, **kwargs):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.register_buffer('training_node_mask', training_node_mask, persistent=False)
        self.register_buffer('active_id_mask', active_id_mask, persistent=False)

    def _sequence(self):
        if self.sequence_encoder == 'transformer':
            return (self.mir_encoder(self.mir_tokens, self.mir_mask),
                    self.circ_encoder(self.circ_tokens, self.circ_mask))
        if self.sequence_encoder == 'kmer':
            return self.mir_encoder(self.mir_features), self.circ_encoder(self.circ_features)
        return (self.mir_encoder(self.mir_features, self.mir_tokens, self.mir_mask),
                self.circ_encoder(self.circ_features, self.circ_tokens, self.circ_mask))

    def encode_nodes(self, norm_adj):
        # This path preserves full-model numerical behavior when no masks are requested.
        if self.variant in ('full', 'no_alignment', 'no_pu') and self.active_id_mask is None:
            return super().encode_nodes(norm_adj)
        device = self.node_embedding.weight.device
        n_mir = self.mir_tokens.shape[0] if self.sequence_encoder != 'kmer' else self.mir_features.shape[0]
        n_nodes = self.node_embedding.num_embeddings
        types = torch.cat([torch.zeros(n_mir, dtype=torch.long, device=device),
                           torch.ones(n_nodes-n_mir, dtype=torch.long, device=device)])
        node_ids = self.node_embedding.weight
        if self.active_id_mask is not None:
            node_ids = node_ids * self.active_id_mask[:, None].to(node_ids.dtype)
        if self.variant == 'graph_only':
            graph = self.graph_encoder(node_ids + self.type_embedding(types), norm_adj)
            return F.normalize(graph[:n_mir], dim=-1), F.normalize(graph[n_mir:], dim=-1), None, None
        mir_sequence, circ_sequence = self._sequence()
        if self.variant == 'sequence_only':
            return F.normalize(mir_sequence, dim=-1), F.normalize(circ_sequence, dim=-1), mir_sequence, circ_sequence
        initial = torch.cat([mir_sequence, circ_sequence]) + self.type_embedding(types)
        if self.variant != 'no_node_id':
            initial = initial + node_ids
        graph = self.graph_encoder(initial, norm_adj)
        graph_mir, graph_circ = graph[:n_mir], graph[n_mir:]
        if self.variant == 'fixed_mean_fusion':
            fused_mir = F.normalize(self.mir_fusion.output(0.5*(mir_sequence+graph_mir)), dim=-1)
            fused_circ = F.normalize(self.circ_fusion.output(0.5*(circ_sequence+graph_circ)), dim=-1)
        else:
            fused_mir = self.mir_fusion(mir_sequence, graph_mir)
            fused_circ = self.circ_fusion(circ_sequence, graph_circ)
        return fused_mir, fused_circ, mir_sequence, circ_sequence

    def forward(self, mir_index, circ_index, norm_adj,
                structure_features=None, return_alignment=False):
        fm, fc, sm, sc = self.encode_nodes(norm_adj)
        scores = self.decoder(fc[circ_index], fm[mir_index], structure_features)
        if not return_alignment:
            return scores
        if self.variant in ('sequence_only', 'graph_only', 'no_alignment'):
            return scores, None, None
        if self.training_node_mask is not None:
            mm, cm = self.training_node_mask[:len(sm)], self.training_node_mask[len(sm):]
            sm, fm, sc, fc = sm[mm], fm[mm], sc[cm], fc[cm]
        temp = self.alignment_temperature.clamp_min(0.02)
        return scores, F.normalize(sm,dim=-1) @ F.normalize(fm,dim=-1).T / temp, F.normalize(sc,dim=-1) @ F.normalize(fc,dim=-1).T / temp
