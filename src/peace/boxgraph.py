"""BoxGraph code encoder: a weight-shared message-passing network over Faust boxes."""

from dataclasses import dataclass
from typing import Sequence

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax.ops import segment_max, segment_sum

from .graph import EffectLibrary, GraphBatch, NodeType


@dataclass
class BoxGraphConfig:
    """BoxGraph encoder configuration.

    Attributes:
        hidden_dim: Node feature width.
        out_dim: Graph embedding width.
        num_layers: Minimum number of message-passing steps. Deeper graphs run
            one step per level so every node reaches the root.
    """

    hidden_dim: int = 1024
    out_dim: int = 512
    num_layers: int = 4


class MessagePassing(nnx.Module):
    """One shared message, max-aggregation, and update step.

    Each child sends a category-specific projection of its state, concatenated
    with a learned embedding of its argument position. Parents take the
    elementwise max over children and recompute their state from their initial
    features and the aggregate, followed by a residual feed-forward layer.
    """

    def __init__(self, dim: int, rngs: nnx.Rngs) -> None:
        self.param_msg = nnx.Linear(dim, dim, rngs=rngs)
        self.dsp_msg = nnx.Linear(dim, dim, rngs=rngs)
        self.comp_msg = nnx.Linear(dim, dim, rngs=rngs)
        self.bus_msg = nnx.Linear(dim, dim, rngs=rngs)
        self.pos_proj = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.norm = nnx.LayerNorm(2 * dim, rngs=rngs)
        self.update = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.ffn_in = nnx.Linear(dim, dim, rngs=rngs)
        self.ffn_out = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(
        self,
        h: jax.Array,
        x: jax.Array,
        graphs: GraphBatch,
        positions: jax.Array,
    ) -> jax.Array:
        node_type = graphs.node_types[:, None]
        messages = jnp.select(
            [
                (node_type == NodeType.FLOAT_PARAM) | (node_type == NodeType.INT_PARAM),
                node_type == NodeType.DSP,
                (node_type == NodeType.PAR) | (node_type == NodeType.SEQ),
            ],
            [self.param_msg(h), self.dsp_msg(h), self.comp_msg(h)],
            self.bus_msg(h),
        )
        messages = messages[graphs.edge_sources]
        messages = self.pos_proj(jnp.concatenate([messages, positions], axis=-1))
        aggregate = segment_max(messages, graphs.edge_targets, num_segments=h.shape[0])
        aggregate = jnp.where(jnp.isneginf(aggregate), 0.0, aggregate)
        h = nnx.relu(self.update(self.norm(jnp.concatenate([x, aggregate], axis=-1))))
        return h + self.ffn_out(nnx.relu(self.ffn_in(h)))


class BoxGraphEncoder(nnx.Module):
    """Embed BoxGraphs by message passing and mean pooling ``seq`` and DSP nodes."""

    def __init__(
        self, config: BoxGraphConfig, library: EffectLibrary, rngs: nnx.Rngs
    ) -> None:
        self.config = config
        self.library = library
        dim = config.hidden_dim
        sizes = library.int_table_sizes
        self.int_offsets = tuple(int(n) for n in np.cumsum([0, *[n + 1 for n in sizes[:-1]]]))
        self.int_sizes = tuple(sizes)
        self.type_embed = nnx.Embed(len(NodeType), dim, rngs=rngs)
        self.effect_embed = nnx.Embed(len(library.effects), dim, rngs=rngs)
        self.int_embed = nnx.Embed(sum(n + 1 for n in sizes), dim, rngs=rngs)
        self.value_proj = nnx.Linear(1, dim, rngs=rngs)
        self.value_mask = nnx.Param(jnp.zeros(dim))
        self.proj_param = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.proj_dsp = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.position_embed = nnx.Embed(library.max_arguments, dim, rngs=rngs)
        self.input_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.message_passing = MessagePassing(dim, rngs)
        self.output_proj = nnx.Linear(dim, config.out_dim, rngs=rngs)

    @property
    def out_features(self) -> int:
        return self.config.out_dim

    def prepare(
        self, programs: Sequence[str], mask_parameters: bool
    ) -> tuple[GraphBatch, tuple[int, int, bool]]:
        """Parse programs into one graph batch and the static call arguments."""
        graphs = [self.library.parse(program) for program in programs]
        num_layers = max(self.config.num_layers, max(g.depth for g in graphs))
        return self.library.batch(graphs), (len(graphs), num_layers, mask_parameters)

    def node_features(self, graphs: GraphBatch, mask_parameters: bool) -> jax.Array:
        """Initial node features from type, value, and effect embeddings."""
        node_type = graphs.node_types
        is_float = node_type == NodeType.FLOAT_PARAM
        is_int = node_type == NodeType.INT_PARAM
        types = self.type_embed(node_type)

        float_values = self.value_proj(graphs.node_values[:, None] * 2.0 - 1.0)
        table = jnp.maximum(graphs.node_int_tables, 0)
        offset = jnp.asarray(self.int_offsets)[table]
        size = jnp.asarray(self.int_sizes)[table]
        int_rows = offset + jnp.clip(graphs.node_values.astype(jnp.int32), 0, size - 1)
        if mask_parameters:
            float_values = jnp.broadcast_to(self.value_mask[...], float_values.shape)
            int_rows = offset + size
        values = jnp.where(is_float[:, None], float_values, self.int_embed(int_rows))

        params = self.proj_param(jnp.concatenate([types, values], axis=-1))
        effects = self.effect_embed(graphs.node_effects)
        dsp = self.proj_dsp(jnp.concatenate([types, effects], axis=-1))
        x = jnp.where(
            (is_float | is_int)[:, None],
            params,
            jnp.where((node_type == NodeType.DSP)[:, None], dsp, types),
        )
        return self.input_norm(x)

    def __call__(
        self,
        graphs: GraphBatch,
        num_graphs: int,
        num_layers: int,
        mask_parameters: bool = False,
    ) -> jax.Array:
        """Return ``[num_graphs, out_dim]`` graph embeddings.

        Args:
            graphs: Padded disjoint union of graphs.
            num_graphs: Number of real graphs in the batch.
            num_layers: Message-passing steps; at least the deepest graph's depth.
            mask_parameters: Replace every parameter value with its mask token so
                the embedding describes only effect choice and order.
        """
        x = self.node_features(graphs, mask_parameters)
        positions = self.position_embed(graphs.edge_positions)
        h = jnp.zeros_like(x)
        for _ in range(num_layers):
            h = self.message_passing(h, x, graphs, positions)
        pooled = (graphs.node_types == NodeType.SEQ) | (graphs.node_types == NodeType.DSP)
        total = segment_sum(
            jnp.where(pooled[:, None], h, 0.0), graphs.node_graphs, num_graphs + 1
        )
        count = segment_sum(pooled.astype(h.dtype), graphs.node_graphs, num_graphs + 1)
        return self.output_proj(total[:num_graphs] / jnp.maximum(count[:num_graphs], 1.0)[:, None])
