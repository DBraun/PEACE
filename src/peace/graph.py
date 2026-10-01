"""Parse Faust effect chains into padded BoxGraph arrays.

A BoxGraph is a DAG of Faust Box API terms. Leaves are parameter constants, the
stereo bus, and library effects; composition nodes are ``par`` and ``seq``. An
effect call ``reverb(0.2, 0.7)`` becomes ``(0.2, 0.7, bus) : reverb``, and a
chain ``a : b`` becomes a ``seq`` node over the two effect subtrees. Edges point
from children to parents and carry the child's argument position.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from functools import cache
from typing import NamedTuple, Sequence

import numpy as np


class NodeType(IntEnum):
    """BoxGraph node categories seen during training."""

    FLOAT_PARAM = 0
    INT_PARAM = 1
    BUS = 2
    PAR = 3
    SEQ = 4
    DSP = 5


@dataclass(frozen=True)
class Parameter:
    """One effect parameter.

    Attributes:
        name: Parameter name in the effect library.
        kind: ``"float"`` for values in ``[0, 1]`` or ``"int"`` for choices.
        default: Library default value.
        num_values: Number of integer choices; zero for float parameters.
    """

    name: str
    kind: str
    default: float
    num_values: int = 0


@dataclass(frozen=True)
class Effect:
    """A Faust library effect with a single stereo bus.

    Attributes:
        name: Display name, such as ``"ReverbZita"``.
        function: Faust function name used in programs, such as ``"reverb_zita"``.
        parameters: Parameters in call order.
    """

    name: str
    function: str
    parameters: tuple[Parameter, ...]


@dataclass
class BoxGraph:
    """A BoxGraph whose nodes are stored in creation order.

    Children are always created before their parents, so the last node is the
    root and a single forward pass computes node depths.
    """

    node_types: list[int] = field(default_factory=list)
    node_values: list[float] = field(default_factory=list)
    node_effects: list[int] = field(default_factory=list)
    node_int_tables: list[int] = field(default_factory=list)
    edges: list[tuple[int, int, int]] = field(default_factory=list)

    def add_node(
        self,
        node_type: NodeType,
        value: float = 0.0,
        effect: int = 0,
        int_table: int = -1,
        children: Sequence[int] = (),
    ) -> int:
        """Append a node and edges from its children in argument order."""
        node = len(self.node_types)
        self.node_types.append(int(node_type))
        self.node_values.append(float(value))
        self.node_effects.append(effect)
        self.node_int_tables.append(int_table)
        self.edges.extend((child, node, pos) for pos, child in enumerate(children))
        return node

    @property
    def num_nodes(self) -> int:
        return len(self.node_types)

    @property
    def depth(self) -> int:
        """Longest leaf-to-root path, in edges."""
        depth = [0] * self.num_nodes
        for child, parent, _ in self.edges:
            depth[parent] = max(depth[parent], depth[child] + 1)
        return depth[-1] if depth else 0


class GraphBatch(NamedTuple):
    """A disjoint union of BoxGraphs padded to fixed node and edge counts.

    Padding nodes belong to graph ``num_graphs`` and padding edges connect the
    last padding node to itself, so neither affects real graphs.
    """

    node_types: np.ndarray  # [N] int32
    node_values: np.ndarray  # [N] float32
    node_effects: np.ndarray  # [N] int32, effect index for DSP nodes
    node_int_tables: np.ndarray  # [N] int32, discrete table for INT_PARAM nodes
    node_graphs: np.ndarray  # [N] int32
    edge_sources: np.ndarray  # [E] int32
    edge_targets: np.ndarray  # [E] int32
    edge_positions: np.ndarray  # [E] int32


class EffectLibrary:
    """Effects a model was trained with, and their embedding indices.

    Effect indices follow the sorted Faust function names. Discrete parameter
    tables are numbered by effect in that order, then by parameter call order.
    """

    def __init__(self, effects: Sequence[Effect]) -> None:
        self.effects = tuple(sorted(effects, key=lambda e: e.function))
        self.by_function = {e.function: e for e in self.effects}
        self.effect_index = {e.function: i for i, e in enumerate(self.effects)}
        self.int_tables: dict[tuple[str, str], int] = {}
        self.int_table_sizes: list[int] = []
        for effect in self.effects:
            for param in effect.parameters:
                if param.kind == "int":
                    self.int_tables[effect.function, param.name] = len(
                        self.int_table_sizes
                    )
                    self.int_table_sizes.append(param.num_values)
        self.max_arguments = max(len(e.parameters) for e in self.effects) + 1

    def parse(self, code: str) -> BoxGraph:
        """Parse a Faust program whose ``process`` chains library effects.

        Supported programs define ``process`` as effects joined by ``:``, where
        each effect is a call with numeric arguments, an effect name without
        parameters, or a reference to another definition. Float arguments lie in
        ``[0, 1]``; integer arguments index the parameter's choices.

        Raises:
            ValueError: If the program uses unsupported syntax or effects, or
                has arguments that do not match the effect's parameters.
        """
        root = _parser().parse(code.encode()).root_node
        if root.has_error:
            raise ValueError("Faust program has syntax errors")
        definitions = {}
        for child in root.children:
            named = _named(child)
            if child.type != "definition" or len(named) < 2:
                continue
            name_node = named[0]
            if name_node.child_count:
                name_node = name_node.children[0]
            definitions[_text(name_node)] = named[1]
        if "process" not in definitions:
            raise ValueError("Faust program has no process definition")
        graph = BoxGraph()
        self._convert(graph, definitions["process"], definitions, ())
        return graph

    def chain(self, code: str) -> list[tuple[Effect, tuple[float, ...]]]:
        """Effects in processing order, each with its parameter values.

        Parameter nodes are created just before their effect's DSP node, and
        ``a : b`` builds ``a`` first, so creation order is processing order.
        """
        graph = self.parse(code)
        chain, values = [], []
        for node_type, value, effect in zip(
            graph.node_types, graph.node_values, graph.node_effects
        ):
            if node_type in (NodeType.FLOAT_PARAM, NodeType.INT_PARAM):
                values.append(value)
            elif node_type == NodeType.DSP:
                chain.append((self.effects[effect], tuple(values)))
                values = []
        return chain

    def program(self, chain: Sequence[tuple[Effect | str, Sequence[float]]]) -> str:
        """Write a chain as Faust code in the format of the training programs.

        Each effect is bound to ``box1``, ``box2``, and so on, one parameter per
        line with its name as a comment. The BoxGraph encoder ignores layout, but
        the T5 encoder reads the text, so its inputs should use this format.

        Args:
            chain: ``(effect, values)`` pairs in processing order, where the
                effect is an ``Effect`` or its function name, as ``chain`` returns.
        """
        definitions = []
        for i, (effect, values) in enumerate(chain, 1):
            function = effect.function if isinstance(effect, Effect) else effect
            effect = self.by_function.get(function)
            if effect is None:
                raise ValueError(f"Unknown effect: {function}")
            if len(values) != len(effect.parameters):
                raise ValueError(
                    f"{function} takes {len(effect.parameters)} arguments, got {len(values)}"
                )
            lines = [
                f"    {int(v) if p.kind == 'int' else f'{v:.4f}'}"
                f"{',' if j < len(values) - 1 else ''}  // {p.name}"
                for j, (v, p) in enumerate(zip(values, effect.parameters))
            ]
            body = f"{function}(\n" + "\n".join(lines) + "\n)" if lines else function
            definitions.append(f"box{i} = {body};")
        boxes = " : ".join(f"box{i}" for i in range(1, len(chain) + 1))
        return "\n\n".join(definitions) + f"\n\nprocess = {boxes};\n"

    def _convert(self, graph: BoxGraph, node, definitions: dict, stack: tuple) -> int:
        if node.type == "identifier":
            name = _text(node)
            if name in definitions:
                if name in stack:
                    raise ValueError(f"Recursive definition: {name}")
                return self._convert(
                    graph, definitions[name], definitions, stack + (name,)
                )
            return self._add_effect(graph, name, [])
        if node.type == "function_call":
            callee, *rest = _named(node)
            values = []
            for arg in _named(rest[0]) if rest else ():
                if arg.type in ("int", "real", "unary_number"):
                    values.append(float(_text(arg)))
                elif arg.type != "comment":
                    raise ValueError(f"Effect arguments must be numbers: {_text(arg)}")
            return self._add_effect(graph, _text(callee), values)
        if node.type == "sequential":
            left, right = _named(node)
            left = self._convert(graph, left, definitions, stack)
            right = self._convert(graph, right, definitions, stack)
            return graph.add_node(NodeType.SEQ, children=(left, right))
        raise ValueError(f"Unsupported Faust expression: {_text(node)!r}")

    def _add_effect(self, graph: BoxGraph, function: str, values: list[float]) -> int:
        effect = self.by_function.get(function)
        if effect is None:
            raise ValueError(f"Unknown effect: {function}")
        if len(values) != len(effect.parameters):
            raise ValueError(
                f"{function} takes {len(effect.parameters)} arguments, got {len(values)}"
            )
        inputs = []
        for value, param in zip(values, effect.parameters):
            if param.kind == "int":
                if not (value.is_integer() and 0 <= value < param.num_values):
                    raise ValueError(
                        f"{function}.{param.name} must be an integer in "
                        f"[0, {param.num_values - 1}], got {value}"
                    )
                table = self.int_tables[function, param.name]
                inputs.append(graph.add_node(NodeType.INT_PARAM, value, int_table=table))
            else:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{function}.{param.name} must be in [0, 1], got {value}"
                    )
                inputs.append(graph.add_node(NodeType.FLOAT_PARAM, value))
        inputs.append(graph.add_node(NodeType.BUS))
        arguments = (
            inputs[0] if len(inputs) == 1 else graph.add_node(NodeType.PAR, children=inputs)
        )
        dsp = graph.add_node(NodeType.DSP, effect=self.effect_index[function])
        return graph.add_node(NodeType.SEQ, children=(arguments, dsp))

    def batch(self, graphs: Sequence[BoxGraph]) -> GraphBatch:
        """Pack graphs into one padded disjoint graph.

        Node and edge counts round up to powers of two so compiled shapes are
        reused, leaving at least one padding node.
        """
        node_count = _bucket(sum(g.num_nodes for g in graphs) + 1)
        edge_count = _bucket(sum(len(g.edges) for g in graphs))
        pad = node_count - 1
        node_types = np.full(node_count, NodeType.BUS, np.int32)
        node_values = np.zeros(node_count, np.float32)
        node_effects = np.zeros(node_count, np.int32)
        node_int_tables = np.full(node_count, -1, np.int32)
        node_graphs = np.full(node_count, len(graphs), np.int32)
        edges = np.array([[pad, pad, 0]] * edge_count, np.int32)
        node_offset = edge_offset = 0
        for i, graph in enumerate(graphs):
            span = slice(node_offset, node_offset + graph.num_nodes)
            node_types[span] = graph.node_types
            node_values[span] = graph.node_values
            node_effects[span] = graph.node_effects
            node_int_tables[span] = graph.node_int_tables
            node_graphs[span] = i
            if graph.edges:
                block = np.asarray(graph.edges, np.int32)
                block[:, :2] += node_offset
                edges[edge_offset : edge_offset + len(block)] = block
            node_offset += graph.num_nodes
            edge_offset += len(graph.edges)
        return GraphBatch(
            node_types,
            node_values,
            node_effects,
            node_int_tables,
            node_graphs,
            edges[:, 0],
            edges[:, 1],
            edges[:, 2],
        )


def parameter_spans(code: str) -> list[tuple[int, int]]:
    """Character spans of the numeric arguments of every call in a program."""
    encoded = code.encode()
    spans = []

    def walk(node) -> None:
        named = _named(node)
        if node.type == "function_call" and len(named) > 1 and named[1].type == "arguments":
            for arg in _named(named[1]):
                if arg.type in ("int", "real", "unary_number"):
                    spans.append(
                        (
                            len(encoded[: arg.start_byte].decode()),
                            len(encoded[: arg.end_byte].decode()),
                        )
                    )
            return
        for child in node.children:
            walk(child)

    walk(_parser().parse(encoded).root_node)
    return spans


def _bucket(count: int) -> int:
    return max(64, 1 << (count - 1).bit_length())


@cache
def _parser():
    from tree_sitter import Language, Parser
    from tree_sitter_faust import language

    return Parser(Language(language()))


def _named(node) -> list:
    return [child for child in node.children if child.is_named]


def _text(node) -> str:
    return node.text.decode()
