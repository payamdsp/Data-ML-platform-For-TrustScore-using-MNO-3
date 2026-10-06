"""Chain traversal, shared by the account and customer stages.

Both stages have the same shape. Accounts chain lifecycles together through
number changes; customers chain accounts together through ports. In each case
the edges form paths - one predecessor, one successor - and the job is to find
each path's root, number the positions along it, and label the ends.

Written as pure Python over a collected edge list rather than as an iterative
Spark join. Edges only exist where a number change or a port happened, which is
a small fraction of the input, so the list is small enough to collect; and the
two things that go wrong in this traversal - cycles, and runaway depth - are far
easier to detect and report here than inside a fixed-point join.

That first clause is the load-bearing one, and it is worth being precise about
what it does and does not say. A small fraction of a nightly slice of a few
thousand phone numbers is a handful of edges. A small fraction of the entire
history is not small in any sense that matters to a driver, and the first
full-history run of the account stage killed the Glue session outright on this
code. So the callers now bound what they hand over -
:data:`~.common.MAX_COLLECTED_EDGES` - and refuse in one legible line rather
than dying ten minutes later in a traceback that never mentions Spark.

What has *not* changed is that this is the right shape for the problem, and the
reason is what the customer stage went on to use. Every decision below is local
to a connected component: which of two conflicting edges to drop depends only on
the other edges touching those nodes, and where to break a cycle depends only on
that cycle. Nothing here reads across components. So the way to scale this is not
to rewrite the traversal - it is to label components in Spark and run this
function unchanged, once per component, inside a ``groupBy``.

That is no longer hypothetical. The bound was hit for real on 20 August 2026,
with more than five million port edges, and ``customers._resolve_positions`` now
does exactly that: :func:`~.common.component_labels` puts a component id on every
node and this function runs per group on the executors, with only the repairs
coming back. The account stage did the same on the same day and for the same
reason - its driver died twice in notebook 04 on an edge list of 3,962,773 rows -
so ``accounts._resolve_positions`` also labels components in Spark, groups on the
label and runs ``_resolve_component`` on an executor once per component. Neither
stage collects its edge list any more.

What still comes to the driver from both is the repairs - the broken cycles, the
truncated roots and the edges that lost a conflict - because the report quotes
the pairs rather than counting them, and those are pathologies rather than a
population: hundreds against millions of edges.
:data:`~.common.MAX_COLLECTED_EDGES` bounds that list, which is the only list
left the driver has to hold. The truncation tail is the exception and is counted
rather than collected, being a population and not a pathology.

The labelling that makes it possible is iterative, which is a hazard of its own
and cost a second run the same day: each round has to be *checkpointed* rather
than cached, or the logical plan carries a copy of every round and the driver
spends the job analysing it instead of running it. That is
:func:`~.common.truncate_lineage`.

Both failures are reported rather than swallowed. The previous code did neither:
a cycle simply had no root, so a root-based walk never visited it and the
lifecycles fell through to being unlinked singles with no signal that anything
had happened; and a chain past the depth cap was truncated into the same
silence.

The truncation half of that was only half-fixed, and the August 2026 correctness
audit is what found the other half. ``truncated_roots`` named the 38 roots that
hit the cap on the 24 August 2026 rebuild, but the 1,596 lifecycles hanging off
the far side of those cuts were reported nowhere at all, so the account funnel
did not close: 3,962,770 edges minus 6,268 dropped minus 217 broken left a
residual of exactly 1,596 that nothing in the report line accounted for. Those
lifecycles now carry ``ChainPosition.truncated_tail`` and are counted in
:attr:`ChainResult.truncated_tail_nodes`, which is what makes the subtraction
visible to whoever reads the log next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Sequence

__all__ = ["ChainPosition", "ChainResult", "build_chains"]


@dataclass(frozen=True)
class ChainPosition:
    """Where one node sits in its chain."""

    root: Hashable
    seq_order: int
    chain_length: int
    truncated: bool = False
    #: This node is not a chain of one, it is the part of a chain the walk gave
    #: up on. The depth cap stopped the walk partway down a path and everything
    #: past the cut has a predecessor that was never visited, so the traversal
    #: knows the node is mid-chain and knows nothing else about it.
    #:
    #: It is a separate flag rather than an inference from
    #: ``chain_length == 1 and truncated`` because those two are also true of the
    #: single kept node of a chain cut at ``max_depth=1``, which is the opposite
    #: case - the head of a prefix we did resolve, not the debris of one we did
    #: not. The August 2026 audit is a lesson in how far a value that is almost
    #: right travels before anybody notices, so the distinction is carried
    #: explicitly rather than reconstructed.
    truncated_tail: bool = False

    @property
    def is_single(self) -> bool:
        """A genuine chain of one: nothing joined to it and nothing lost.

        A truncation tail is deliberately excluded. It is a chain of one only in
        the sense that we could not find the rest of it, and 1,596 lifecycles
        published as clean standalone accounts on 24 August 2026 because this
        property could not tell the two apart.
        """
        return self.chain_length == 1 and not (self.truncated or self.truncated_tail)

    @property
    def is_first(self) -> bool:
        return self.seq_order == 0 and not self.truncated_tail

    @property
    def is_last(self) -> bool:
        """The end of the chain, and known to be the end.

        Neither a truncation tail nor the last node of a cut prefix qualifies:
        both have a successor the walk did not reach, so calling either one the
        end of its chain is a claim the data contradicts.
        """
        return (
            self.seq_order == self.chain_length - 1
            and not (self.truncated or self.truncated_tail)
        )


@dataclass
class ChainResult:
    """Positions for every node, plus what had to be repaired to get them."""

    positions: dict[Hashable, ChainPosition]
    broken_cycle_edges: list[tuple[Hashable, Hashable]]
    truncated_roots: list[Hashable]
    dropped_edges: list[tuple[Hashable, Hashable]]
    kept_edges: list[tuple[Hashable, Hashable]] = field(default_factory=list)
    #: The nodes the depth-capped walk abandoned, one entry each. Reported for
    #: the same reason the dropped and broken edges are: it is the only term that
    #: closes the funnel. A caller can subtract
    #: ``len(dropped_edges) + len(broken_cycle_edges) + len(truncated_tail_nodes)``
    #: from the edge count and land on the kept-edge count exactly, which nobody
    #: could do before August 2026 because this list did not exist.
    truncated_tail_nodes: list[Hashable] = field(default_factory=list)

    def __getitem__(self, node: Hashable) -> ChainPosition:
        return self.positions[node]

    @property
    def clean(self) -> bool:
        return not (
            self.broken_cycle_edges
            or self.truncated_roots
            or self.dropped_edges
            or self.truncated_tail_nodes
        )


def build_chains(
    nodes: Iterable[Hashable],
    edges: Sequence[tuple[Hashable, Hashable, Any]],
    max_depth: int = 50,
) -> ChainResult:
    """Resolve ``edges`` into chains over ``nodes``.

    ``edges`` are ``(predecessor, successor, ordering_key)``. The ordering key is
    the transition's timestamp; it is what decides which edge to keep when a
    node has more than one, and which edge to break when a cycle is found.

    Every node appears in the result. A node with no edges is a chain of one -
    which is the point of using one recipe for chains and singles: an account
    that grows from one lifecycle to two keeps its identity, because a single
    was always just a chain of length one.
    """
    node_list = list(dict.fromkeys(nodes))
    node_set = set(node_list)

    # Keep the earliest edge when a node has several successors or several
    # predecessors. More than one means the evidence disagrees with itself - a
    # phone number that appears to have changed to two different numbers - and
    # the earliest transition is the one the rest of the timeline is consistent
    # with. The rejected edges are reported, not dropped silently.
    dropped: list[tuple[Hashable, Hashable]] = []
    ordered = sorted(
        (e for e in edges if e[0] in node_set and e[1] in node_set and e[0] != e[1]),
        key=lambda e: (e[2] is None, e[2], str(e[0]), str(e[1])),
    )
    successor: dict[Hashable, Hashable] = {}
    predecessor: dict[Hashable, Hashable] = {}
    #: The ordering key of each surviving edge, kept so that a cycle can be
    #: broken at its latest transition rather than at an arbitrary node.
    edge_key: dict[Hashable, Any] = {}
    for src, dst, key in ordered:
        if src in successor or dst in predecessor:
            dropped.append((src, dst))
            continue
        successor[src] = dst
        predecessor[dst] = src
        edge_key[src] = key

    # Cycles. A -> B -> A happens when a subscriber changes number and then
    # changes back, and it has no root, so a root-based walk would never visit
    # it. Break the cycle at its latest edge: the earlier transitions are the
    # ones that built the chain, the last one is the one that closed it.
    broken: list[tuple[Hashable, Hashable]] = []
    seen_global: set[Hashable] = set()
    for node in node_list:
        if node in seen_global:
            continue
        path: list[Hashable] = []
        seen_local: dict[Hashable, int] = {}
        current: Hashable | None = node
        while current is not None and current not in seen_global:
            if current in seen_local:
                cycle = path[seen_local[current] :]
                # Latest by the transition's own timestamp, with the node as the
                # tie-break so that two transitions in the same second cannot
                # break differently on different runs. An undated edge sorts
                # first, so a dated edge is always the one broken in preference
                # to it.
                def _break_rank(pair: tuple[Hashable, Hashable]):
                    key = edge_key.get(pair[0])
                    return (key is not None, key if key is not None else 0, str(pair[0]))

                latest = max(
                    ((n, successor[n]) for n in cycle if n in successor),
                    key=_break_rank,
                )
                del successor[latest[0]]
                predecessor.pop(latest[1], None)
                broken.append(latest)
                break
            seen_local[current] = len(path)
            path.append(current)
            current = successor.get(current)
        seen_global.update(path)

    roots = [n for n in node_list if n not in predecessor]

    positions: dict[Hashable, ChainPosition] = {}
    truncated_roots: list[Hashable] = []
    kept: list[tuple[Hashable, Hashable]] = []
    for root in roots:
        chain: list[Hashable] = []
        current: Hashable | None = root
        truncated = False
        while current is not None:
            if len(chain) >= max_depth:
                truncated = True
                truncated_roots.append(root)
                break
            chain.append(current)
            current = successor.get(current)
        length = len(chain)
        for index, node in enumerate(chain):
            positions[node] = ChainPosition(root, index, length, truncated)
        # The edges that actually made it into a chain. Callers need these to
        # find each node's incoming and outgoing transition, and it is only here
        # - after conflicting edges were dropped, cycles broken and the depth cap
        # applied - that the surviving set is known.
        kept.extend(zip(chain, chain[1:]))

    # Anything the walk did not reach sits past a cut the depth cap made, so it
    # is part of a chain rather than a chain of its own. It is still given a
    # position of its own - the node has to appear in the result and there is no
    # honest root to give it - but it is flagged ``truncated_tail`` and listed
    # below, and both of those exist because of what the August 2026 audit found
    # when they did not.
    #
    # The comment that stood here claimed these nodes were "reported through
    # ``dropped_edges`` so it is not invisible". No code ever appended them to
    # ``dropped``, and the consequence was not merely a stale comment: the
    # account funnel for the 24 August 2026 rebuild left a residual of exactly
    # 1,596 lifecycles that no reported quantity explained, across 38 truncated
    # chains averaging ~92 lifecycles each. They were invisible in the report and
    # they published as clean single accounts. The claim is now true because the
    # list below makes it true, rather than because the sentence says so.
    truncated_tail_nodes: list[Hashable] = []
    for node in node_list:
        if node not in positions:
            truncated_tail_nodes.append(node)
            positions[node] = ChainPosition(node, 0, 1, True, truncated_tail=True)

    return ChainResult(
        positions=positions,
        broken_cycle_edges=broken,
        truncated_roots=truncated_roots,
        dropped_edges=dropped,
        kept_edges=kept,
        truncated_tail_nodes=truncated_tail_nodes,
    )
