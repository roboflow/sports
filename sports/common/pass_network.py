"""Pass network: collaboration aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sports.common.passes import InferredPass, InferredTurnover


@dataclass(frozen=True)
class CollaborationLink:
    """Directed pass count between two teammates."""

    passer_tid: int
    receiver_tid: int
    team: int
    count: int


@dataclass(frozen=True)
class PlayerPassSummary:
    """Pass activity for one tracked player."""

    tracker_id: int
    team: int
    passes_made: int
    passes_received: int


@dataclass(frozen=True)
class PassNetwork:
    """Full v1 pass-interaction snapshot for a sequence."""

    sequence: str
    metric: bool
    n_passes: int
    n_turnovers: int
    passes: tuple[InferredPass, ...]
    turnovers: tuple[InferredTurnover, ...]
    links: tuple[CollaborationLink, ...]
    players: tuple[PlayerPassSummary, ...]


def build_collaboration_links(events: list[InferredPass]) -> list[CollaborationLink]:
    """Count directed A -> B passes per link."""
    counts: dict[tuple[int, int, int], int] = defaultdict(int)

    for event in events:
        key = (event.passer_tid, event.receiver_tid, event.team)
        counts[key] += 1

    links: list[CollaborationLink] = []
    for (passer, receiver, team), count in counts.items():
        links.append(
            CollaborationLink(
                passer_tid=passer,
                receiver_tid=receiver,
                team=team,
                count=count,
            )
        )
    links.sort(key=lambda link: link.count, reverse=True)
    return links


def strongest_collaboration_pair(
    links: list[CollaborationLink] | tuple[CollaborationLink, ...],
) -> tuple[int, int, int, int] | None:
    """Undirected pair with the most passes between them: (tid_a, tid_b, team, count)."""
    totals: dict[tuple[int, int, int], int] = defaultdict(int)
    for link in links:
        a, b = sorted((link.passer_tid, link.receiver_tid))
        totals[(a, b, link.team)] += link.count
    if not totals:
        return None
    (a, b, team), count = max(totals.items(), key=lambda item: item[1])
    return a, b, team, count


def build_player_summaries(events: list[InferredPass]) -> list[PlayerPassSummary]:
    """Per-player pass counts as passer vs receiver."""
    teams: dict[int, int] = {}
    made_counts: dict[int, int] = defaultdict(int)
    recv_counts: dict[int, int] = defaultdict(int)

    for event in events:
        teams[event.passer_tid] = event.team
        teams[event.receiver_tid] = event.team
        made_counts[event.passer_tid] += 1
        recv_counts[event.receiver_tid] += 1

    summaries: list[PlayerPassSummary] = []
    for tid in sorted(teams):
        summaries.append(
            PlayerPassSummary(
                tracker_id=tid,
                team=teams[tid],
                passes_made=made_counts[tid],
                passes_received=recv_counts[tid],
            )
        )
    summaries.sort(key=lambda row: row.passes_made + row.passes_received, reverse=True)
    return summaries


def build_pass_network(
    sequence_name: str,
    events: list[InferredPass],
    turnovers: list[InferredTurnover] | None = None,
    *,
    metric: bool,
) -> PassNetwork:
    """Build the v1 collaboration snapshot from inferred pass events."""
    turnovers = turnovers or []
    links = build_collaboration_links(events)
    players = build_player_summaries(events)
    return PassNetwork(
        sequence=sequence_name,
        metric=metric,
        n_passes=len(events),
        n_turnovers=len(turnovers),
        passes=tuple(events),
        turnovers=tuple(turnovers),
        links=tuple(links),
        players=tuple(players),
    )
