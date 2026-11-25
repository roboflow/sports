from typing import Dict, List, Optional, Union, Iterable

Value = Union[int, str, None]


class ConsecutiveValueTracker:
    """
    Validate a property per tracker_id after N consecutive identical reports.
    Once validated, the assignment stays fixed until reset.
    """

    def __init__(self, n_consecutive: int):
        if n_consecutive < 1:
            raise ValueError("n_consecutive must be >= 1")
        self.n = n_consecutive
        self._streak: Dict[int, int] = {}
        self._last: Dict[int, Optional[str]] = {}
        self._validated: Dict[int, Optional[str]] = {}

    def reset_id(self, tracker_id: int) -> None:
        self._streak.pop(tracker_id, None)
        self._last.pop(tracker_id, None)
        self._validated.pop(tracker_id, None)

    def reset_all(self) -> None:
        self._streak.clear()
        self._last.clear()
        self._validated.clear()

    def _normalize(self, value: Value) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None

    def update(self, tracker_ids: List[int], values: List[Value]) -> List[Optional[str]]:
        if len(tracker_ids) != len(values):
            raise ValueError("tracker_ids and values must have the same length")

        output: List[Optional[str]] = []

        for tid, raw in zip(tracker_ids, values):
            if tid in self._validated and self._validated[tid] is not None:
                output.append(self._validated[tid])
                continue

            val = self._normalize(raw)

            if val is None:
                output.append(None)
                self._streak.setdefault(tid, 0)
                self._last.setdefault(tid, None)
                self._validated.setdefault(tid, None)
                continue

            last = self._last.get(tid)

            if last == val:
                self._streak[tid] = self._streak.get(tid, 0) + 1
            else:
                self._streak[tid] = 1
                self._last[tid] = val

            if self._streak[tid] >= self.n:
                self._validated[tid] = self._last[tid]

            output.append(self._validated.get(tid))
            self._last.setdefault(tid, val)
            self._validated.setdefault(tid, None)

        return output

    def get_validated(
        self, tracker_ids: Union[int, Iterable[int]]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """
        Query validated properties for one or many tracker_ids.

        Args:
            tracker_ids (int | Iterable[int]): Single tracker_id or an
                iterable of tracker_ids.

        Returns:
            Optional[str] | List[Optional[str]]:
                Single validated value if input is an int.
                List of validated values in input order if iterable.
        """
        if isinstance(tracker_ids, int):
            return self._validated.get(tracker_ids)
        return [self._validated.get(tid) for tid in tracker_ids]

    def validated_dict(self) -> Dict[int, str]:
        """
        Snapshot of validated assignments. Excludes None values.
        """
        return {tid: val for tid, val in self._validated.items() if val is not None}
