from collections import defaultdict
import logging
from typing import List, Set    


class EvolvingSetsTracker:
    """
    Incremental tracker for a stream of *sets* that

    • creates a new ID the very first time a set is seen  
    • keeps a continuous `confidence` ∈ [0, 1] that rises when the set
      is matched, falls when it is missed, and purges the set at 0  
    • smooths element membership with consecutive‑appearance / consecutive‑absence
      thresholds (`elem_add_threshold`, `elem_remove_threshold`)  

    You will usually:

        tracker = EvolvingSetsTracker()
        tracker.add_observation([set1, set2, ...])   # each time step
        strong, all_ = tracker.postprocess()         # whenever you need results
    """

    # ---------- constructor -------------------------------------------------
    def __init__(
        self,
        *,
        similarity_threshold: float = 0.8,
        init_confidence: float = 0.20,
        up_rate: float = 0.15,
        down_rate: float = 0.10,
        strength_thr: float = 0.0,
        elem_add_threshold: int = 2,
        elem_remove_threshold: int = 2,
        logger: None,
    ):
        use_stable_params = False  # set to True to use stable parameters
        if use_stable_params:
            # stable parameters from the original implementation
            similarity_threshold = 0.9
            init_confidence = 0.5
            up_rate = 0.25
            down_rate = 0.05
            strength_thr = 0.5
            elem_add_threshold = 4
            elem_remove_threshold = 4
        else:
            # use the parameters provided in the constructor
            pass

        # parameters
        self.similarity_threshold = similarity_threshold
        self.init_confidence = init_confidence
        self.up_rate = up_rate
        self.down_rate = down_rate
        self.strength_thr = strength_thr
        self.elem_add_threshold = elem_add_threshold
        self.elem_remove_threshold = elem_remove_threshold

        # timeline storage
        self.history: list[list[set]] = []
        self.set_id_by_timestep: list[list[int | None]] = []

        # ID bookkeeping
        self.current_max_id = 0
        self.confidence = defaultdict(float)      # eid -> confidence ∈ [0,1]
        self.first_appearance = {}                # eid -> timestep
        self.last_appearance = {}                 # eid -> timestep
        self.removed_ids: set[int] = set()

        # element–level membership state
        # membership_state[eid][elem] = {in_count, out_count, is_member}
        self.membership_state = defaultdict(
            lambda: defaultdict(
                lambda: {"in_count": 0, "out_count": 0, "is_member": False}
            )
        )

        # logging
        self.logger = logger or logging.getLogger(__name__)

    # ---------- utility -----------------------------------------------------
    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    # ---------- incremental update ------------------------------------------
    def add_observation(self, sets_this_step: List[Set]):
        """
        Feed one time‑step worth of observations (list of Python sets).
        Creates new IDs immediately, pre‑loads their elements into
        membership (provisional), and updates confidence / membership.
        """
        if not isinstance(sets_this_step, list):
            raise TypeError("add_observation expects a *list* of sets")

        t = len(self.history)
        self.history.append(sets_this_step)
        self.set_id_by_timestep.append([None] * len(sets_this_step))

        # -------- helper lookup to previous step ----------------------------
        prev_ids = self.set_id_by_timestep[t - 1] if t > 0 else []
        prev_sets = self.history[t - 1]           if t > 0 else []

        appeared_ids: set[int] = set()            # IDs matched this step

        # -------- loop over current sets ------------------------------------
        for i, curr_set in enumerate(sets_this_step):
            best_sim = -1.0
            best_id: int | None = None

            # try to match to each set from previous step
            for j, prev_set in enumerate(prev_sets):
                cand_id = prev_ids[j]
                if cand_id is None or cand_id in self.removed_ids:
                    continue
                sim = self._set_similarity(curr_set, prev_set)
                if sim > best_sim:
                    best_sim, best_id = sim, cand_id

            # ----------------------------------------------------------------
            if best_id is not None and best_sim >= self.similarity_threshold:
                # matched an existing ID
                eid = best_id
                appeared_ids.add(eid)
                self.last_appearance[eid] = t
                self.set_id_by_timestep[t][i] = eid

            else:
                # -------- create a brand‑new ID -----------------------------
                eid = self.current_max_id
                self.current_max_id += 1

                self.first_appearance[eid] = t
                self.last_appearance[eid]  = t
                self.confidence[eid]       = self.init_confidence
                self.set_id_by_timestep[t][i] = eid
                appeared_ids.add(eid)

                # *** PRE‑LOAD every element so it appears immediately ****
                for e in curr_set:
                    st = self.membership_state[eid][e]
                    st["in_count"]  = 1
                    st["out_count"] = 0
                    st["is_member"] = True

            # update element membership for this eid with the current set
            self._update_membership(eid, curr_set)

        # ----------- confidence update phase --------------------------------
        for eid in appeared_ids:
            if eid in self.removed_ids:
                continue
            self.confidence[eid] = min(1.0, self.confidence[eid] + self.up_rate)

        for eid in list(self.confidence):
            if eid in self.removed_ids or eid in appeared_ids:
                continue
            self.confidence[eid] = max(0.0, self.confidence[eid] - self.down_rate)
            if self.confidence[eid] == 0.0:
                self.removed_ids.add(eid)

    # ---------- element‑level smoothing ------------------------------------
    def _update_membership(self, eid: int, seen_elems: set):
        """
        Update per‑element in/out counts & is_member flags for a set ID
        based on the elements *seen in this step*.
        """
        state = self.membership_state[eid]

        # increment in_count for elements we *just* saw
        for e in seen_elems:
            st = state[e]
            st["in_count"] += 1
            st["out_count"] = 0
            if not st["is_member"] and st["in_count"] >= self.elem_add_threshold:
                st["is_member"] = True

        # increment out_count for elements we *didn't* see
        for e in list(state):               # list() to avoid dict‑size change
            if e in seen_elems:
                continue
            st = state[e]
            st["out_count"] += 1
            st["in_count"] = 0
            if st["is_member"] and st["out_count"] >= self.elem_remove_threshold:
                st["is_member"] = False

    # ---------- query / results --------------------------------------------
    def postprocess(self):
        """
        Returns:
            strong_sets : list[(eid, confidence, members)]
            all_sets    : list[(eid, confidence, members)]

        *strong_sets* are those with confidence > `strength_thr`,
        both lists sorted by descending confidence.
        """
        strong, all_sets = [], []

        for eid, conf in self.confidence.items():
            if eid in self.removed_ids:
                continue

            members = {
                e
                for e, st in self.membership_state[eid].items()
                if st["is_member"]
            }
            all_sets.append((eid, conf, members))
            if conf > self.strength_thr:
                strong.append((eid, conf, members))

        all_sets.sort(key=lambda x: x[1], reverse=True)
        strong.sort(key=lambda x: x[1], reverse=True)

        self.logger.info(
            f"dbg strong {strong} all_sets {all_sets} )"
        )
        return strong, all_sets

    def _set_similarity(self, a: set, b: set) -> float:
        """max( Jaccard , overlap‑coefficient )."""
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        jaccard = inter / union if union else 0.0
        overlap = inter / min(len(a), len(b)) if min(len(a), len(b)) else 0.0
        return max(jaccard, overlap)