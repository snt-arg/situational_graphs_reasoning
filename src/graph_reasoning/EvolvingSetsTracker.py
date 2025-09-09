# GREEDY

from collections import defaultdict

class EvolvingSetsTracker:
    def __init__(
        self,
        similarity_threshold=0.8,
        min_consecutive_appearances=2,
        max_missing_steps=2,
        elem_add_threshold=2,
        elem_remove_threshold=2,
        strength_thr=0.5
    ):
        """
        :param similarity_threshold:   Jaccard threshold for matching sets across steps.
        :param min_consecutive_appearances: consecutive steps to confirm a new set (tentative->confirmed).
        :param max_missing_steps:      consecutive absences of a confirmed set before removing it entirely.
        :param elem_add_threshold:     consecutive appearances required to accept an element into set membership.
        :param elem_remove_threshold:  consecutive absences required to remove an element from set membership.
        :param strength_thr:           final stability threshold to classify "strong" sets in postprocess().

        Notes:
          - Negative IDs => "tentative" sets. Once confirmed, assigned a non-negative ID.
          - We store per-element state for each confirmed set, so membership is only updated
            after crossing these consecutive thresholds.
        """

        # Core timeline tracking
        self.history = []            # All time steps: list of (list-of-sets)
        self.set_id_by_timestep = [] # Parallel: store ID for each set in each time step
        self.current_max_id = 0      # Next new confirmed set ID

        # Global thresholds
        self.similarity_threshold = similarity_threshold
        self.min_consecutive_appearances = min_consecutive_appearances
        self.max_missing_steps = max_missing_steps
        self.elem_add_threshold = elem_add_threshold
        self.elem_remove_threshold = elem_remove_threshold
        self.strength_thr = strength_thr

        # TENTATIVE sets
        self.tentative_sets = {}     # tid -> dict(...)
        self.next_tentative_id = -1

        # CONFIRMED sets
        self.first_appearance = {}
        self.last_appearance = {}
        self.missing_count = defaultdict(int)   # how many consecutive steps a confirmed set is missing

        # REMOVED sets (we won't keep tracking them)
        self.removed_ids = set()

        # For each confirmed set, track a dictionary of elements -> membership state
        # membership_state[eid][element] = {
        #    "in_count": 0,
        #    "out_count": 0,
        #    "is_member": False
        # }
        self.membership_state = defaultdict(lambda: defaultdict(lambda: {
            "in_count": 0,
            "out_count": 0,
            "is_member": False
        }))

    # --- Basic Utilities ---
    def _jaccard_similarity(self, a, b):
        if not a and not b:
            return 1.0
        union_size = len(a.union(b))
        if union_size == 0:
            return 0.0
        intersect_size = len(a.intersection(b))
        return intersect_size / union_size

    # --- Step 1: Add Observations ---
    def add_observation(self, sets_for_this_time):
        """
        Add a new time step (list of sets).
        1) Match sets with previous step (tentative or confirmed) via Jaccard >= similarity_threshold.
        2) If no match, create new tentative set.
        3) If a tentative set meets min_consecutive_appearances => confirm it (assign ID).
        4) For confirmed sets that do not appear => increment missing_count; remove if > max_missing_steps.
        5) For each confirmed set that appears, update element-level states
           (consecutive in/out counts, is_member) with the new observation's elements.
        """
        if not isinstance(sets_for_this_time, list):
            raise TypeError("sets_for_this_time must be a list of sets")

        t = len(self.history)
        self.history.append(sets_for_this_time)
        self.set_id_by_timestep.append([None]*len(sets_for_this_time))

        if t == 0:
            # First observation => all sets are TENTATIVE
            for i, s in enumerate(sets_for_this_time):
                tid = self.next_tentative_id
                self.next_tentative_id -= 1
                self.tentative_sets[tid] = {
                    "elements": s,
                    "consecutive_count": 1,
                    "last_timestep": t
                }
                self.set_id_by_timestep[t][i] = tid
            return

        # We have a previous step
        prev_t = t - 1
        prev_ids = self.set_id_by_timestep[prev_t]
        prev_sets = self.history[prev_t]

        # Track which confirmed sets actually appeared this step (for missing_count logic)
        appeared_confirmed_ids = set()

        # --- Matching each current set ---
        for i, curr_set in enumerate(sets_for_this_time):
            best_sim = -1
            best_id = None

            for j, prev_set in enumerate(prev_sets):
                candidate_id = prev_ids[j]
                if candidate_id is None or candidate_id in self.removed_ids:
                    continue
                sim = self._jaccard_similarity(curr_set, prev_set)
                if sim > best_sim:
                    best_sim = sim
                    best_id = candidate_id

            if best_sim >= self.similarity_threshold and best_id is not None:
                # Good match
                if best_id < 0:
                    # It's a tentative ID
                    tent_info = self.tentative_sets.get(best_id)
                    if tent_info:
                        tent_info["consecutive_count"] += 1
                        tent_info["last_timestep"] = t
                        if tent_info["consecutive_count"] >= self.min_consecutive_appearances:
                            # Confirm it => new ID
                            new_eid = self.current_max_id
                            self.current_max_id += 1
                            del self.tentative_sets[best_id]
                            # set appearances
                            start_t = t - (self.min_consecutive_appearances - 1)
                            self.first_appearance[new_eid] = start_t
                            self.last_appearance[new_eid] = t
                            self.missing_count[new_eid] = 0
                            best_id = new_eid
                        else:
                            # remain tentative
                            pass
                else:
                    # It's confirmed => update last_appearance, reset missing_count
                    self.last_appearance[best_id] = t
                    self.missing_count[best_id] = 0

                if best_id >= 0:
                    appeared_confirmed_ids.add(best_id)

                self.set_id_by_timestep[t][i] = best_id
            else:
                # No match => new TENTATIVE
                tid = self.next_tentative_id
                self.next_tentative_id -= 1
                self.tentative_sets[tid] = {
                    "elements": curr_set,
                    "consecutive_count": 1,
                    "last_timestep": t
                }
                self.set_id_by_timestep[t][i] = tid

        # 1) Remove orphaned tentative sets (didn't appear this step)
        to_remove_tent = []
        for tid, info in self.tentative_sets.items():
            if info["last_timestep"] < t:
                to_remove_tent.append(tid)
        for tid in to_remove_tent:
            del self.tentative_sets[tid]

        # 2) For confirmed sets not in 'appeared_confirmed_ids', increment missing_count
        confirmed_ids = [
            eid for eid in self.first_appearance.keys()
            if eid not in self.removed_ids
        ]
        for cid in confirmed_ids:
            if cid not in appeared_confirmed_ids:
                self.missing_count[cid] += 1
                if self.missing_count[cid] > self.max_missing_steps:
                    self.removed_ids.add(cid)
            else:
                # This set appeared => update element membership states
                # We have to find which (curr_set) matched this set ID
                # so we know which elements to increment in_count
                # (In principle, more than one new set could match the same ID
                #  if best_sim was the same, but let's assume a 1-1 match in this code.)
                # We'll do a second pass to get the union of all sets that matched this ID.
                pass

        # 3) Update per-element membership states for each confirmed set that did appear
        #    i.e., we find the union of sets that matched a given ID in this time step,
        #    then increment in_count for those elements, increment out_count for elements not in it.
        #    If in_count >= elem_add_threshold => is_member=True
        #       out_count >= elem_remove_threshold => is_member=False
        matched_elements_by_id = defaultdict(set)

        # Collect which sets matched each ID
        for i, s in enumerate(sets_for_this_time):
            matched_id = self.set_id_by_timestep[t][i]
            if matched_id is not None and matched_id >= 0 and matched_id not in self.removed_ids:
                matched_elements_by_id[matched_id].update(s)

        # Now update membership states
        for cid in appeared_confirmed_ids:
            if cid in self.removed_ids:
                continue
            # The set of elements that actually appeared for cid at time t
            new_elems = matched_elements_by_id[cid]
            # We consider the union of known elements for this set
            # so we can update out_count for those not present
            tracked_elements = list(self.membership_state[cid].keys())

            # For elements that appear now, increment in_count
            for e in new_elems:
                state = self.membership_state[cid][e]
                state["in_count"] += 1
                state["out_count"] = 0
                # If we cross the threshold => officially add
                if not state["is_member"] and state["in_count"] >= self.elem_add_threshold:
                    state["is_member"] = True

            # For elements we track but didn't appear => increment out_count
            for e in tracked_elements:
                if e not in new_elems:
                    st = self.membership_state[cid][e]
                    st["out_count"] += 1
                    st["in_count"] = 0
                    # If we cross remove threshold => remove
                    if st["is_member"] and st["out_count"] >= self.elem_remove_threshold:
                        st["is_member"] = False

    def _compute_time_weights(self):
        """
        Example: Returns an array of normalized 2^t weights for each time step.
        Modify if you prefer a different scheme.
        """
        n = len(self.history)
        if n == 0:
            return []
        raw_weights = [2**i for i in range(n)]
        total_weight = sum(raw_weights)
        return [w / total_weight for w in raw_weights]


    # --- Step 2: Postprocess / Return Results ---
    def postprocess(self):
        """
        Return (strong_sets, all_sets), where each entry is:
        (eid, trust_score, final_members, details)

        `trust_score` is computed as (WeightedFractionOfAppearances * AvgElementConsistency).
        - WeightedFractionOfAppearances = sum of weights for steps the set actually appeared
                                        / sum of weights in [first_appearance..last_appearance]
        - AvgElementConsistency = average fraction of time each final member was 'is_member=True'
                                within [first_appearance..last_appearance].

        `final_members` is the set of elements currently is_member=True at the end.

        `details` can include sub-scores or anything else you want to return, for clarity.
        """
        n = len(self.history)
        if n == 0:
            return [], []

        # 1) Build time-step weights once, e.g. 2^t normalized.
        time_weights = self._compute_time_weights()

        # 2) We'll gather sets that haven't been removed, i.e. confirmed sets with id >= 0
        confirmed_active_ids = [
            cid for cid in self.first_appearance 
            if cid >= 0 and cid not in self.removed_ids
        ]

        results = []
        strong_results = []

        for cid in confirmed_active_ids:
            t0 = self.first_appearance[cid]
            t1 = self.last_appearance.get(cid, t0)
            if t1 < t0:  # degenerate
                continue

            # ------------------------------
            # (A) Compute WeightedFractionOfAppearances
            # ------------------------------
            # sum of weights in the [t0..t1] window
            window_weight = sum(time_weights[t] for t in range(t0, t1 + 1) if t < n)
            
            # sum of weights for time steps the set was actually observed
            # We'll look at each step [t0..t1]; if the set is missing in that step, we skip
            appear_weight = 0.0

            # Find all time steps where cid actually appeared
            #   => we can find them by scanning self.set_id_by_timestep[t] == cid
            # or we can store an internal mapping as we go. For clarity, let's do a quick scan:
            for t_idx in range(t0, t1 + 1):
                if t_idx >= n:
                    break
                matched_any = False
                for s_id in self.set_id_by_timestep[t_idx]:
                    if s_id == cid:
                        matched_any = True
                        break
                if matched_any:
                    appear_weight += time_weights[t_idx]

            if window_weight > 0:
                fraction_of_appearances = appear_weight / window_weight
            else:
                fraction_of_appearances = 0.0

            # ------------------------------
            # (B) Compute AvgElementConsistency
            # ------------------------------
            # For each element that ever got tracked in membership_state[cid],
            # measure how many steps in [t0..t1] it was `is_member=True`.
            # We'll store that sum in `member_weight_sum[e]`.
            # Then the fraction = sum_for_element / window_weight.
            # We'll average across all final members who are is_member=True at the *end*.
            mem_state = self.membership_state[cid]
            
            # Find the elements that are is_member=True at the end
            final_members = [e for e, st in mem_state.items() if st["is_member"]]
            if not final_members or window_weight == 0:
                avg_elem_consistency = 0.0
            else:
                # We need to know for how many steps (weighted) each final element was is_member=True
                # That requires storing a per-time-step log or re-creating it from in/out counts.
                # For simplicity, let's approximate using in_count / remove_threshold, etc.
                # Alternatively, you can store a "time_series" for each element if you want exact tracking.
                #
                # We'll do a simple approach:
                #   If "in_count" is large, presumably it's been in for multiple consecutive appearances.
                #   This is a rough approximation. For a perfect method, you'd store the actual time steps
                #   it was is_member=True.
                
                # For demonstration, let's approximate that "in_count" is how many consecutive steps
                # they've appeared up to now. We'll define:
                #   element_consistency(e) = min( in_count[e], (t1 - t0 + 1) ) / (t1 - t0 + 1)
                # i.e., fraction of the overall window they've been in, up to a max of the window size.
                
                # A truly accurate approach would require storing a record each time "is_member" flips 
                # from False to True, to know exactly which time steps it was True. 
                # We'll show this simpler approximation for demonstration.
                
                sum_consistency = 0.0
                total_window_steps = (t1 - t0 + 1)
                for e in final_members:
                    st = mem_state[e]
                    approximate_in_steps = min(st["in_count"], total_window_steps)
                    elem_consistency = approximate_in_steps / total_window_steps
                    sum_consistency += elem_consistency

                avg_elem_consistency = sum_consistency / len(final_members)

            # Combine them: trust = fraction_of_appearances * avg_elem_consistency
            trust_score = fraction_of_appearances * avg_elem_consistency

            # Build final membership as a set (for display)
            final_members_set = set(final_members)

            # Collect result (could store sub-scores in 'details')
            details = {
                "fraction_of_appearances": fraction_of_appearances,
                "avg_elem_consistency": avg_elem_consistency,
                "window_weight": window_weight
            }
            results.append((cid, trust_score, final_members_set, details))

        # Sort by descending trust_score
        results.sort(key=lambda x: x[1], reverse=True)

        # Create strong_results if trust > self.strength_thr
        for cid, ts, fm, det in results:
            if ts > self.strength_thr:
                strong_results.append((cid, ts, fm, det))

        return strong_results, results
    

# CONSERVATIVE

# from collections import defaultdict
# import logging
# from typing import List, Set    


# class EvolvingSetsTracker:
#     """
#     Incremental tracker for a stream of *sets* that

#     • creates a new ID the very first time a set is seen  
#     • keeps a continuous `confidence` ∈ [0, 1] that rises when the set
#       is matched, falls when it is missed, and purges the set at 0  
#     • smooths element membership with consecutive‑appearance / consecutive‑absence
#       thresholds (`elem_add_threshold`, `elem_remove_threshold`)  

#     You will usually:

#         tracker = EvolvingSetsTracker()
#         tracker.add_observation([set1, set2, ...])   # each time step
#         strong, all_ = tracker.postprocess()         # whenever you need results
#     """

#     # ---------- constructor -------------------------------------------------
#     def __init__(
#         self,
#         *,
#         similarity_threshold: float = 0.8,
#         init_confidence: float = 0.20,
#         up_rate: float = 0.15,
#         down_rate: float = 0.10,
#         strength_thr: float = 0.0,
#         elem_add_threshold: int = 2,
#         elem_remove_threshold: int = 2,
#         logger: None,
#     ):
#         use_stable_params = False  # set to True to use stable parameters
#         if use_stable_params:
#             # stable parameters from the original implementation
#             similarity_threshold = 0.9
#             init_confidence = 0.5
#             up_rate = 0.25
#             down_rate = 0.05
#             strength_thr = 0.5
#             elem_add_threshold = 4
#             elem_remove_threshold = 4
#         else:
#             # use the parameters provided in the constructor
#             pass

#         # parameters
#         self.similarity_threshold = similarity_threshold
#         self.init_confidence = init_confidence
#         self.up_rate = up_rate
#         self.down_rate = down_rate
#         self.strength_thr = strength_thr
#         self.elem_add_threshold = elem_add_threshold
#         self.elem_remove_threshold = elem_remove_threshold

#         # timeline storage
#         self.history: list[list[set]] = []
#         self.set_id_by_timestep: list[list[int | None]] = []

#         # ID bookkeeping
#         self.current_max_id = 0
#         self.confidence = defaultdict(float)      # eid -> confidence ∈ [0,1]
#         self.first_appearance = {}                # eid -> timestep
#         self.last_appearance = {}                 # eid -> timestep
#         self.removed_ids: set[int] = set()

#         # element–level membership state
#         # membership_state[eid][elem] = {in_count, out_count, is_member}
#         self.membership_state = defaultdict(
#             lambda: defaultdict(
#                 lambda: {"in_count": 0, "out_count": 0, "is_member": False}
#             )
#         )

#         # logging
#         self.logger = logger or logging.getLogger(__name__)

#     # ---------- utility -----------------------------------------------------
#     @staticmethod
#     def _jaccard(a: set, b: set) -> float:
#         if not a and not b:
#             return 1.0
#         u = len(a | b)
#         return len(a & b) / u if u else 0.0

#     # ---------- incremental update ------------------------------------------
#     def add_observation(self, sets_this_step: List[Set]):
#         """
#         Feed one time‑step worth of observations (list of Python sets).
#         Creates new IDs immediately, pre‑loads their elements into
#         membership (provisional), and updates confidence / membership.
#         """
#         if not isinstance(sets_this_step, list):
#             raise TypeError("add_observation expects a *list* of sets")

#         t = len(self.history)
#         self.history.append(sets_this_step)
#         self.set_id_by_timestep.append([None] * len(sets_this_step))

#         # -------- helper lookup to previous step ----------------------------
#         prev_ids = self.set_id_by_timestep[t - 1] if t > 0 else []
#         prev_sets = self.history[t - 1]           if t > 0 else []

#         appeared_ids: set[int] = set()            # IDs matched this step

#         # -------- loop over current sets ------------------------------------
#         for i, curr_set in enumerate(sets_this_step):
#             best_sim = -1.0
#             best_id: int | None = None

#             # try to match to each set from previous step
#             for j, prev_set in enumerate(prev_sets):
#                 cand_id = prev_ids[j]
#                 if cand_id is None or cand_id in self.removed_ids:
#                     continue
#                 sim = self._set_similarity(curr_set, prev_set)
#                 if sim > best_sim:
#                     best_sim, best_id = sim, cand_id

#             # ----------------------------------------------------------------
#             if best_id is not None and best_sim >= self.similarity_threshold:
#                 # matched an existing ID
#                 eid = best_id
#                 appeared_ids.add(eid)
#                 self.last_appearance[eid] = t
#                 self.set_id_by_timestep[t][i] = eid

#             else:
#                 # -------- create a brand‑new ID -----------------------------
#                 eid = self.current_max_id
#                 self.current_max_id += 1

#                 self.first_appearance[eid] = t
#                 self.last_appearance[eid]  = t
#                 self.confidence[eid]       = self.init_confidence
#                 self.set_id_by_timestep[t][i] = eid
#                 appeared_ids.add(eid)

#                 # *** PRE‑LOAD every element so it appears immediately ****
#                 for e in curr_set:
#                     st = self.membership_state[eid][e]
#                     st["in_count"]  = 1
#                     st["out_count"] = 0
#                     st["is_member"] = True

#             # update element membership for this eid with the current set
#             self._update_membership(eid, curr_set)

#         # ----------- confidence update phase --------------------------------
#         for eid in appeared_ids:
#             if eid in self.removed_ids:
#                 continue
#             self.confidence[eid] = min(1.0, self.confidence[eid] + self.up_rate)

#         for eid in list(self.confidence):
#             if eid in self.removed_ids or eid in appeared_ids:
#                 continue
#             self.confidence[eid] = max(0.0, self.confidence[eid] - self.down_rate)
#             if self.confidence[eid] == 0.0:
#                 self.removed_ids.add(eid)

#     # ---------- element‑level smoothing ------------------------------------
#     def _update_membership(self, eid: int, seen_elems: set):
#         """
#         Update per‑element in/out counts & is_member flags for a set ID
#         based on the elements *seen in this step*.
#         """
#         state = self.membership_state[eid]

#         # increment in_count for elements we *just* saw
#         for e in seen_elems:
#             st = state[e]
#             st["in_count"] += 1
#             st["out_count"] = 0
#             if not st["is_member"] and st["in_count"] >= self.elem_add_threshold:
#                 st["is_member"] = True

#         # increment out_count for elements we *didn't* see
#         for e in list(state):               # list() to avoid dict‑size change
#             if e in seen_elems:
#                 continue
#             st = state[e]
#             st["out_count"] += 1
#             st["in_count"] = 0
#             if st["is_member"] and st["out_count"] >= self.elem_remove_threshold:
#                 st["is_member"] = False

#     # ---------- query / results --------------------------------------------
#     def postprocess(self):
#         """
#         Returns:
#             strong_sets : list[(eid, confidence, members)]
#             all_sets    : list[(eid, confidence, members)]

#         *strong_sets* are those with confidence > `strength_thr`,
#         both lists sorted by descending confidence.
#         """
#         strong, all_sets = [], []

#         for eid, conf in self.confidence.items():
#             if eid in self.removed_ids:
#                 continue

#             members = {
#                 e
#                 for e, st in self.membership_state[eid].items()
#                 if st["is_member"]
#             }
#             all_sets.append((eid, conf, members))
#             if conf > self.strength_thr:
#                 strong.append((eid, conf, members))

#         all_sets.sort(key=lambda x: x[1], reverse=True)
#         strong.sort(key=lambda x: x[1], reverse=True)

#         self.logger.info(
#             f"dbg strong {strong} all_sets {all_sets} )"
#         )
#         return strong, all_sets

#     def _set_similarity(self, a: set, b: set) -> float:
#         """max( Jaccard , overlap‑coefficient )."""
#         if not a and not b:
#             return 1.0
#         inter = len(a & b)
#         union = len(a | b)
#         jaccard = inter / union if union else 0.0
#         overlap = inter / min(len(a), len(b)) if min(len(a), len(b)) else 0.0
#         return max(jaccard, overlap)