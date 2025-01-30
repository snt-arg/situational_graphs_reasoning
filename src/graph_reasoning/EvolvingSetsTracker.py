from collections import defaultdict

class EvolvingSetsTracker:
    def __init__(
        self,
        similarity_threshold=0.8,
        min_consecutive_appearances=2,
        membership_threshold=0.8,
        strength_thr=0.8
    ):
        """
        :param similarity_threshold:  Jaccard threshold for deciding whether
                                      two sets match across consecutive steps.
        :param min_consecutive_appearances: number of consecutive appearances
                                            required before a new set is "confirmed."
        :param membership_threshold:  fraction of weighted appearances needed
                                      for an element to remain in the final set.
                                      (E.g. 0.5 => element must appear in >=50% of
                                      that set's weighted occurrences to remain.)
        :param strength_thr:          after computing stability scores, sets with
                                      score > strength_thr are considered "strong."
        """
        # Core tracking
        self.history = []              # full timeline: list of lists-of-sets
        self.set_id_by_timestep = []   # parallel structure for storing ID of each set
        self.current_max_id = 0        # next new confirmed ID to assign

        # Thresholds / parameters
        self.similarity_threshold = similarity_threshold
        self.min_consecutive_appearances = min_consecutive_appearances
        self.membership_threshold = membership_threshold
        self.strength_thr = strength_thr

        # Tentative sets storage
        # negative IDs => "tentative" sets not yet confirmed
        self.tentative_sets = {}
        self.next_tentative_id = -1

        # Keep track of each confirmed set’s first & last appearance
        self.first_appearance = {}
        self.last_appearance = {}

    # --- BASIC UTILITIES ---
    def _jaccard_similarity(self, a, b):
        intersect = len(a.intersection(b))
        union = len(a.union(b))
        return intersect / union if union > 0 else 0.0

    def _compute_time_weights(self):
        """
        Default weighting scheme:
          weight(t) = 2^t, then normalized so sum=1.
        Change this if you want a different weighting approach.
        """
        n = len(self.history)
        if n == 0:
            return []
        raw_weights = [2**i for i in range(n)]
        total_weight = sum(raw_weights)
        return [w / total_weight for w in raw_weights]

    # --- MAIN UPDATE METHOD ---
    def add_observation(self, sets_for_this_time):
        """
        Add a new time step (list of sets) to the history.
        1) Matches sets with previous step’s sets (if any), first to confirmed IDs, then to tentative.
        2) If no match passes similarity_threshold => create new tentative set.
        3) If a tentative set reappears enough times in a row => confirm it (assign permanent ID).
        4) Discard tentative sets that fail to reappear (orphan them).
        """
        if not isinstance(sets_for_this_time, list):
            raise TypeError("sets_for_this_time must be a list of Python sets")

        t = len(self.history)  # index of this new time step
        self.history.append(sets_for_this_time)
        self.set_id_by_timestep.append([None]*len(sets_for_this_time))

        # If this is the first time step, everything is tentative by definition.
        if t == 0:
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

        # Otherwise, match with the previous step
        prev_t = t - 1
        prev_ids = self.set_id_by_timestep[prev_t]
        prev_sets = self.history[prev_t]

        for i, curr_set in enumerate(sets_for_this_time):
            best_sim = -1
            best_id = None

            # Find best match from previous step
            for j, prev_set in enumerate(prev_sets):
                candidate_id = prev_ids[j]
                if candidate_id is None:
                    continue
                sim = self._jaccard_similarity(curr_set, prev_set)
                if sim > best_sim:
                    best_sim = sim
                    best_id = candidate_id

            if best_sim >= self.similarity_threshold and best_id is not None:
                # We have a good match
                if best_id < 0:
                    # It's a tentative ID
                    tent_info = self.tentative_sets.get(best_id, None)
                    if tent_info:
                        tent_info["consecutive_count"] += 1
                        tent_info["last_timestep"] = t

                        # If it meets min_consecutive_appearances => confirm it
                        if tent_info["consecutive_count"] >= self.min_consecutive_appearances:
                            new_eid = self.current_max_id
                            self.current_max_id += 1

                            # Mark first appearance for the new ID
                            first_t_approx = t - (self.min_consecutive_appearances - 1)
                            # If we want the earliest actual time step, we might
                            # store it in "tent_info" too. Here we assume
                            # first_t_approx is close enough, or you can store
                            # the real first time.

                            self.first_appearance[new_eid] = first_t_approx
                            self.last_appearance[new_eid] = t

                            # Remove old tentative record
                            del self.tentative_sets[best_id]
                            best_id = new_eid
                        else:
                            # remain tentative
                            pass
                else:
                    # It's already a confirmed ID
                    self.last_appearance[best_id] = t

                self.set_id_by_timestep[t][i] = best_id
            else:
                # No good match => new tentative record
                tid = self.next_tentative_id
                self.next_tentative_id -= 1
                self.tentative_sets[tid] = {
                    "elements": curr_set,
                    "consecutive_count": 1,
                    "last_timestep": t
                }
                self.set_id_by_timestep[t][i] = tid

        # Clean up orphaned tentative sets
        to_remove = []
        for tid, info in self.tentative_sets.items():
            if info["last_timestep"] < t:
                # This set didn't appear in this time step => discard it
                to_remove.append(tid)
        for tid in to_remove:
            del self.tentative_sets[tid]

    # --- FINAL PROCESSING ---
    def postprocess(self):
        """
        Compute the stability and membership of each confirmed set, returning:
            strong_results, results
        Where strong_results are those with average stability > self.strength_thr,
        and results is all sets sorted descending by stability.

        Steps:
         1) Compute time weights.
         2) Collect appearances of each *confirmed* ID.
         3) For each ID, consider only time steps from [first_appearance..last_appearance].
         4) Compute (a) Weighted presence of each element
                    (b) Total or average stability
         5) Filter final membership to elements that appear in >= membership_threshold fraction
            of that set's weighted appearances. This removes ephemeral “bad” elements.
        """
        n = len(self.history)
        if n == 0:
            return [], []

        time_weights = self._compute_time_weights()

        # Gather appearances by ID
        # evolving_sets[eid] = list of (t, set_of_elements)
        evolving_sets = defaultdict(list)

        for t, timestep_sets in enumerate(self.history):
            for i, s in enumerate(timestep_sets):
                eid = self.set_id_by_timestep[t][i]
                # collect data only for confirmed IDs (eid >= 0)
                if eid is not None and eid >= 0:
                    evolving_sets[eid].append((t, s))

        results = []
        strong_results = []

        for eid, appearances in evolving_sets.items():
            # figure out first & last step for this set
            if eid not in self.first_appearance:
                first_t = min(a[0] for a in appearances)
                self.first_appearance[eid] = first_t
            else:
                first_t = self.first_appearance[eid]

            if eid not in self.last_appearance:
                last_t = max(a[0] for a in appearances)
                self.last_appearance[eid] = last_t
            else:
                last_t = self.last_appearance[eid]

            # In case there's some mismatch, ensure consistent ordering
            if last_t < first_t:
                last_t = first_t

            # Weighted presence count
            element_weight_sum = defaultdict(float)
            # We'll also track the sum of weights across [first_t..last_t]
            total_relevant_weight = 0.0

            # We'll store how many "weighted times" we see the set itself
            # so we can compute an average-based stability.
            # The sum_of_weighted_set_appearances = sum of weights(t)
            # for all t in which the set appears. This is our denominator
            # for computing the final stability score (if we want average).
            sum_of_weighted_set_appearances = 0.0

            # Only consider time steps in [first_t..last_t].
            # If it didn't appear at a particular step within that range,
            # we skip it—but that step's weight is still "in the range."
            # This might or might not factor into penalizing "missing" steps;
            # for your approach, you might prefer not to penalize them.
            for (t_idx, s) in appearances:
                if first_t <= t_idx <= last_t:
                    w = time_weights[t_idx]
                    sum_of_weighted_set_appearances += w
                    # Add each element's presence
                    for e in s:
                        element_weight_sum[e] += w

            # Now, compute final membership by discarding ephemeral elements
            # i.e., require element_weight_sum[e] / sum_of_weighted_set_appearances
            # to be >= membership_threshold.
            # If sum_of_weighted_set_appearances is 0, it means no appearances in range => empty set
            if sum_of_weighted_set_appearances <= 0:
                final_elements = set()
            else:
                final_elements = set()
                for e, wsum in element_weight_sum.items():
                    ratio = wsum / sum_of_weighted_set_appearances
                    if ratio >= self.membership_threshold:
                        final_elements.add(e)

            # Compute stability score
            # The user’s formula was: (sum of element-weight-sums) / (num distinct elements)
            # but we might adapt it for ephemeral membership. We'll define:
            #   total_weighted_presence = sum(element_weight_sum[e] for e in final_elements).
            if len(final_elements) == 0:
                avg_stability = 0.0
            else:
                total_weighted_presence = sum(element_weight_sum[e] for e in final_elements)
                avg_stability = total_weighted_presence / len(final_elements)

            results.append((eid, avg_stability, final_elements))

            if avg_stability > self.strength_thr:
                strong_results.append((eid, avg_stability, final_elements))

        # Sort by descending stability
        results.sort(key=lambda x: x[1], reverse=True)
        strong_results.sort(key=lambda x: x[1], reverse=True)

        return strong_results, results
