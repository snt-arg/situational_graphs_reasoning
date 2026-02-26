# Variability Analysis in EvolvingSetsTracker

Everything is now integrated into `EvolvingSetsTracker.py`. No separate files needed.

## Quick Start

### 1. Basic Usage - Run the Example
```bash
cd /path/to/situational_graphs_reasoning/src/graph_reasoning
python EvolvingSetsTracker.py
```

This runs with synthetic data and produces:
- Console report with variability metrics
- `variability_results.json` with detailed results

### 2. Use in Your Code

```python
from EvolvingSetsTracker import (
    EvolvingSetsTrackerConservative, 
    EvolvingSetsTrackerGreedy,
    compare_trackers,
    generate_variability_report
)

# Your observations (list of timesteps, each with list of sets)
observations = [
    [{"elem_1", "elem_2"}, {"elem_3"}],
    [{"elem_1", "elem_2", "elem_4"}, {"elem_3"}],
    # ... more timesteps
]

# Create trackers
cons_tracker = EvolvingSetsTrackerConservative(
    similarity_threshold=0.8,
    min_consecutive_appearances=2,
    max_missing_steps=2,
)

greedy_tracker = EvolvingSetsTrackerGreedy(
    similarity_threshold=0.8,
)

# Compare
results = compare_trackers(observations, cons_tracker, greedy_tracker)

# Generate report
report = generate_variability_report(results)
print(report)
```

## What Gets Measured

### 1. **Membership Churn**
   - How much elements appear/disappear from sets per timestep
   - Lower = more stable membership

### 2. **Set Count Variance**
   - How much the number of active sets fluctuates
   - Coefficient of variation (CV): lower = more stable
   - `CV = std_dev / mean`

### 3. **Element Consistency**
   - How often elements "flicker" in/out of sets
   - 0 = perfect (always in or always out)
   - 1 = maximum flickering (50/50)

## Output Metrics

The comparison returns:

```json
{
  "conservative": {
    "membership_churn": {...},      // How much membership changes
    "set_count_variance": {...},    // Set count stability
    "element_consistency": {...},   // Element membership stability
    "total_timesteps": 10,
    "final_set_count": 3
  },
  "greedy": { ... },  // Same metrics for greedy baseline
  "improvement": {
    "churn_reduction_percent": 37.0,
    "set_count_stability_improvement_percent": 15.3,
    "element_consistency_improvement_percent": 22.1
  }
}
```

## Interpretation

- **Positive % values** = Conservative is better (more stable)
- **Negative % values** = Greedy is better (more stable)

### Example Results
```
Conservative reduces membership churn by 37.0%
  → 37% fewer element changes per timestep
  
Conservative reduces set count variability by 15.3%
  → More stable number of active sets
  
Conservative improves element consistency by 22.1%
  → Less flickering of elements in/out
```

## Available Classes

### `EvolvingSetsTrackerConservative`
- **Requires consecutive confirmations** before creating new sets
- **min_consecutive_appearances**: How many steps to confirm
- **max_missing_steps**: How long to keep without seeing
- **Result**: More stable, fewer spurious sets

### `EvolvingSetsTrackerGreedy`
- **Creates sets immediately** when first seen
- **No tentative phase**
- **Result**: More sets, more variability (baseline)

### `VariabilityAnalyzer`
Static methods to analyze any tracker:
- `measure_membership_churn()`
- `measure_set_count_variance()`
- `measure_element_consistency()`

## Functions

- `compare_trackers()` - Run both on same data and compare
- `extract_membership_timeline()` - Get membership per timestep
- `generate_variability_report()` - Pretty-print results

---

All code is in: [EvolvingSetsTracker.py](EvolvingSetsTracker.py)
