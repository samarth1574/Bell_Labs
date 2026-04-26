# Experiment Log

| Date | Experiment | Config | Result | Notes |
|------|-----------|--------|--------|-------|

## Log Entries

### [Date] - Experiment Name
- **Goal:**
- **Setup:**
- **Result:**
- **Conclusion:**

---
## Experiment: Baseline Comparison — 20260322_022639
- **Date**: 2026-03-22 02:26:39
- **Type**: Baseline method comparison (all classical methods)
- **Dataset**: Synthetic overlapping shapes (4 occlusion levels)
- **Methods**: Heuristic, Watershed, GraphSeg, RetailPrior

### Configuration
- `Heuristic`: HeuristicDetector(block_size=11, C=2, min_area=100, max_area_ratio=0.5)
- `Watershed`: WatershedSegmenter(blur_ksize=7, min_distance=10, min_area=80)
- `GraphSeg`: GraphSegmenter(scale=200, sigma=0.5, min_size=50)
- `RetailPrior`: RetailPriorDetector(hough_thresh=50, strip_width=30, fallback_scale=150)

### Overall Results

| method      |   count_mae |   precision |   recall |    f1 |   mean_iou |   avg_time_ms |
|:------------|------------:|------------:|---------:|------:|-----------:|--------------:|
| GraphSeg    |       15.1  |       0.451 |    0.612 | 0.488 |      0.838 |        52.365 |
| Heuristic   |       23.45 |       0.424 |    0.185 | 0.223 |      0.531 |         2.68  |
| RetailPrior |       15    |       0.463 |    0.614 | 0.497 |      0.834 |        54.68  |
| Watershed   |       22.7  |       0.672 |    0.156 | 0.222 |      0.703 |         2.675 |

### Key Findings
- Best overall F1: **RetailPrior** (0.497)
- All methods degrade at high occlusion; classical CV insufficient for dense scenes
- See `experiments/baseline_results.json` for full data
