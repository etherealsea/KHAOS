
## 6. OverflowError: Python int too large to convert to C long
**Problem:** In `resample_btc_only.py`, `pd.read_csv` with `engine='c'` (default) failed on some rows in the BTC dataset.
**Cause:** The dataset contained extremely large integer-like strings in the 'timestamp' column or corrupted lines, which the default C parser couldn't handle as standard integers.
**Solution:** Switched to `engine='python'` and `dtype=str` to read everything as strings first. Then cleaned the data by coercing to numeric (`pd.to_numeric(..., errors='coerce')`) and filtering out invalid ranges (e.g., timestamps < 2009 or > 2030).

## 7. NameError: name 'all_X' is not defined
**Problem:** In `train_khaos_walk_forward.py`, the training loop failed with `NameError`.
**Cause:** The variables `all_X` and `all_Y` were used inside `try-except` blocks but were not initialized in the parent scope, causing them to be undefined if certain branches were taken or just due to scoping oversight.
**Solution:** Initialized `all_X = []` and `all_Y = []` at the beginning of the `train_walk_forward` function, before entering the file processing loop.

## 8. FutureWarning: parsing datetimes with mixed time zones
**Problem:** Pandas raised a warning (and potential future error) when parsing dates from YFinance CSVs: `parsing datetimes with mixed time zones will raise an error unless utc=True`.
**Cause:** YFinance data often includes timezone offsets (e.g., `-05:00`). Pandas' default parser handles them but warns about future deprecation for mixed types.
**Solution:** Explicitly used `pd.to_datetime(..., utc=True)` to convert all timestamps to UTC first, and then optionally used `.dt.tz_localize(None)` to make them timezone-naive for consistency across datasets.

## 9. Pine compile error: plotshape text requires const string
**Problem:** TradingView raised compile errors when `plotshape(..., text=...)` used dynamic ternary expressions.
**Cause:** Pine requires `text` to be a compile-time const string, while `language_mode == "中文" ? "买" : "BUY"` is an input-dependent expression.
**Solution:** Split into separate `plotshape` calls for CN/EN branches, each with const `text` literals.

## 10. Signal collapse after stacked precision filters
**Problem:** After enabling multiple filters together (`min_confidence`, `min_main_bias`, `cooldown`, `bar-close`), live charts showed very low signal frequency and weak practical performance.
**Cause:** Filter stacking reduced recall too aggressively; constraints interacted non-linearly in trending sessions.
**Solution:** Paused this optimization direction and recorded a rollback/re-evaluation plan in the development log before next session restarts.
