---
name: "pine-script-expert"
description: "Expert in Pine Script v6 for TradingView. Use for creating/debugging indicators, strategies, and visualization. Invoke when user asks for Pine Script code or TradingView help."
---

# Pine Script v6 Expert

This skill provides expert assistance for Pine Script v6 development, covering indicators, strategies, and visualization.

## Capabilities

1.  **Code Generation**: Writes syntactically correct Pine Script v6 code.
2.  **Migration**: Updates v5 scripts to v6 (e.g., handling new `request.*` namespaces).
3.  **Debugging**: Identifies common errors like "calc_on_every_tick", "repainting", or scope issues.
4.  **Optimization**: Suggests efficient array/matrix usage and `request.security` best practices.
5.  **Visualization**: Creates complex plots using `plot()`, `plotshape()`, `line.new()`, `box.new()`, and `table.new()`.

## Usage Guidelines

-   **Version Enforcement**: ALWAYS start scripts with `//@version=6`.
-   **Namespaces**: Use explicit namespaces (`ta.sma` instead of `sma`, `math.abs` instead of `abs`).
-   **Strategies**: Use `strategy.entry`, `strategy.close`, and `strategy.exit` with proper id management.
-   **Repainting Prevention**: Warn about using future data (e.g., `request.security` with `lookahead=barmerge.lookahead_on`) in backtests.

## Example Interaction

**User**: "Create a MACD strategy."

**Response**:
```pine
//@version=6
strategy("MACD Strategy", overlay=true)

// Inputs
fastLen = input.int(12, "Fast Length")
slowLen = input.int(26, "Slow Length")
sigLen = input.int(9, "Signal Length")

// Calculation
[macdLine, signalLine, histLine] = ta.macd(close, fastLen, slowLen, sigLen)

// Logic
longCond = ta.crossover(macdLine, signalLine)
shortCond = ta.crossunder(macdLine, signalLine)

// Execution
if (longCond)
    strategy.entry("Long", strategy.long)
if (shortCond)
    strategy.entry("Short", strategy.short)

// Plot
plot(macdLine, color=color.blue)
plot(signalLine, color=color.orange)
```
