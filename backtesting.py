
import numpy as np

def calculate_metrics(trades):
    pnl = np.array(trades)

    equity_curve = pnl.cumsum()

    returns = pnl / 100000

    sharpe = returns.mean() / (returns.std() + 1e-9)

    drawdown = equity_curve - np.maximum.accumulate(equity_curve)

    return {
        "net_profit": float(pnl.sum()),
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown": round(float(drawdown.min()), 2),
        "expectancy": round(float(pnl.mean()), 2)
    }
