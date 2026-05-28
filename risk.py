
def calculate_position_size(balance, risk_pct, stop_loss_points):
    risk_amount = balance * (risk_pct / 100)

    if stop_loss_points <= 0:
        return 0

    return int(risk_amount / stop_loss_points)
