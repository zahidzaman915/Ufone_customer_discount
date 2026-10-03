
def add_features(df):
    df = df.copy()
    # Recharge relative to past data use (affordability signal)
    df["recharge_per_gb"] = df["avg_monthly_recharge_amount"] / (df["avg_monthly_data_gb"] + 1)
    # Recharge regularity: frequent and recent recharging scores high
    df["recharge_regularity"] = df["recharge_frequency"] / (df["days_since_last_recharge"] + 1)
    # Package activations per year of tenure
    df["activations_per_year"] = df["pre_window_package_activations"] / (df["customer_tenure_months"] / 12)
    return df
