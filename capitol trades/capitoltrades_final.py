import pandas as pd

# Load your data
df = pd.read_csv(r"capitol trades\trades_with_returns.csv")

# Step 4️⃣: Calculate politician-level influence
grouped = df.groupby('Politician').agg({
    'return_5d_trade': 'mean',
    'return_5d_disclosed': 'mean',
    'win_flag': 'mean'
}).reset_index()

grouped.rename(columns={
    'return_5d_trade': 'avg_return_5d_trade',
    'return_5d_disclosed': 'avg_return_5d_disclosed',
    'win_flag': 'win_rate'
}, inplace=True)

# Influence score formula
grouped['influence_score'] = (
    0.5 * grouped['win_rate'] +
    0.25 * grouped['avg_return_5d_trade'] +
    0.25 * grouped['avg_return_5d_disclosed']
)

# Merge back
df = df.merge(grouped, on='Politician', how='left')

# Step 5️⃣: Filter trades
# First filter on influence score
filtered_df = df[df['influence_score'] >= 0.5].copy()

# Optional return filter — loosened thresholds
def meets_return_condition(row):
    if pd.isna(row['return_5d_disclosed']):
        return False
    if row['Trade_Direction'] == 1:
        return row['return_5d_disclosed'] >= 0.02  # +2% threshold
    elif row['Trade_Direction'] == -1:
        return row['return_5d_disclosed'] <= -0.02 # -2% threshold
    return False

filtered_df = filtered_df[filtered_df.apply(meets_return_condition, axis=1)].copy()

# Step 6️⃣: Generate signals
def generate_signal(row):
    if row['influence_score'] >= 0.5:
        if row['Trade_Direction'] == 1:
            return 'BUY'
        elif row['Trade_Direction'] == -1:
            return 'SHORT/EXIT'
    return None

filtered_df['trade_signal'] = filtered_df.apply(generate_signal, axis=1)

# Log and save
print(f"✅ Remaining rows after filtering: {len(filtered_df)}")
if len(filtered_df) == 0:
    print("⚠ No rows met the conditions. Consider loosening filters further or reviewing data.")


# Save output
filtered_df.to_csv(r"capitol trades\filtered_trade_signals.csv", index=False)
