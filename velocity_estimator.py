import pandas as pd

# ---- Input file (adjust the path to your file) ----
input_file = "assets/offensive_players_hockey.csv"
output_file = "assets/offensive_players_with_velocity.csv"

# Load CSV
df = pd.read_csv(input_file)

# Sort to ensure correct order
df = df.sort_values(by=["player_id", "timeframe"]).reset_index(drop=True)

# Compute velocity components
df["vx"] = df.groupby("player_id")["x"].diff() / df.groupby("player_id")["timeframe"].diff()
df["vy"] = df.groupby("player_id")["y"].diff() / df.groupby("player_id")["timeframe"].diff()

# Replace NaN (first row per player) with 0
df[["vx", "vy"]] = df[["vx", "vy"]].fillna(0)

# Save to new CSV
df.to_csv(output_file, index=False)

print(f"File saved as {output_file}")
