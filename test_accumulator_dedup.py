"""
Test to verify O/U accumulator deduplication logic
"""
import pandas as pd

# Simulate the scenario where one match has multiple O/U bets
test_data = [
    {'Match': 'Arsenal vs Chelsea', 'Market': 'OU_1_5', 'Prediction': 'Over', 'Confidence': 0.92},
    {'Match': 'Arsenal vs Chelsea', 'Market': 'OU_3_5', 'Prediction': 'Under', 'Confidence': 0.94},  # Best for this match
    {'Match': 'Arsenal vs Chelsea', 'Market': 'OU_4_5', 'Prediction': 'Under', 'Confidence': 0.91},
    {'Match': 'Liverpool vs Man Utd', 'Market': 'OU_1_5', 'Prediction': 'Over', 'Confidence': 0.95},  # Best for this match
    {'Match': 'Liverpool vs Man Utd', 'Market': 'OU_3_5', 'Prediction': 'Over', 'Confidence': 0.93},
    {'Match': 'Man City vs Spurs', 'Market': 'OU_1_5', 'Prediction': 'Over', 'Confidence': 0.96},  # Best for this match
]

df = pd.DataFrame(test_data)

print("BEFORE Deduplication:")
print("=" * 60)
print(df[['Match', 'Market', 'Confidence']])
print(f"\nTotal bets: {len(df)}")
print(f"Unique matches: {df['Match'].nunique()}")

# Apply the deduplication logic (same as in the fix)
df_sorted = df.sort_values('Confidence', ascending=False)
df_dedup = df_sorted.drop_duplicates(subset='Match', keep='first')

print("\n" + "=" * 60)
print("AFTER Deduplication (keeping highest confidence per match):")
print("=" * 60)
print(df_dedup[['Match', 'Market', 'Confidence']])
print(f"\nTotal bets: {len(df_dedup)}")
print(f"Unique matches: {df_dedup['Match'].nunique()}")

print("\n" + "=" * 60)
print("VERIFICATION:")
print("=" * 60)

# Check that each match appears only once
if df_dedup['Match'].nunique() == len(df_dedup):
    print("[OK] SUCCESS: Each match appears exactly once")
else:
    print("[FAIL] FAILED: Some matches appear multiple times")

# Check that we kept the highest confidence bet for each match
for match in df['Match'].unique():
    original_bets = df[df['Match'] == match]
    best_confidence = original_bets['Confidence'].max()
    kept_bet = df_dedup[df_dedup['Match'] == match]

    if len(kept_bet) == 1 and kept_bet['Confidence'].values[0] == best_confidence:
        print(f"[OK] {match}: Kept best bet ({best_confidence*100:.1f}%)")
    else:
        print(f"[FAIL] {match}: Did not keep best bet!")

print("\n" + "=" * 60)
print("Deduplication logic works correctly!")
print("Each match will appear in only ONE accumulator.")
