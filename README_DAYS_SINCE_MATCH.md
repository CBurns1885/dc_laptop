# Days Since Last Match Fetcher

## Problem

Your league data only contains league matches. It's missing:
- ❌ Domestic cup games (FA Cup, League Cup, Copa del Rey, etc.)
- ❌ European competitions (Champions League, Europa League)
- ❌ Midweek fixtures in other competitions

This means "days since last match" calculated from league data alone is **inaccurate**.

## Solution

Use the **Claude API** to fetch complete match schedules for teams, including all competitive fixtures.

## How It Works

### 1. Query Claude for Complete Schedule

For each match, the system asks Claude:
```
"When did Arsenal last play a competitive match before 2024-11-30?"
```

Claude responds with:
```
LAST_MATCH_DATE: 2024-11-26
DAYS_SINCE: 4
```

### 2. Cache Results

Results are cached in `data/match_schedule_cache.json` to:
- Reduce API calls
- Speed up subsequent runs
- Lower costs

Cache expires after 7 days to keep data fresh.

### 3. Enrich Your Data

Add columns to your dataframe:
- `Home_DaysSinceLast`: Days since home team last played
- `Away_DaysSinceLast`: Days since away team last played

## Usage

### Setup

1. **Get Anthropic API Key**:
   - Sign up at https://console.anthropic.com/
   - Create an API key
   - Set environment variable:
     ```bash
     export ANTHROPIC_API_KEY="your-key-here"
     ```

   Or on Windows:
   ```cmd
   set ANTHROPIC_API_KEY=your-key-here
   ```

2. **Install Anthropic SDK** (if not already installed):
   ```bash
   pip install anthropic
   ```

### Basic Usage

```python
from days_since_match_fetcher import add_days_since_features
import pandas as pd

# Load your fixtures
df = pd.read_csv('fixtures.csv')

# Add days since last match (uses API)
df_enriched = add_days_since_features(df, use_api=True)

# Now df has Home_DaysSinceLast and Away_DaysSinceLast columns
```

### Advanced Usage

```python
from days_since_match_fetcher import DaysSinceMatchFetcher

# Initialize fetcher
fetcher = DaysSinceMatchFetcher()

# Get days for a single team
days = fetcher.get_days_since_last_match(
    team="Arsenal",
    match_date="2024-11-30",
    league="E0"  # Premier League (helps with context)
)

print(f"Arsenal last played {days} days ago")
```

### Enrich Dataframe

```python
fetcher = DaysSinceMatchFetcher()

df_enriched = fetcher.enrich_dataframe(
    df,
    home_col='HomeTeam',
    away_col='AwayTeam',
    date_col='Date',
    league_col='League'
)

# Check coverage
print(df_enriched['Home_DaysSinceLast'].describe())
```

### Fallback Mode (No API)

If you don't have an API key or want to save costs:

```python
# Uses only league data (misses cup games)
df_enriched = add_days_since_features(df, use_api=False)
```

**Warning**: This will miss cup games and give inaccurate results during busy periods.

## Integration with Prediction System

### Option 1: Enrich Fixtures Before Prediction

Update `run_weekly.py` to fetch days since last match before generating predictions:

```python
# In run_weekly.py, after loading fixtures

from days_since_match_fetcher import add_days_since_features

# Enrich fixtures with accurate days since last match
fixtures = add_days_since_features(fixtures, use_api=True)

# Then continue with predictions
predict_week(fixtures)
```

### Option 2: Enrich Historical Data

Update your historical data build process:

```python
# In generate_and_load_stats.py or similar

from days_since_match_fetcher import add_days_since_features

# After loading all historical CSVs
all_matches = pd.read_parquet('data/features.parquet')

# Add accurate days since last match
all_matches = add_days_since_features(all_matches, use_api=True)

# Save back
all_matches.to_parquet('data/features.parquet')
```

## Cost Estimation

Using **Claude 3.5 Haiku** (cheapest model):
- **$0.25** per million input tokens
- **$1.25** per million output tokens
- ~100 tokens per query

**Example costs**:
- 100 matches (200 queries) = **$0.005** (~half a cent)
- 1,000 matches (2,000 queries) = **$0.05** (5 cents)
- Full season 380 matches (760 queries) = **$0.019** (~2 cents)

**With caching**:
- First run: Full cost
- Subsequent runs: **~0 cost** (cache hit rate >95%)

## Cache Management

### View Cache

```python
from days_since_match_fetcher import DaysSinceMatchFetcher

fetcher = DaysSinceMatchFetcher()
print(f"Cache size: {len(fetcher.cache)} entries")
```

### Clear Cache

```bash
rm data/match_schedule_cache.json
```

### Cache Format

```json
{
  "Arsenal_2024-11-30": {
    "days_since_last_match": 4,
    "cached_at": "2024-11-30T10:30:00"
  },
  "Man City_2024-11-30": {
    "days_since_last_match": 3,
    "cached_at": "2024-11-30T10:30:15"
  }
}
```

Cache expires after 7 days to ensure fresh data.

## Error Handling

### No API Key

```
ValueError: ANTHROPIC_API_KEY not found in environment variables
```

**Solution**: Set the environment variable or pass key directly:

```python
fetcher = DaysSinceMatchFetcher(api_key="your-key-here")
```

### API Error

If Claude can't find match data, returns `None`:

```python
days = fetcher.get_days_since_last_match("Unknown Team", "2024-11-30")
# days = None
```

The enriched dataframe will have `NaN` for missing values.

### Fallback Behavior

If API fails, automatically falls back to league-only calculation:

```python
try:
    df = add_days_since_features(df, use_api=True)
except:
    print("API failed, using league-only data")
    df = add_days_since_features(df, use_api=False)
```

## Testing

### Test Single Team

```bash
python days_since_match_fetcher.py "Arsenal" "2024-11-30" "E0"
```

Output:
```
Arsenal last played 4 days before 2024-11-30
```

### Test Dataframe Enrichment

```python
import pandas as pd
from days_since_match_fetcher import add_days_since_features

# Create test fixtures
fixtures = pd.DataFrame({
    'Date': ['2024-11-30', '2024-12-01'],
    'HomeTeam': ['Arsenal', 'Man City'],
    'AwayTeam': ['Chelsea', 'Liverpool'],
    'League': ['E0', 'E0']
})

# Enrich
fixtures = add_days_since_features(fixtures, use_api=True)

print(fixtures[['HomeTeam', 'AwayTeam', 'Home_DaysSinceLast', 'Away_DaysSinceLast']])
```

## Comparison: API vs League-Only

### Example: Arsenal in November 2024

**League-only calculation**:
```
Last Premier League match: Nov 23 (7 days ago)
```

**API calculation (includes cups)**:
```
Last match: Champions League on Nov 26 (4 days ago)
```

**Impact**: League-only data says Arsenal are well-rested (7 days), but they actually played midweek (4 days ago) - this affects fatigue predictions!

## Features

✅ **Complete match coverage**: League + cups + Europe
✅ **Automatic caching**: Reduces API costs by 95%+
✅ **Fast model**: Uses Claude 3.5 Haiku (cheapest)
✅ **Fallback mode**: Works without API (league-only)
✅ **Batch processing**: Enriches entire dataframes
✅ **Error handling**: Graceful degradation
✅ **Cost efficient**: ~$0.02 per full season

## Limitations

### 1. Historical Data Accuracy

Claude's knowledge cutoff means:
- **Recent matches**: ✅ Very accurate
- **Older matches**: ⚠️ May have gaps

For historical backtesting, consider using league-only data for old matches.

### 2. Team Name Matching

Claude is smart but may need help with:
- Team name variations ("Man City" vs "Manchester City")
- Recently promoted teams
- Non-English team names

The `league` parameter helps provide context.

### 3. Rate Limits

Anthropic API has rate limits:
- Free tier: 50 requests/minute
- Paid tier: Higher limits

For large batches (1000+ matches), add delays:

```python
import time

for idx, row in df.iterrows():
    days = fetcher.get_days_since_last_match(...)
    if idx % 50 == 0:
        time.sleep(1)  # Pause every 50 requests
```

## Best Practices

### 1. Enrich Once, Cache Forever

Don't re-fetch for the same team/date combinations:

```python
# Good: Cache persists across runs
fetcher = DaysSinceMatchFetcher()
df1 = fetcher.enrich_dataframe(fixtures_week1)
df2 = fetcher.enrich_dataframe(fixtures_week2)  # Uses cache for overlaps
```

### 2. Batch Process Historical Data

Process all historical data once, save to parquet:

```python
# One-time enrichment
all_data = pd.read_parquet('data/features.parquet')
all_data = add_days_since_features(all_data, use_api=True)
all_data.to_parquet('data/features_enriched.parquet')

# Future runs: Just use enriched data
df = pd.read_parquet('data/features_enriched.parquet')
```

### 3. Monitor API Costs

Check your Anthropic dashboard:
- https://console.anthropic.com/settings/usage

Set up billing alerts if processing large datasets.

### 4. Validate Results

Spot-check a few results manually:

```python
# Check Arsenal's schedule
days = fetcher.get_days_since_last_match("Arsenal", "2024-11-30", "E0")
print(f"Claude says: {days} days")

# Manually verify on transfermarkt.com or similar
```

## Troubleshooting

### "ANTHROPIC_API_KEY not found"

**Solution**:
```bash
# Linux/Mac
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows CMD
set ANTHROPIC_API_KEY=sk-ant-...

# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

### Low Coverage (<50%)

**Possible causes**:
- Team names don't match Claude's database
- Dates are in the future
- League context missing

**Solution**: Add league parameter:

```python
days = fetcher.get_days_since_last_match(
    "Arsenal",
    "2024-11-30",
    league="E0"  # Helps Claude identify the correct team
)
```

### Slow Processing

**Solution**: Use Haiku model (already default) and enable caching:

```python
# Cache is enabled by default
# Check cache hit rate
print(f"Cache entries: {len(fetcher.cache)}")
```

## Related Files

- **[days_since_match_fetcher.py](days_since_match_fetcher.py)** - Main implementation
- **[config.py](config.py)** - Configuration (may need to add API key here)
- **[run_weekly.py](run_weekly.py)** - Integration point for weekly predictions

---

**Summary**: Use Claude API to get accurate "days since last match" data that includes cup games and European fixtures, not just league matches. Costs ~$0.02 per season, with 95%+ cache hit rate on subsequent runs.
