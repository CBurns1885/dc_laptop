# Setting Up Claude API for Days Since Last Match

## Quick Setup (5 minutes)

### Step 1: Get Your API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Click "API Keys" in the sidebar
4. Click "Create Key"
5. Copy your key (starts with `sk-ant-api...`)

### Step 2: Set Environment Variable

**Windows (PowerShell)**:
```powershell
# Temporary (this session only)
$env:ANTHROPIC_API_KEY = "sk-ant-api-YOUR-KEY-HERE"

# Permanent (all future sessions)
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'sk-ant-api-YOUR-KEY-HERE', 'User')
```

**Windows (CMD)**:
```cmd
# Temporary
set ANTHROPIC_API_KEY=sk-ant-api-YOUR-KEY-HERE

# Permanent - use System Properties > Environment Variables
```

**To make it permanent via Windows GUI**:
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Click "Advanced" tab
3. Click "Environment Variables"
4. Under "User variables", click "New"
5. Variable name: `ANTHROPIC_API_KEY`
6. Variable value: `sk-ant-api-YOUR-KEY-HERE`
7. Click OK

### Step 3: Verify Setup

Open a **new** terminal and run:

```bash
python -c "import os; print('API key set!' if os.environ.get('ANTHROPIC_API_KEY') else 'Not set')"
```

Should print: `API key set!`

### Step 4: Install Anthropic SDK

```bash
pip install anthropic
```

## Cost Information

Using Claude 3.5 Haiku (cheapest model):

**Pricing**:
- Input: $0.25 per million tokens
- Output: $1.25 per million tokens

**Actual Costs**:
- 1 match (2 teams) = ~$0.00005 (~0.005 cents)
- 100 matches = ~$0.005 (half a cent)
- Full season (380 matches) = ~$0.02 (2 cents)

**With caching** (after first run):
- Cache hit rate: 95%+
- Subsequent runs: Nearly free

**Example monthly cost**:
- Week 1: Process 150 fixtures = $0.0075 (~0.75 cents)
- Week 2-4: Cache hits = ~$0.001 (~0.1 cent)
- **Total month**: <$0.02 (2 cents)

## Troubleshooting

### "ANTHROPIC_API_KEY not found"

**Solution**: Check if variable is set:

```bash
# PowerShell
echo $env:ANTHROPIC_API_KEY

# CMD
echo %ANTHROPIC_API_KEY%
```

If empty, go back to Step 2 and set it.

### "Invalid API key"

**Solution**:
1. Check you copied the full key (starts with `sk-ant-api`)
2. Make sure there are no extra spaces
3. Generate a new key at https://console.anthropic.com/

### "Module 'anthropic' not found"

**Solution**:
```bash
pip install anthropic
```

### Rate limit errors

**Solution**: You're on free tier (50 requests/minute). Either:
1. Wait a minute
2. Upgrade at https://console.anthropic.com/settings/plans

## Free Tier Limits

**Anthropic free tier**:
- 50 requests per minute
- $5 free credits per month

**What this means**:
- ~100,000 matches per month (way more than needed)
- Plenty for football predictions

## Alternative: Skip API (Not Recommended)

If you don't want to use the API, the system falls back to league-only data:

```python
# In days_since_match_fetcher.py
# No API key needed - uses only league data
df = add_days_since_features(df, use_api=False)
```

**Warning**: This misses cup games and gives inaccurate results.

## Testing Your Setup

```bash
# Test single team query
python days_since_match_fetcher.py "Arsenal" "2024-11-30" "E0"
```

**Expected output**:
```
Arsenal last played 4 days before 2024-11-30
```

**If you see an error**, check:
1. API key is set (Step 2)
2. `anthropic` package is installed (Step 4)
3. No typos in the API key

## Security Best Practices

✅ **DO**:
- Keep your API key private
- Set it as an environment variable (not in code)
- Regenerate if leaked

❌ **DON'T**:
- Commit API keys to git
- Share keys publicly
- Hardcode keys in Python files

## Next Steps

Once setup is complete, just run:

```bash
python run_weekly.py
```

The pipeline will automatically:
1. Use Claude API to fetch complete match schedules
2. Cache results to reduce costs
3. Fall back to league-only if API fails

No additional configuration needed!

---

**Questions?**
- Anthropic Docs: https://docs.anthropic.com/
- API Console: https://console.anthropic.com/
- Pricing: https://www.anthropic.com/pricing
