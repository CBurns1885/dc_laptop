# test_api_football.py
"""
Test Script for API-Football Enhanced System
Tests all modules with synthetic data (no API key required)

Run: python test_api_football.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import sqlite3


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_test_data(n_matches: int = 500, n_teams: int = 20) -> pd.DataFrame:
    """Generate realistic test data with xG and all features"""
    np.random.seed(42)
    
    teams = [f"Team_{i}" for i in range(n_teams)]
    leagues = ['E0', 'D1', 'SP1', 'FA_CUP', 'UCL']
    league_types = {'E0': 'league', 'D1': 'league', 'SP1': 'league', 
                    'FA_CUP': 'cup', 'UCL': 'cup'}
    
    data = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n_matches):
        # Random teams
        home_idx = i % n_teams
        away_idx = (i + 3 + np.random.randint(1, n_teams-1)) % n_teams
        if away_idx == home_idx:
            away_idx = (away_idx + 1) % n_teams
        
        # League (80% main leagues, 20% cups)
        if np.random.random() < 0.8:
            league = np.random.choice(['E0', 'D1', 'SP1'])
        else:
            league = np.random.choice(['FA_CUP', 'UCL'])
        
        # Goals with home advantage
        home_lambda = 1.5 if league_types[league] == 'league' else 1.3
        away_lambda = 1.2 if league_types[league] == 'league' else 1.1
        
        home_goals = min(np.random.poisson(home_lambda), 6)
        away_goals = min(np.random.poisson(away_lambda), 5)
        
        # xG with some variance from actual goals
        home_xg = max(0.2, home_goals + np.random.normal(0, 0.4))
        away_xg = max(0.2, away_goals + np.random.normal(0, 0.4))
        
        # Rest days
        home_rest = np.random.choice([3, 4, 5, 6, 7, 8, 10, 14], 
                                     p=[0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.03, 0.02])
        away_rest = np.random.choice([3, 4, 5, 6, 7, 8, 10, 14], 
                                     p=[0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.03, 0.02])
        
        # Injuries (0-5)
        home_injuries = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.1, 0.07, 0.03])
        away_injuries = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.1, 0.07, 0.03])
        
        # Formations
        formations = ['4-3-3', '4-4-2', '4-2-3-1', '3-5-2', '5-3-2', '4-5-1']
        form_ratings = {'4-3-3': 0.85, '4-4-2': 0.70, '4-2-3-1': 0.80, 
                       '3-5-2': 0.65, '5-3-2': 0.50, '4-5-1': 0.55}
        home_formation = np.random.choice(formations)
        away_formation = np.random.choice(formations)
        
        # Stats
        home_shots = max(5, int(home_xg * 6 + np.random.normal(0, 2)))
        away_shots = max(3, int(away_xg * 6 + np.random.normal(0, 2)))
        home_sot = max(1, int(home_shots * 0.35 + np.random.normal(0, 1)))
        away_sot = max(1, int(away_shots * 0.35 + np.random.normal(0, 1)))
        home_corners = np.random.poisson(5)
        away_corners = np.random.poisson(4)
        home_poss = max(30, min(70, 50 + np.random.normal(0, 10)))
        
        data.append({
            'fixture_id': 1000 + i,
            'Date': base_date + timedelta(days=i*2),
            'League': league,
            'LeagueName': league,
            'LeagueType': league_types[league],
            'Season': 2023 if i < 250 else 2024,
            'Round': f"Round {(i % 38) + 1}",
            'HomeTeam': teams[home_idx],
            'AwayTeam': teams[away_idx],
            'HomeTeamID': home_idx + 100,
            'AwayTeamID': away_idx + 100,
            'FTHG': home_goals,
            'FTAG': away_goals,
            'HTHG': min(home_goals, np.random.poisson(0.7)),
            'HTAG': min(away_goals, np.random.poisson(0.5)),
            'home_xG': home_xg,
            'away_xG': away_xg,
            'home_xG_combined': home_xg,
            'away_xG_combined': away_xg,
            'Referee': f"Ref_{np.random.randint(1, 20)}",
            'Venue': f"Stadium_{home_idx}",
            # Stats
            'HS': home_shots,
            'AS': away_shots,
            'HST': home_sot,
            'AST': away_sot,
            'HC': home_corners,
            'AC': away_corners,
            'HPoss': home_poss,
            'APoss': 100 - home_poss,
            'HF': np.random.poisson(12),
            'AF': np.random.poisson(12),
            'HY': np.random.poisson(1.5),
            'AY': np.random.poisson(1.5),
            'HR': 1 if np.random.random() < 0.05 else 0,
            'AR': 1 if np.random.random() < 0.05 else 0,
            # Enhanced features
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            'home_had_cup_midweek': 1 if home_rest < 5 and np.random.random() < 0.3 else 0,
            'away_had_cup_midweek': 1 if away_rest < 5 and np.random.random() < 0.3 else 0,
            'home_injuries_count': home_injuries,
            'away_injuries_count': away_injuries,
            'home_key_injuries': min(home_injuries // 2, 2),
            'away_key_injuries': min(away_injuries // 2, 2),
            'home_formation': home_formation,
            'away_formation': away_formation,
            'home_formation_attack': form_ratings[home_formation],
            'away_formation_attack': form_ratings[away_formation],
            'is_cup_match': 1 if league_types[league] == 'cup' else 0,
            'is_knockout': 1 if league in ['FA_CUP', 'UCL'] and np.random.random() < 0.3 else 0,
        })
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df


def setup_test_db(df: pd.DataFrame, db_path: Path):
    """Create test database from DataFrame"""
    with sqlite3.connect(db_path) as conn:
        # Fixtures table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fixtures (
                fixture_id INTEGER PRIMARY KEY,
                league_code TEXT,
                league_id INTEGER,
                league_name TEXT,
                league_type TEXT,
                season INTEGER,
                round TEXT,
                date DATE,
                home_team_id INTEGER,
                home_team TEXT,
                away_team_id INTEGER,
                away_team TEXT,
                home_goals INTEGER,
                away_goals INTEGER,
                ht_home_goals INTEGER,
                ht_away_goals INTEGER,
                status TEXT,
                home_xG REAL,
                away_xG REAL,
                referee TEXT,
                venue_name TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fixture_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER,
                team_id INTEGER,
                team_name TEXT,
                is_home INTEGER,
                shots_total INTEGER,
                shots_on_target INTEGER,
                ball_possession REAL,
                corners INTEGER,
                fouls INTEGER,
                yellow_cards INTEGER,
                red_cards INTEGER,
                expected_goals REAL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS injuries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                player_name TEXT,
                team_id INTEGER,
                team_name TEXT,
                fixture_id INTEGER,
                league_id INTEGER,
                season INTEGER,
                injury_type TEXT,
                injury_reason TEXT,
                date DATE
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lineups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER,
                team_id INTEGER,
                team_name TEXT,
                formation TEXT,
                player_id INTEGER,
                player_name TEXT,
                position TEXT,
                is_starter INTEGER
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                name TEXT,
                short_name TEXT,
                country TEXT
            )
        """)
        
        # Insert fixtures
        for _, row in df.iterrows():
            conn.execute("""
                INSERT INTO fixtures (
                    fixture_id, league_code, league_name, league_type, season,
                    round, date, home_team_id, home_team, away_team_id, away_team,
                    home_goals, away_goals, ht_home_goals, ht_away_goals,
                    status, home_xG, away_xG, referee, venue_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['fixture_id'], row['League'], row['LeagueName'], row['LeagueType'],
                row['Season'], row['Round'], row['Date'].strftime('%Y-%m-%d'),
                row['HomeTeamID'], row['HomeTeam'], row['AwayTeamID'], row['AwayTeam'],
                row['FTHG'], row['FTAG'], row['HTHG'], row['HTAG'],
                'FT', row['home_xG'], row['away_xG'], row['Referee'], row['Venue']
            ))
        
        conn.commit()


# =============================================================================
# TESTS
# =============================================================================

def test_dc_model():
    """Test Dixon-Coles model with xG"""
    print("\n" + "="*60)
    print("TEST 1: xG-Integrated Dixon-Coles Model")
    print("="*60)
    
    from models_dc_xg import fit_league_xg, price_match_xg
    
    df = generate_test_data(200, 10)
    df_e0 = df[df['League'] == 'E0'].copy()
    
    # Fit model
    params = fit_league_xg(df_e0, use_xg=True)
    
    print(f"✓ Fitted E0 with {len(df_e0)} matches")
    print(f"  Home advantage: {params.home_adv:.4f}")
    print(f"  Rho: {params.rho:.4f}")
    print(f"  League avg goals: {params.league_avg_goals:.2f}")
    print(f"  xG attack teams: {len(params.recent_xg_attack)}")
    
    # Price a match
    home = df_e0['HomeTeam'].iloc[0]
    away = df_e0['AwayTeam'].iloc[1]
    
    probs = price_match_xg(
        params, home, away,
        use_xg_form=True,
        home_rest_days=4,
        away_rest_days=7,
        home_injuries_count=2,
        home_key_injuries=1,
        home_formation_attack=0.85,
        away_formation_attack=0.70,
        is_cup_match=0
    )
    
    print(f"\n  {home} vs {away}:")
    print(f"    Expected goals: {probs['expected_home_goals']:.2f} - {probs['expected_away_goals']:.2f}")
    print(f"    BTTS Yes: {probs['DC_BTTS_Y']:.1%}")
    print(f"    Over 2.5: {probs['DC_OU_2_5_O']:.1%}")
    
    # Verify reasonable ranges
    assert 0.1 < probs['DC_BTTS_Y'] < 0.9, "BTTS out of range"
    assert 0.1 < probs['DC_OU_2_5_O'] < 0.9, "Over 2.5 out of range"
    assert abs(probs['DC_1X2_H'] + probs['DC_1X2_D'] + probs['DC_1X2_A'] - 1.0) < 0.01, "1X2 doesn't sum to 1"
    
    print("\n✅ TEST 1 PASSED")
    return True


def test_cup_model():
    """Test cup competition handling"""
    print("\n" + "="*60)
    print("TEST 2: Cup Competition Handling")
    print("="*60)
    
    from models_dc_xg import fit_league_xg, price_match_xg, LEAGUE_CONFIG
    
    df = generate_test_data(300, 15)
    
    # Test cup config
    cup_config = LEAGUE_CONFIG.get('FA_CUP', {})
    print(f"✓ FA Cup config:")
    print(f"    Home advantage mult: {cup_config.get('home_adv_mult', 1.0)}")
    print(f"    xG weight: {cup_config.get('xg_weight', 0.5)}")
    
    # Fit cup model
    df_cup = df[df['League'] == 'FA_CUP'].copy()
    if len(df_cup) >= 30:
        params_cup = fit_league_xg(df_cup, use_xg=True)
        print(f"\n✓ Fitted FA Cup with {len(df_cup)} matches")
        print(f"    Is cup: {params_cup.is_cup}")
        print(f"    Home adv mult: {params_cup.home_adv_multiplier}")
        
        # Compare league vs cup predictions
        df_league = df[df['League'] == 'E0'].copy()
        params_league = fit_league_xg(df_league, use_xg=True)
        
        # Same teams, different competition
        home = df_league['HomeTeam'].iloc[0]
        away = df_league['AwayTeam'].iloc[1]
        
        if home in params_cup.attack and away in params_cup.attack:
            probs_cup = price_match_xg(params_cup, home, away, is_cup_match=1)
            probs_league = price_match_xg(params_league, home, away, is_cup_match=0)
            
            print(f"\n  {home} vs {away}:")
            print(f"    League BTTS: {probs_league['DC_BTTS_Y']:.1%}")
            print(f"    Cup BTTS:    {probs_cup['DC_BTTS_Y']:.1%}")
    
    print("\n✅ TEST 2 PASSED")
    return True


def test_feature_impacts():
    """Test that features have expected impacts"""
    print("\n" + "="*60)
    print("TEST 3: Feature Impact Verification")
    print("="*60)
    
    from models_dc_xg import fit_league_xg, price_match_xg
    
    df = generate_test_data(200, 10)
    df_e0 = df[df['League'] == 'E0'].copy()
    params = fit_league_xg(df_e0, use_xg=True)
    
    home = df_e0['HomeTeam'].iloc[0]
    away = df_e0['AwayTeam'].iloc[1]
    
    # Baseline
    base = price_match_xg(params, home, away)
    
    # Short rest should reduce expected goals
    short_rest = price_match_xg(params, home, away, home_rest_days=3)
    print(f"✓ Short rest impact:")
    print(f"    Base home goals: {base['expected_home_goals']:.2f}")
    print(f"    Short rest:      {short_rest['expected_home_goals']:.2f}")
    assert short_rest['expected_home_goals'] < base['expected_home_goals'], "Short rest should reduce goals"
    
    # Injuries should reduce expected goals
    injured = price_match_xg(params, home, away, home_injuries_count=5, home_key_injuries=2)
    print(f"\n✓ Injury impact:")
    print(f"    Base home goals: {base['expected_home_goals']:.2f}")
    print(f"    With injuries:   {injured['expected_home_goals']:.2f}")
    assert injured['expected_home_goals'] < base['expected_home_goals'], "Injuries should reduce goals"
    
    # Attacking formation should increase goals
    attacking = price_match_xg(params, home, away, home_formation_attack=0.90)
    defensive = price_match_xg(params, home, away, home_formation_attack=0.50)
    print(f"\n✓ Formation impact:")
    print(f"    Attacking formation: {attacking['expected_home_goals']:.2f}")
    print(f"    Defensive formation: {defensive['expected_home_goals']:.2f}")
    assert attacking['expected_home_goals'] > defensive['expected_home_goals'], "Attacking should score more"
    
    print("\n✅ TEST 3 PASSED")
    return True


def test_backtest():
    """Test backtest module"""
    print("\n" + "="*60)
    print("TEST 4: Backtest Module")
    print("="*60)
    
    from backtest_api import Backtester, BacktestConfig, calculate_metrics
    
    # Generate test data
    df = generate_test_data(400, 15)
    
    # Add required columns for backtest
    df['home_xG_for_ma5'] = df.groupby('League')['home_xG'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    df['away_xG_for_ma5'] = df.groupby('League')['away_xG'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    
    # Save temp parquet
    temp_dir = Path(tempfile.mkdtemp())
    features_path = temp_dir / "features.parquet"
    df.to_parquet(features_path)
    
    # Run backtest
    config = BacktestConfig(
        name='test',
        leagues=['E0', 'D1'],
        include_cups=False,
        retrain_frequency=30,
        min_train_matches=50
    )
    
    bt = Backtester(config)
    predictions = bt.run(df)
    
    print(f"✓ Generated {len(predictions)} predictions")
    
    if len(predictions) > 0:
        results = bt.evaluate(predictions)
        
        btts_acc = results['by_market'].get('BTTS', {}).get('accuracy', 0)
        ou25_acc = results['by_market'].get('OU_2_5', {}).get('accuracy', 0)
        
        print(f"  BTTS accuracy: {btts_acc:.1%}")
        print(f"  Over 2.5 accuracy: {ou25_acc:.1%}")
        
        assert btts_acc > 0.4, "BTTS accuracy too low"
        assert ou25_acc > 0.4, "O/U accuracy too low"
    
    # Cleanup
    features_path.unlink()
    
    print("\n✅ TEST 4 PASSED")
    return True


def test_predictions():
    """Test prediction module"""
    print("\n" + "="*60)
    print("TEST 5: Prediction Module")
    print("="*60)
    
    from predict_api import MatchPredictor
    
    # Generate and save test data
    df = generate_test_data(300, 15)
    
    # Add rolling features
    for league in df['League'].unique():
        mask = df['League'] == league
        df.loc[mask, 'home_xG_for_ma5'] = df.loc[mask, 'home_xG'].shift(1).rolling(5).mean()
        df.loc[mask, 'away_xG_for_ma5'] = df.loc[mask, 'away_xG'].shift(1).rolling(5).mean()
    
    temp_dir = Path(tempfile.mkdtemp())
    features_path = temp_dir / "features.parquet"
    df.to_parquet(features_path)
    
    # Test predictor
    predictor = MatchPredictor(api_key=None, features_path=features_path)
    
    print(f"✓ Loaded models for {len(predictor.params_by_league)} leagues")
    
    # Make prediction
    home = df[df['League'] == 'E0']['HomeTeam'].iloc[0]
    away = df[df['League'] == 'E0']['AwayTeam'].iloc[1]
    
    pred = predictor.predict_fixture(
        home=home,
        away=away,
        league='E0'
    )
    
    if 'error' not in pred:
        print(f"\n  {home} vs {away}:")
        print(f"    BTTS Yes: {pred['btts_yes']:.1%}")
        print(f"    Over 2.5: {pred['over_2_5']:.1%}")
        print(f"    Expected: {pred['expected_home_goals']:.2f} - {pred['expected_away_goals']:.2f}")
        
        assert 0 < pred['btts_yes'] < 1, "Invalid BTTS probability"
        assert 0 < pred['over_2_5'] < 1, "Invalid O/U probability"
    
    # Cleanup
    features_path.unlink()
    
    print("\n✅ TEST 5 PASSED")
    return True


def test_league_configs():
    """Test all league configurations"""
    print("\n" + "="*60)
    print("TEST 6: League Configurations")
    print("="*60)
    
    from api_football_client import LEAGUES
    from models_dc_xg import LEAGUE_CONFIG, get_league_config
    
    print(f"✓ {len(LEAGUES)} leagues configured")
    
    # Check all leagues have valid config
    leagues_count = 0
    cups_count = 0
    
    for code, info in LEAGUES.items():
        config = get_league_config(code)
        
        if info['type'] == 'cup':
            cups_count += 1
            assert 'home_adv_mult' in config or code not in LEAGUE_CONFIG, f"{code} missing home_adv_mult"
        else:
            leagues_count += 1
        
        assert 'rho_init' in config, f"{code} missing rho_init"
        assert 'rho_bounds' in config, f"{code} missing rho_bounds"
    
    print(f"  League competitions: {leagues_count}")
    print(f"  Cup competitions: {cups_count}")
    print(f"  European: {sum(1 for c in LEAGUES if c in ['UCL', 'UEL', 'UECL'])}")
    
    print("\n✅ TEST 6 PASSED")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("API-FOOTBALL ENHANCED SYSTEM - TEST SUITE")
    print("="*70)
    
    tests = [
        ("DC Model with xG", test_dc_model),
        ("Cup Competition Handling", test_cup_model),
        ("Feature Impacts", test_feature_impacts),
        ("Backtest Module", test_backtest),
        ("Prediction Module", test_predictions),
        ("League Configurations", test_league_configs),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
