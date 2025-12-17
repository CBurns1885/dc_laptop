#!/usr/bin/env python3
"""
BTTS/OU Market Optimizer
Specialized module for improving BTTS and Over/Under predictions
Uses historical patterns and advanced statistical analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sqlite3
from datetime import datetime, timedelta

class BTTSOUOptimizer:
    """
    Advanced optimizer for BTTS and Over/Under markets
    Learns from historical patterns to improve predictions
    """
    
    def __init__(self, historical_data: pd.DataFrame = None):
        self.historical_data = historical_data
        self.patterns = {}
        self.league_profiles = {}
        
    def analyze_league_patterns(self) -> Dict:
        """
        NEW: Analyze league-specific scoring patterns
        Different leagues have different goal tendencies
        """
        if self.historical_data is None:
            return {}
        
        leagues = self.historical_data['League'].unique()
        
        for league in leagues:
            league_data = self.historical_data[self.historical_data['League'] == league]
            
            if len(league_data) < 50:  # Need sufficient data
                continue
            
            profile = {
                'avg_goals': league_data[['FTHG', 'FTAG']].sum(axis=1).mean(),
                'btts_rate': (
                    ((league_data['FTHG'] > 0) & (league_data['FTAG'] > 0)).mean()
                ),
                'over_2_5_rate': ((league_data['FTHG'] + league_data['FTAG']) > 2.5).mean(),
                'over_1_5_rate': ((league_data['FTHG'] + league_data['FTAG']) > 1.5).mean(),
                'over_3_5_rate': ((league_data['FTHG'] + league_data['FTAG']) > 3.5).mean(),
                'clean_sheet_home': (league_data['FTAG'] == 0).mean(),
                'clean_sheet_away': (league_data['FTHG'] == 0).mean(),
            }
            
            # Goal distribution
            total_goals = league_data['FTHG'] + league_data['FTAG']
            profile['goal_distribution'] = {
                '0': (total_goals == 0).mean(),
                '1': (total_goals == 1).mean(),
                '2': (total_goals == 2).mean(),
                '3': (total_goals == 3).mean(),
                '4': (total_goals == 4).mean(),
                '5+': (total_goals >= 5).mean(),
            }
            
            self.league_profiles[league] = profile
        
        return self.league_profiles
    
    def calculate_team_btts_tendency(self, team: str, last_n_games: int = 10) -> float:
        """
        NEW: Calculate team's BTTS tendency in recent games
        """
        if self.historical_data is None:
            return 0.5
        
        team_games = self.historical_data[
            (self.historical_data['HomeTeam'] == team) | 
            (self.historical_data['AwayTeam'] == team)
        ].tail(last_n_games)
        
        if team_games.empty:
            return 0.5
        
        btts_count = ((team_games['FTHG'] > 0) & (team_games['FTAG'] > 0)).sum()
        return btts_count / len(team_games)
    
    def calculate_team_goal_tendency(self, team: str, home: bool = True, last_n_games: int = 10) -> Dict:
        """
        NEW: Calculate team's goal-scoring and conceding patterns
        """
        if self.historical_data is None:
            return {'scored': 1.5, 'conceded': 1.5, 'over_2_5': 0.5}
        
        if home:
            team_games = self.historical_data[
                self.historical_data['HomeTeam'] == team
            ].tail(last_n_games)
            scored = team_games['FTHG'].mean() if not team_games.empty else 1.5
            conceded = team_games['FTAG'].mean() if not team_games.empty else 1.5
        else:
            team_games = self.historical_data[
                self.historical_data['AwayTeam'] == team
            ].tail(last_n_games)
            scored = team_games['FTAG'].mean() if not team_games.empty else 1.5
            conceded = team_games['FTHG'].mean() if not team_games.empty else 1.5
        
        if not team_games.empty:
            total_goals = team_games['FTHG'] + team_games['FTAG']
            over_2_5_rate = (total_goals > 2.5).mean()
        else:
            over_2_5_rate = 0.5
        
        return {
            'scored': scored,
            'conceded': conceded,
            'over_2_5': over_2_5_rate,
            'total_expected': scored + conceded
        }
    
    def adjust_btts_probability(self, base_prob: float, home_team: str, away_team: str, 
                               league: str = None) -> float:
        """
        NEW: Adjust BTTS probability based on team and league patterns
        """
        # Start with base probability
        adjusted_prob = base_prob
        
        # Team-specific adjustments
        home_btts = self.calculate_team_btts_tendency(home_team)
        away_btts = self.calculate_team_btts_tendency(away_team)
        team_btts_avg = (home_btts + away_btts) / 2
        
        # Blend team tendency with base (70% base, 30% team)
        adjusted_prob = (adjusted_prob * 0.7) + (team_btts_avg * 0.3)
        
        # League adjustment if available
        if league and league in self.league_profiles:
            league_btts = self.league_profiles[league]['btts_rate']
            # Blend with league tendency (80% current, 20% league)
            adjusted_prob = (adjusted_prob * 0.8) + (league_btts * 0.2)
        
        # Check if teams have strong clean sheet tendencies
        home_goals = self.calculate_team_goal_tendency(home_team, home=True)
        away_goals = self.calculate_team_goal_tendency(away_team, home=False)
        
        # If either team rarely scores, reduce BTTS probability
        if home_goals['scored'] < 0.8 or away_goals['scored'] < 0.8:
            adjusted_prob *= 0.85
        
        # If both teams score frequently, increase BTTS probability
        if home_goals['scored'] > 1.5 and away_goals['scored'] > 1.2:
            adjusted_prob = min(adjusted_prob * 1.15, 0.95)
        
        return np.clip(adjusted_prob, 0.05, 0.95)
    
    def adjust_ou_probability(self, base_prob: float, line: float, home_team: str, 
                             away_team: str, over: bool = True, league: str = None) -> float:
        """
        NEW: Adjust Over/Under probability based on team scoring patterns
        """
        # Get team goal tendencies
        home_pattern = self.calculate_team_goal_tendency(home_team, home=True)
        away_pattern = self.calculate_team_goal_tendency(away_team, home=False)
        
        # Calculate expected total goals
        expected_goals = home_pattern['scored'] + away_pattern['scored']
        
        # Poisson-based adjustment
        from scipy import stats
        poisson_prob = 1 - stats.poisson.cdf(line, expected_goals)  # Probability of over
        
        if not over:
            poisson_prob = 1 - poisson_prob
        
        # Blend base with Poisson (60% base, 40% Poisson)
        adjusted_prob = (base_prob * 0.6) + (poisson_prob * 0.4)
        
        # League-specific adjustment
        if league and league in self.league_profiles:
            profile = self.league_profiles[league]
            
            if line == 2.5:
                league_rate = profile['over_2_5_rate']
            elif line == 1.5:
                league_rate = profile['over_1_5_rate']
            elif line == 3.5:
                league_rate = profile['over_3_5_rate']
            else:
                league_rate = 0.5
            
            if not over:
                league_rate = 1 - league_rate
            
            # Blend with league rate (85% current, 15% league)
            adjusted_prob = (adjusted_prob * 0.85) + (league_rate * 0.15)
        
        return np.clip(adjusted_prob, 0.05, 0.95)
    
    def identify_value_bets(self, predictions: pd.DataFrame, min_edge: float = 0.05) -> pd.DataFrame:
        """
        NEW: Identify value bets by comparing predictions to market odds
        """
        value_bets = []
        
        for _, row in predictions.iterrows():
            # Check BTTS market
            if 'BLEND_BTTS_Y' in row and 'B365BTTS_Y' in row:
                our_prob = row['BLEND_BTTS_Y']
                market_odds = row.get('B365BTTS_Y', 0)
                if market_odds > 0:
                    implied_prob = 1 / market_odds
                    edge = our_prob - implied_prob
                    if edge > min_edge:
                        value_bets.append({
                            'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                            'Market': 'Over 2.5',
                            'Our_Prob': our_prob,
                            'Market_Prob': implied_prob,
                            'Edge': edge,
                            'Kelly': self.kelly_criterion(our_prob, market_odds)
                        })
        
        return pd.DataFrame(value_bets).sort_values('Edge', ascending=False)
    
    def kelly_criterion(self, prob: float, odds: float, fraction: float = 0.25) -> float:
        """
        NEW: Calculate Kelly Criterion stake size (fractional Kelly for safety)
        """
        if odds <= 1:
            return 0
        
        q = 1 - prob  # Probability of losing
        b = odds - 1  # Net odds received on the wager
        
        kelly = (prob * b - q) / b
        
        # Use fractional Kelly for safety (default 25%)
        return max(0, kelly * fraction)
    
    def enhance_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Main method to enhance BTTS/OU predictions
        """
        enhanced = predictions_df.copy()
        
        # Analyze league patterns first
        self.analyze_league_patterns()
        
        for idx, row in enhanced.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            league = row.get('League', None)
            
            # Enhance BTTS predictions
            if 'BLEND_BTTS_Y' in row:
                base_btts_yes = row['BLEND_BTTS_Y']
                adjusted_btts_yes = self.adjust_btts_probability(
                    base_btts_yes, home_team, away_team, league
                )
                enhanced.at[idx, 'BLEND_BTTS_Y'] = adjusted_btts_yes
                enhanced.at[idx, 'BLEND_BTTS_N'] = 1 - adjusted_btts_yes
            
            # Enhance Over/Under predictions
            for line in ['1_5', '2_5', '3_5']:
                over_col = f'BLEND_OU_{line}_O'
                under_col = f'BLEND_OU_{line}_U'
                
                if over_col in row:
                    base_over = row[over_col]
                    line_value = float(line.replace('_', '.'))
                    
                    adjusted_over = self.adjust_ou_probability(
                        base_over, line_value, home_team, away_team, 
                        over=True, league=league
                    )
                    
                    enhanced.at[idx, over_col] = adjusted_over
                    enhanced.at[idx, under_col] = 1 - adjusted_over
        
        return enhanced


class Match1X2Optimizer:
    """
    NEW: Specialized optimizer for 1X2 market predictions
    """
    
    def __init__(self, historical_data: pd.DataFrame = None):
        self.historical_data = historical_data
        self.home_advantage = {}
        self.team_form = {}
    
    def calculate_home_advantage(self, league: str = None) -> float:
        """
        NEW: Calculate home advantage factor for league or overall
        """
        if self.historical_data is None:
            return 0.1  # Default 10% home advantage
        
        data = self.historical_data
        if league:
            data = data[data['League'] == league]
        
        if len(data) < 50:
            return 0.1
        
        home_wins = (data['FTR'] == 'H').mean()
        away_wins = (data['FTR'] == 'A').mean()
        
        # Home advantage as the difference
        advantage = home_wins - away_wins
        
        return max(0, min(0.3, advantage))  # Cap between 0 and 30%
    
    def calculate_team_form(self, team: str, last_n: int = 5) -> Dict:
        """
        NEW: Calculate recent form metrics for a team
        """
        if self.historical_data is None:
            return {'win_rate': 0.33, 'draw_rate': 0.33, 'loss_rate': 0.33, 'points_per_game': 1.0}
        
        # Get last N games for team
        team_games = self.historical_data[
            (self.historical_data['HomeTeam'] == team) | 
            (self.historical_data['AwayTeam'] == team)
        ].tail(last_n)
        
        if team_games.empty:
            return {'win_rate': 0.33, 'draw_rate': 0.33, 'loss_rate': 0.33, 'points_per_game': 1.0}
        
        wins = 0
        draws = 0
        losses = 0
        points = 0
        
        for _, game in team_games.iterrows():
            if game['HomeTeam'] == team:
                if game['FTR'] == 'H':
                    wins += 1
                    points += 3
                elif game['FTR'] == 'D':
                    draws += 1
                    points += 1
                else:
                    losses += 1
            else:  # Away team
                if game['FTR'] == 'A':
                    wins += 1
                    points += 3
                elif game['FTR'] == 'D':
                    draws += 1
                    points += 1
                else:
                    losses += 1
        
        n = len(team_games)
        return {
            'win_rate': wins / n,
            'draw_rate': draws / n,
            'loss_rate': losses / n,
            'points_per_game': points / n
        }
    
    def adjust_1x2_probabilities(self, base_probs: Dict[str, float], 
                                 home_team: str, away_team: str, 
                                 league: str = None) -> Dict[str, float]:
        """
        NEW: Adjust 1X2 probabilities based on form and home advantage
        """
        # Get team forms
        home_form = self.calculate_team_form(home_team)
        away_form = self.calculate_team_form(away_team)
        
        # Calculate form difference
        form_diff = home_form['points_per_game'] - away_form['points_per_game']
        
        # Get home advantage
        home_adv = self.calculate_home_advantage(league)
        
        # Start with base probabilities
        adjusted = base_probs.copy()
        
        # Apply form-based adjustment
        form_adjustment = form_diff * 0.1  # 10% adjustment per point difference
        
        # Apply home advantage
        total_adjustment = form_adjustment + home_adv
        
        # Redistribute probabilities
        if total_adjustment > 0:  # Favor home team
            transfer = min(adjusted['A'] * 0.3, total_adjustment)
            adjusted['H'] += transfer * 0.7
            adjusted['D'] += transfer * 0.3
            adjusted['A'] -= transfer
        elif total_adjustment < 0:  # Favor away team
            transfer = min(adjusted['H'] * 0.3, abs(total_adjustment))
            adjusted['A'] += transfer * 0.7
            adjusted['D'] += transfer * 0.3
            adjusted['H'] -= transfer
        
        # Normalize to sum to 1
        total = sum(adjusted.values())
        for key in adjusted:
            adjusted[key] /= total
        
        # Apply slight draw bias correction (draws are often underestimated)
        if adjusted['D'] < 0.2 and home_form['draw_rate'] > 0.3 and away_form['draw_rate'] > 0.3:
            # Both teams draw frequently, boost draw probability
            draw_boost = 0.05
            adjusted['D'] += draw_boost
            adjusted['H'] -= draw_boost * 0.5
            adjusted['A'] -= draw_boost * 0.5
        
        # Ensure probabilities are valid
        for key in adjusted:
            adjusted[key] = max(0.01, min(0.98, adjusted[key]))
        
        # Renormalize
        total = sum(adjusted.values())
        for key in adjusted:
            adjusted[key] /= total
        
        return adjusted


def optimize_weekly_predictions(predictions_file: Path, historical_file: Path, output_file: Path):
    """
    Main function to optimize weekly predictions
    """
    # Load data
    predictions = pd.read_csv(predictions_file)
    historical = pd.read_csv(historical_file) if historical_file.exists() else None
    
    print("Optimizing BTTS and Over/Under predictions...")
    
    # Initialize optimizers
    btts_ou_optimizer = BTTSOUOptimizer(historical)
    match_optimizer = Match1X2Optimizer(historical)
    
    # Enhance predictions
    enhanced = predictions.copy()
    
    # Optimize each row
    for idx, row in enhanced.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        league = row.get('League', None)
        
        # Optimize 1X2
        if all(col in row for col in ['BLEND_1X2_H', 'BLEND_1X2_D', 'BLEND_1X2_A']):
            base_1x2 = {
                'H': row['BLEND_1X2_H'],
                'D': row['BLEND_1X2_D'],
                'A': row['BLEND_1X2_A']
            }
            
            adjusted_1x2 = match_optimizer.adjust_1x2_probabilities(
                base_1x2, home_team, away_team, league
            )
            
            enhanced.at[idx, 'BLEND_1X2_H'] = adjusted_1x2['H']
            enhanced.at[idx, 'BLEND_1X2_D'] = adjusted_1x2['D']
            enhanced.at[idx, 'BLEND_1X2_A'] = adjusted_1x2['A']
    
    # Apply BTTS/OU optimization
    enhanced = btts_ou_optimizer.enhance_predictions(enhanced)
    
    # Identify value bets
    if historical is not None:
        value_bets = btts_ou_optimizer.identify_value_bets(enhanced)
        if not value_bets.empty:
            print("\nTop Value Bets Found:")
            print(value_bets.head(10))
    
    # Save enhanced predictions
    enhanced.to_csv(output_file, index=False)
    print(f"\nOptimized predictions saved to {output_file}")
    
    # Calculate improvement metrics
    print("\nOptimization Summary:")
    print(f"Total predictions enhanced: {len(enhanced)}")
    
    # Show league profiles if available
    if btts_ou_optimizer.league_profiles:
        print("\nLeague Profiles Learned:")
        for league, profile in btts_ou_optimizer.league_profiles.items():
            print(f"\n{league}:")
            print(f"  Avg Goals: {profile['avg_goals']:.2f}")
            print(f"  BTTS Rate: {profile['btts_rate']:.1%}")
            print(f"  Over 2.5 Rate: {profile['over_2_5_rate']:.1%}")
    
    return enhanced


if __name__ == "__main__":
    from pathlib import Path
    
    # Define paths
    OUTPUT_DIR = Path("outputs")
    predictions_file = OUTPUT_DIR / "weekly_bets_lite.csv"
    historical_file = Path("historical_data.csv")  # Your historical data
    output_file = OUTPUT_DIR / "weekly_bets_optimized.csv"
    
    if predictions_file.exists():
        optimize_weekly_predictions(predictions_file, historical_file, output_file)
    else:
        print("No predictions file found!")