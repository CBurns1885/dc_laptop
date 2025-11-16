#!/usr/bin/env python3
"""
BET FINDER - Automatically find high-quality betting opportunities
Filters weekly_bets.csv based on:
- Probability thresholds
- Confidence scores
- Model agreement
- Dixon-Coles validation
- Cross-market logic checks
ALL MARKETS INCLUDED
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================
# Set output directory to today's date (YYYY-MM-DD format)
today = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = Path(f"outputs/{today}")
WEEKLY_BETS_FILE = OUTPUT_DIR / "weekly_bets_lite.csv"

#yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
#OUTPUT_DIR = Path(f"outputs/{yesterday}")


# Betting Criteria - Adjust these to be more/less strict
CONFIG = {
    # Probability thresholds (minimum confidence in prediction)
    'min_prob_1x2': 0.60,       # 60% for match result (home/away)
    'min_prob_btts': 0.65,      # 65% for both teams score
    'min_prob_ou': 0.65,        # 65% for over/under (all lines)
    'min_prob_draw': 0.30,      # 30% for draws (lower because draws are rarer)
    'min_prob_ah': 0.60,        # 60% for Asian handicaps
    'min_prob_cs': 0.15,        # 15% for correct scores (very specific)
    'min_prob_ht': 0.40,        # 40% for half-time results
    'min_prob_htft': 0.20,      # 20% for HT/FT (very specific)
    'min_prob_team_goals': 0.65, # 65% for team goal lines
    'min_prob_cards': 0.40,     # 40% for cards (unpredictable)
    'min_prob_corners': 0.35,   # 35% for corners (very unpredictable)
    
    # Confidence thresholds (model certainty)
    'min_confidence': 0.60,     # 60% confidence minimum
    
    # Agreement thresholds (model consensus)
    'min_agreement': 70.0,      # 70% agreement between models
    
    # Dixon-Coles validation
    'dc_validation': True,      # Check DC model agrees
    'dc_max_diff': 0.15,        # DC can differ by max 15% from main model
    
    # Value betting
    'calculate_value': True,    # Calculate implied odds and value
    'min_value_threshold': 0.05, # 5% minimum edge for value bets
}

# ============================================================================
# BET FINDER CLASS
# ============================================================================

class BetFinder:
    def __init__(self, config=CONFIG):
        self.config = config
        self.df = None
        self.quality_bets = []
        
    def load_data(self, filepath=WEEKLY_BETS_FILE):
        """Load weekly_bets.csv"""
        print(f"Loading {filepath}...")
        
        if not filepath.exists():
            print(f"❌ Error: {filepath} not found!")
            print(f"   Make sure you've run the weekly pipeline first.")
            return False
            
        self.df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(self.df)} matches")
        print(f"   Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        return True
    
    # ========================================================================
    # 1X2 MARKETS
    # ========================================================================
    
    def check_home_win(self, row):
        """Check if home win meets criteria"""
        prob = row.get('P_1X2_H', 0)
        conf = row.get('CONF_1X2_H', 0)
        agree = row.get('AGREE_1X2_H', 0)
        dc_prob = row.get('DC_1X2_H', 0)
        
        if prob < self.config['min_prob_1x2']:
            return None
        if conf < self.config['min_confidence']:
            return None
        if agree < self.config['min_agreement']:
            return None
            
        if self.config['dc_validation'] and pd.notna(dc_prob):
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None
        
        return {
            'Market': 'Home Win',
            'Selection': row['HomeTeam'],
            'Probability': prob,
            'Confidence': conf,
            'Agreement': agree,
            'DC_Probability': dc_prob if pd.notna(dc_prob) else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    def check_away_win(self, row):
        """Check if away win meets criteria"""
        prob = row.get('P_1X2_A', 0)
        conf = row.get('CONF_1X2_A', 0)
        agree = row.get('AGREE_1X2_A', 0)
        dc_prob = row.get('DC_1X2_A', 0)
        
        if prob < self.config['min_prob_1x2']:
            return None
        if conf < self.config['min_confidence']:
            return None
        if agree < self.config['min_agreement']:
            return None
            
        if self.config['dc_validation'] and pd.notna(dc_prob):
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None
        
        return {
            'Market': 'Away Win',
            'Selection': row['AwayTeam'],
            'Probability': prob,
            'Confidence': conf,
            'Agreement': agree,
            'DC_Probability': dc_prob if pd.notna(dc_prob) else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    def check_draw(self, row):
        """Check if draw meets criteria"""
        prob = row.get('P_1X2_D', 0)
        conf = row.get('CONF_1X2_D', 0)
        agree = row.get('AGREE_1X2_D', 0)
        dc_prob = row.get('DC_1X2_D', 0)
        
        if prob < self.config['min_prob_draw']:
            return None
        if conf < self.config['min_confidence']:
            return None
        if agree < self.config['min_agreement']:
            return None
            
        if self.config['dc_validation'] and pd.notna(dc_prob):
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None
        
        return {
            'Market': 'Draw',
            'Selection': 'X',
            'Probability': prob,
            'Confidence': conf,
            'Agreement': agree,
            'DC_Probability': dc_prob if pd.notna(dc_prob) else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # BTTS MARKETS
    # ========================================================================
    
    def check_btts_yes(self, row):
        """Check if BTTS Yes meets criteria"""
        prob = row.get('P_BTTS_Y', 0)
        conf = row.get('CONF_BTTS_Y', 0)
        agree = row.get('AGREE_BTTS_Y', 0)
        dc_prob = row.get('DC_BTTS_Y', 0)
        
        if prob < self.config['min_prob_btts']:
            return None
        if conf < self.config['min_confidence']:
            return None
        if agree < self.config['min_agreement']:
            return None
            
        if self.config['dc_validation'] and pd.notna(dc_prob):
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None
        
        ou25_over = row.get('P_OU_2_5_O', 0)
        if prob > 0.7 and ou25_over < 0.5:
            return None
        
        return {
            'Market': 'BTTS',
            'Selection': 'Yes',
            'Probability': prob,
            'Confidence': conf,
            'Agreement': agree,
            'DC_Probability': dc_prob if pd.notna(dc_prob) else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    def check_btts_no(self, row):
        """Check if BTTS No meets criteria"""
        prob = row.get('P_BTTS_N', 0)
        conf = row.get('CONF_BTTS_N', 0)
        agree = row.get('AGREE_BTTS_N', 0)
        dc_prob = row.get('DC_BTTS_N', 0)
        
        if prob < self.config['min_prob_btts']:
            return None
        if conf < self.config['min_confidence']:
            return None
        if agree < self.config['min_agreement']:
            return None
            
        if self.config['dc_validation'] and pd.notna(dc_prob):
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None
        
        return {
            'Market': 'BTTS',
            'Selection': 'No',
            'Probability': prob,
            'Confidence': conf,
            'Agreement': agree,
            'DC_Probability': dc_prob if pd.notna(dc_prob) else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # OVER/UNDER MARKETS (All Lines)
    # ========================================================================
    
    def check_ou_market(self, row, line, over_under):
        """Generic checker for any O/U line"""
        ou_type = 'O' if over_under == 'Over' else 'U'
        prob = row.get(f'P_OU_{line}_{ou_type}', 0)
        
        # No CONF/AGREE for non-2.5 lines, so skip those checks
        if line == '2_5':
            conf_col = f'CONF_OU_2_5_{ou_type}'
            agree_col = f'AGREE_OU_2_5_{ou_type}'
            conf = row.get(conf_col, 0)
            agree = row.get(agree_col, 0)
            
            if conf < self.config['min_confidence']:
                return None
            if agree < self.config['min_agreement']:
                return None
        else:
            conf = None
            agree = None
        
        if prob < self.config['min_prob_ou']:
            return None
            
        dc_prob = row.get(f'DC_OU_{line}_{ou_type}', 0)
        if self.config['dc_validation'] and pd.notna(dc_prob):
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None
        
        return {
            'Market': f'O/U {line.replace("_", ".")}',
            'Selection': over_under,
            'Probability': prob,
            'Confidence': conf if conf else prob,  # Use probability as proxy if no confidence
            'Agreement': agree if agree else prob * 100,  # Use probability as proxy
            'DC_Probability': dc_prob if pd.notna(dc_prob) else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # ASIAN HANDICAP MARKETS
    # ========================================================================
    
    def check_ah_market(self, row, line, selection):
        """Generic checker for Asian Handicap"""
        prob = row.get(f'P_AH_{line}_{selection}', 0)
        
        if prob < self.config['min_prob_ah']:
            return None
        
        dc_prob = row.get(f'DC_AH_{line}_{selection}', 0)
        if self.config['dc_validation'] and pd.notna(dc_prob):
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None
        
        selection_name = {'H': 'Home', 'A': 'Away', 'P': 'Push'}[selection]
        
        return {
            'Market': f'AH {line.replace("_", ".")}',
            'Selection': selection_name,
            'Probability': prob,
            'Confidence': prob,  # Use probability as proxy
            'Agreement': prob * 100,
            'DC_Probability': dc_prob if pd.notna(dc_prob) else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # HALF TIME MARKETS
    # ========================================================================
    
    def check_ht_market(self, row, outcome):
        """Check half-time result"""
        prob = row.get(f'P_HT_{outcome}', 0)
        
        if prob < self.config['min_prob_ht']:
            return None
        
        outcome_name = {'H': 'Home', 'D': 'Draw', 'A': 'Away'}[outcome]
        
        return {
            'Market': 'Half Time',
            'Selection': outcome_name,
            'Probability': prob,
            'Confidence': prob,
            'Agreement': prob * 100,
            'DC_Probability': None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # HT/FT MARKETS
    # ========================================================================
    
    def check_htft_market(self, row, ht_outcome, ft_outcome):
        """Check half-time/full-time double result"""
        prob = row.get(f'P_HTFT_{ht_outcome}_{ft_outcome}', 0)
        
        if prob < self.config['min_prob_htft']:
            return None
        
        ht_name = {'H': 'Home', 'D': 'Draw', 'A': 'Away'}[ht_outcome]
        ft_name = {'H': 'Home', 'D': 'Draw', 'A': 'Away'}[ft_outcome]
        
        return {
            'Market': 'HT/FT',
            'Selection': f'{ht_name}/{ft_name}',
            'Probability': prob,
            'Confidence': prob,
            'Agreement': prob * 100,
            'DC_Probability': None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # TEAM GOAL LINES
    # ========================================================================
    
    def check_team_goals(self, row, team, line, over_under):
        """Check individual team goal lines"""
        ou_type = 'O' if over_under == 'Over' else 'U'
        prob = row.get(f'P_{team}TG_{line}_{ou_type}', 0)
        
        if prob < self.config['min_prob_team_goals']:
            return None
        
        team_name = 'Home' if team == 'Home' else 'Away'
        
        return {
            'Market': f'{team_name} Team Goals {line.replace("_", ".")}',
            'Selection': over_under,
            'Probability': prob,
            'Confidence': prob,
            'Agreement': prob * 100,
            'DC_Probability': None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # CORRECT SCORE
    # ========================================================================
    
    def check_correct_score(self, row, home_goals, away_goals):
        """Check specific correct score"""
        if home_goals == 'Other':
            prob = row.get('P_CS_Other', 0)
            score = 'Other'
        else:
            prob = row.get(f'P_CS_{home_goals}_{away_goals}', 0)
            score = f'{home_goals}-{away_goals}'
        
        if prob < self.config['min_prob_cs']:
            return None
        
        dc_prob = row.get(f'DC_CS_{home_goals}_{away_goals}' if home_goals != 'Other' else 'DC_CS_Other', 0)
        if self.config['dc_validation'] and pd.notna(dc_prob):
            if abs(prob - dc_prob) > self.config['dc_max_diff']:
                return None
        
        return {
            'Market': 'Correct Score',
            'Selection': score,
            'Probability': prob,
            'Confidence': prob,
            'Agreement': prob * 100,
            'DC_Probability': dc_prob if pd.notna(dc_prob) else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # GOAL RANGE
    # ========================================================================
    
    def check_goal_range(self, row, goals):
        """Check total goals in range"""
        prob = row.get(f'P_GR_{goals}', 0)
        
        if prob < self.config['min_prob_cs']:  # Use CS threshold (specific outcome)
            return None
        
        dc_prob = row.get(f'DC_GR_{goals}', 0)
        
        goals_display = f'{goals}+' if goals == '5' else goals
        
        return {
            'Market': 'Goal Range',
            'Selection': f'{goals_display} goals',
            'Probability': prob,
            'Confidence': prob,
            'Agreement': prob * 100,
            'DC_Probability': dc_prob if pd.notna(dc_prob) else None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # CARDS MARKETS
    # ========================================================================
    
    def check_cards(self, row, team, card_range):
        """Check yellow cards for home or away team"""
        prob = row.get(f'P_{team}CardsY_{card_range}', 0)
        
        if prob < self.config.get('min_prob_cards', 0.40):  # Lower threshold for cards
            return None
        
        team_name = 'Home' if team == 'Home' else 'Away'
        
        return {
            'Market': f'{team_name} Yellow Cards',
            'Selection': card_range,
            'Probability': prob,
            'Confidence': prob,
            'Agreement': prob * 100,
            'DC_Probability': None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # CORNERS MARKETS
    # ========================================================================
    
    def check_corners(self, row, team, corner_range):
        """Check corners for home or away team"""
        prob = row.get(f'P_{team}Corners_{corner_range}', 0)
        
        if prob < self.config.get('min_prob_corners', 0.35):  # Lower threshold for corners
            return None
        
        team_name = 'Home' if team == 'Home' else 'Away'
        
        return {
            'Market': f'{team_name} Corners',
            'Selection': corner_range,
            'Probability': prob,
            'Confidence': prob,
            'Agreement': prob * 100,
            'DC_Probability': None,
            'ImpliedOdds': round(1 / prob, 2) if prob > 0 else None,
        }
    
    # ========================================================================
    # FIND QUALITY BETS
    # ========================================================================
    
    def find_quality_bets(self):
        """Scan all matches and find quality betting opportunities"""
        print("\n" + "="*60)
        print("🔍 SCANNING FOR QUALITY BETS (ALL MARKETS)")
        print("="*60)
        
        self.quality_bets = []
        
        for idx, row in self.df.iterrows():
            match_info = {
                'League': row['League'],
                'Date': row['Date'],
                'HomeTeam': row['HomeTeam'],
                'AwayTeam': row['AwayTeam'],
            }
            
            # 1X2 Markets
            checkers = [
                self.check_home_win,
                self.check_away_win,
                self.check_draw,
            ]
            
            # BTTS Markets
            checkers.extend([
                self.check_btts_yes,
                self.check_btts_no,
            ])
            
            # Over/Under Markets (all lines)
            for line in ['0_5', '1_5', '2_5', '3_5', '4_5']:
                checkers.append(lambda r, l=line: self.check_ou_market(r, l, 'Over'))
                checkers.append(lambda r, l=line: self.check_ou_market(r, l, 'Under'))
            
            # Asian Handicap Markets
            for line in ['-1_0', '-0_5', '0_0', '+0_5', '+1_0']:
                for selection in ['H', 'A', 'P']:
                    checkers.append(lambda r, l=line, s=selection: self.check_ah_market(r, l, s))
            
            # Half Time Markets
            for outcome in ['H', 'D', 'A']:
                checkers.append(lambda r, o=outcome: self.check_ht_market(r, o))
            
            # HT/FT Markets
            for ht in ['H', 'D', 'A']:
                for ft in ['H', 'D', 'A']:
                    checkers.append(lambda r, h=ht, f=ft: self.check_htft_market(r, h, f))
            
            # Team Goal Lines
            for team in ['Home', 'Away']:
                for line in ['0_5', '1_5', '2_5', '3_5']:
                    checkers.append(lambda r, t=team, l=line: self.check_team_goals(r, t, l, 'Over'))
                    checkers.append(lambda r, t=team, l=line: self.check_team_goals(r, t, l, 'Under'))
            
            # Correct Scores (most common ones)
            for home in range(6):
                for away in range(6):
                    checkers.append(lambda r, h=home, a=away: self.check_correct_score(r, str(h), str(a)))
            checkers.append(lambda r: self.check_correct_score(r, 'Other', None))
            
            # Goal Range
            for goals in ['0', '1', '2', '3', '4', '5']:
                checkers.append(lambda r, g=goals: self.check_goal_range(r, g))
            
            # Cards Markets
            for team in ['Home', 'Away']:
                for card_range in ['0-2', '3', '4-5', '6+']:
                    checkers.append(lambda r, t=team, c=card_range: self.check_cards(r, t, c))
            
            # Corners Markets
            for team in ['Home', 'Away']:
                for corner_range in ['0-3', '4-5', '6-7', '8-9', '10+']:
                    checkers.append(lambda r, t=team, c=corner_range: self.check_corners(r, t, c))
            
            # Run all checkers
            for checker in checkers:
                try:
                    bet = checker(row)
                    if bet:
                        bet.update(match_info)
                        self.quality_bets.append(bet)
                except Exception as e:
                    # Silently skip any checker errors
                    pass
        
        print(f"✅ Found {len(self.quality_bets)} quality betting opportunities")
        return self.quality_bets
    
    def calculate_kelly_stake(self, probability, odds):
        """Calculate Kelly Criterion stake percentage"""
        if odds <= 1.0:
            return 0.0
        
        kelly = (probability * odds - 1) / (odds - 1)
        return max(0, kelly * 0.25)
    
    def generate_report(self):
        """Generate detailed betting report"""
        if not self.quality_bets:
            print("\n❌ No quality bets found with current criteria.")
            print("💡 Try adjusting the CONFIG thresholds to be less strict.")
            return
        
        # Convert to DataFrame
        bets_df = pd.DataFrame(self.quality_bets)
        
        # Sort by confidence and probability
        bets_df['Score'] = (bets_df['Probability'] * 0.5 +
                            bets_df['Confidence'] * 0.3 +
                            bets_df['Agreement'] / 100 * 0.2)
        bets_df = bets_df.sort_values('Score', ascending=False)

        # Add Kelly stakes
        if self.config['calculate_value']:
            bets_df['Kelly%'] = bets_df.apply(
                lambda x: self.calculate_kelly_stake(x['Probability'], x['ImpliedOdds']),
                axis=1
            )

        # Sort by Date first for better readability, then by Score
        if 'Date' in bets_df.columns:
            import pandas as pd
            bets_df['Date'] = pd.to_datetime(bets_df['Date'], errors='coerce')
            bets_df = bets_df.sort_values(['Date', 'Score'], ascending=[True, False])
            print("✅ Sorted quality bets by Date, then by Score")

        # Display summary
        print("\n" + "="*60)
        print("📊 BETTING SUMMARY")
        print("="*60)
        print(f"Total quality bets: {len(bets_df)}")
        print(f"Matches covered: {bets_df[['HomeTeam', 'AwayTeam']].drop_duplicates().shape[0]}")
        print(f"\nBreakdown by market:")
        print(bets_df['Market'].value_counts().to_string())

        # Save to CSV
        output_file = OUTPUT_DIR / f"quality_bets_{datetime.now().strftime('%Y%m%d')}.csv"
        bets_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")
        
        # Save to HTML
        html_file = OUTPUT_DIR / f"quality_bets_{datetime.now().strftime('%Y%m%d')}.html"
        self.generate_html_report(bets_df, html_file)
        print(f"💾 Saved HTML: {html_file}")
        
        # Display top 20
        print("\n" + "="*60)
        print("🏆 TOP 20 QUALITY BETS")
        print("="*60)
        
        display_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Market', 'Selection', 
                       'Probability', 'Confidence', 'Agreement', 'ImpliedOdds']
        if 'Kelly%' in bets_df.columns:
            display_cols.append('Kelly%')
        
        top20 = bets_df.head(20)[display_cols]
        
        # Format for display
        top20_display = top20.copy()
        top20_display['Probability'] = top20_display['Probability'].apply(lambda x: f"{x:.1%}")
        top20_display['Confidence'] = top20_display['Confidence'].apply(lambda x: f"{x:.1%}")
        top20_display['Agreement'] = top20_display['Agreement'].apply(lambda x: f"{x:.0f}%")
        if 'Kelly%' in top20_display.columns:
            top20_display['Kelly%'] = top20_display['Kelly%'].apply(lambda x: f"{x:.1%}")
        
        print(top20_display.to_string(index=False))
        
        return bets_df
    
    def generate_html_report(self, df, output_path):
        """Generate beautiful HTML report"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Bets - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0;
            font-size: 2em;
        }}
        .stat-card p {{
            margin: 5px 0 0 0;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 0.9em;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .prob-high {{ color: #10b981; font-weight: bold; }}
        .prob-medium {{ color: #f59e0b; font-weight: bold; }}
        .conf-high {{ background: #d1fae5; color: #065f46; padding: 4px 8px; border-radius: 4px; }}
        .conf-medium {{ background: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 4px; }}
        .market {{ 
            background: #f3f4f6; 
            padding: 4px 8px; 
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85em;
        }}
        .odds {{ 
            background: #dbeafe; 
            padding: 4px 8px; 
            border-radius: 4px;
            font-weight: bold;
        }}
        .kelly {{
            background: #fce7f3;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚽ Quality Betting Opportunities - ALL MARKETS</h1>
        <p class="subtitle">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        
        <div class="stats">
            <div class="stat-card">
                <h3>{len(df)}</h3>
                <p>Quality Bets Found</p>
            </div>
            <div class="stat-card">
                <h3>{df[['HomeTeam', 'AwayTeam']].drop_duplicates().shape[0]}</h3>
                <p>Matches Covered</p>
            </div>
            <div class="stat-card">
                <h3>{df['Probability'].mean():.0%}</h3>
                <p>Avg Probability</p>
            </div>
            <div class="stat-card">
                <h3>{len(df['Market'].unique())}</h3>
                <p>Different Markets</p>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Match</th>
                    <th>League</th>
                    <th>Market</th>
                    <th>Selection</th>
                    <th>Probability</th>
                    <th>Confidence</th>
                    <th>Agreement</th>
                    <th>Implied Odds</th>
                    {'<th>Kelly %</th>' if 'Kelly%' in df.columns else ''}
                </tr>
            </thead>
            <tbody>
"""
        
        for _, row in df.iterrows():
            prob_class = 'prob-high' if row['Probability'] > 0.70 else 'prob-medium'
            conf_class = 'conf-high' if row['Confidence'] > 0.70 else 'conf-medium'
            
            kelly_cell = f"<td><span class='kelly'>{row['Kelly%']:.1%}</span></td>" if 'Kelly%' in df.columns else ''
            
            html += f"""
                <tr>
                    <td>{row['Date']}</td>
                    <td><strong>{row['HomeTeam']}</strong> vs <strong>{row['AwayTeam']}</strong></td>
                    <td>{row['League']}</td>
                    <td><span class="market">{row['Market']}</span></td>
                    <td><strong>{row['Selection']}</strong></td>
                    <td class="{prob_class}">{row['Probability']:.1%}</td>
                    <td><span class="{conf_class}">{row['Confidence']:.1%}</span></td>
                    <td>{row['Agreement']:.0f}%</td>
                    <td><span class="odds">{row['ImpliedOdds']:.2f}</span></td>
                    {kelly_cell}
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        
        <div class="footer">
            <p><strong>⚠️ Betting Disclaimer:</strong> These are predictions based on statistical models. 
            Past performance does not guarantee future results. Bet responsibly.</p>
            <p><strong>💡 How to use:</strong> Look for bets with high probability, high confidence, 
            and high agreement. Compare implied odds with bookmaker odds to find value.</p>
            <p><strong>📊 Markets included:</strong> 1X2, BTTS, O/U (all lines), Asian Handicaps, 
            Half Time, HT/FT, Team Goals, Correct Scores, Goal Ranges, Yellow Cards, Corners</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def run(self):
        """Main execution flow"""
        print("="*60)
        print("⚽ BET FINDER - Quality Betting Opportunities (ALL MARKETS)")
        print("="*60)
        
        if not self.load_data():
            return
        
        self.find_quality_bets()
        
        if self.quality_bets:
            self.generate_report()
            
            print("\n" + "="*60)
            print("✅ BET FINDER COMPLETE")
            print("="*60)
            print("\n💡 Next steps:")
            print("   1. Review the HTML report in your browser")
            print("   2. Compare implied odds with bookmaker odds")
            print("   3. Look for value bets (bookmaker odds > implied odds)")
            print("   4. Use Kelly % for stake sizing (bet conservatively!)")
            print("   5. Track your bets in a spreadsheet")
        else:
            print("\n💡 Tips to find more bets:")
            print("   - Lower min_prob thresholds in CONFIG")
            print("   - Lower min_confidence to 0.55")
            print("   - Lower min_agreement to 65")
            print("   - Set dc_validation to False")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    finder = BetFinder(config=CONFIG)
    finder.run()
