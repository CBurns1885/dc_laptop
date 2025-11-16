# accumulator_finder.py
"""
Cross-League Accumulator Builder
Finds high-probability accumulator opportunities from weekly predictions
Excludes historically inaccurate markets like OU_2_5
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Tuple

OUTPUT_DIR = Path("outputs")

class AccumulatorFinder:
    """Find and rank accumulator betting opportunities"""
    
    def __init__(self, weekly_bets_path: Path):
        self.df = pd.read_csv(weekly_bets_path)
        self.opportunities = []
    
    def _extract_prob(self, val) -> float:
        """Convert probability value to float"""
        if pd.isna(val):
            return 0.0
        if isinstance(val, str) and '%' in val:
            return float(val.strip('%')) / 100
        return float(val)
    
    def find_high_confidence_bets(self, min_prob: float = 0.70) -> List[Dict]:
        """Extract all high-confidence predictions - EXCLUDES OU_2_5"""
        
        bets = []
        
        for idx, row in self.df.iterrows():
            match_info = {
                'league': row['League'],
                'date': row.get('Date', ''),
                'home': row['HomeTeam'],
                'away': row['AwayTeam'],
            }
            
            # Define markets with realistic odds - OU_2_5 EXCLUDED
            markets = {
                '1X2_H': ('BLEND_1X2_H', 'Home Win', 2.0),
                '1X2_D': ('BLEND_1X2_D', 'Draw', 3.5),
                '1X2_A': ('BLEND_1X2_A', 'Away Win', 2.5),
                'BTTS_Y': ('BLEND_BTTS_Y', 'BTTS Yes', 1.9),
                'BTTS_N': ('BLEND_BTTS_N', 'BTTS No', 1.8),
                #'OU_1_5_O': ('BLEND_OU_1_5_O', 'Over 1.5', 1.25),
                #'OU_1_5_U': ('BLEND_OU_1_5_U', 'Under 1.5', 3.5),
                #'OU_3_5_U': ('BLEND_OU_3_5_U', 'Under 3.5', 1.35),
                #'OU_4_5_U': ('BLEND_OU_4_5_U', 'Under 4.5', 1.15),
                
                }
            
            for market_key, (col_name, market_desc, realistic_odds) in markets.items():
                if col_name in row:
                    prob = self._extract_prob(row[col_name])
                    
                    if prob >= min_prob:
                        bets.append({
                            **match_info,
                            'match': f"{match_info['home']} vs {match_info['away']}",
                            'market': market_desc,
                            'probability': prob,
                            'odds': realistic_odds,
                            'selection': f"{match_info['home']} v {match_info['away']}: {market_desc}",
                        })
        
        return bets
    
    def build_accumulators(self, min_legs: int = 3, max_legs: int = 6,
                          min_total_prob: float = 0.30, min_prob: float = 0.70) -> List[Dict]:
        """Build accumulator combinations"""

        high_conf_bets = self.find_high_confidence_bets(min_prob=min_prob)
        
        if len(high_conf_bets) < min_legs:
            print(f"⚠️ Only {len(high_conf_bets)} high confidence bets found")
            return []
        
        if len(high_conf_bets) > 20:
            print(f"⚠️ Too many bets ({len(high_conf_bets)}), using top 20 only")
            high_conf_bets = sorted(high_conf_bets, key=lambda x: x['probability'], reverse=True)[:20]
        
        print(f"✅ Found {len(high_conf_bets)} high confidence bets")
        
        accumulators = []
        
        for num_legs in range(min_legs, min(max_legs + 1, len(high_conf_bets) + 1)):
            for combo in combinations(high_conf_bets, num_legs):
                legs = list(combo)
                
                matches = [leg['match'] for leg in legs]
                if len(matches) != len(set(matches)):
                    continue
                
                total_prob = np.prod([leg['probability'] for leg in legs])
                
                if total_prob < min_total_prob:
                    continue
                
                total_odds = np.prod([leg['odds'] for leg in legs])
                ev = (total_prob * total_odds - 1) * 100
                
                accumulators.append({
                    'legs': legs,
                    'num_legs': num_legs,
                    'total_probability': total_prob,
                    'total_odds': total_odds,
                    'expected_value': ev,
                    'leagues': ', '.join(sorted(set(leg['league'] for leg in legs))),
                })
        
        accumulators.sort(key=lambda x: (x['expected_value'], x['total_probability']), 
                         reverse=True)
        
        return accumulators[:30]
    
    def generate_html_report(self, accumulators: List[Dict], output_path: Path = None) -> str:
        """Generate HTML report"""
        
        if output_path is None:
            output_path = OUTPUT_DIR / "accumulators.html"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Accumulator Opportunities</title>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; }}
        .acca-card {{ background: white; margin: 15px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .acca-header {{ display: flex; justify-content: space-between; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #3498db; }}
        .acca-title {{ font-size: 18px; font-weight: bold; color: #2c3e50; }}
        .leg {{ margin: 8px 0; padding: 10px; background: #ecf0f1; border-radius: 5px; }}
        .leg-prob {{ float: right; background: #3498db; color: white; padding: 2px 8px; border-radius: 3px; }}
        .high-ev {{ background: #d4edda; }}
        .medium-ev {{ background: #fff3cd; }}
        .low-ev {{ background: #f8d7da; }}
    </style>
</head>
<body>
    <h1>⚽ Accumulator Opportunities</h1>
    <div style="background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <strong>📊 Total found:</strong> {len(accumulators)}<br>
        Based on ≥70% probability predictions<br>
        <strong>Note:</strong> Over/Under 2.5 excluded
    </div>
"""
        
        for i, acca in enumerate(accumulators, 1):
            ev_class = 'high-ev' if acca['expected_value'] > 0 else 'medium-ev' if acca['expected_value'] > -10 else 'low-ev'
            
            html += f"""
    <div class="acca-card {ev_class}">
        <div class="acca-header">
            <div class="acca-title">#{i} - {acca['num_legs']}-Fold</div>
            <div>Odds: {acca['total_odds']:.2f} | Prob: {acca['total_probability']:.1%} | EV: {acca['expected_value']:+.1f}%</div>
        </div>
        <div><strong>Leagues:</strong> {acca['leagues']}</div>
"""
            for leg in acca['legs']:
                html += f'        <div class="leg"><span>{leg["match"]} - {leg["market"]}</span><span class="leg-prob">{leg["probability"]:.0%}</span></div>\n'
            
            html += "    </div>\n"
        
        html += "</body></html>"
        
        output_path.write_text(html, encoding='utf-8')
        print(f"✅ Generated: {output_path}")
        return str(output_path)
    
    def export_csv(self, accumulators: List[Dict], output_path: Path = None):
        """Export to CSV"""
        if output_path is None:
            output_path = OUTPUT_DIR / "accumulators.csv"
        
        rows = []
        for i, acca in enumerate(accumulators, 1):
            legs_str = " | ".join([f"{leg['selection']}" for leg in acca['legs']])
            rows.append({
                'Acca_ID': i,
                'Num_Legs': acca['num_legs'],
                'Total_Odds': round(acca['total_odds'], 2),
                'Probability': f"{acca['total_probability']:.1%}",
                'Expected_Value': f"{acca['expected_value']:+.1f}%",
                'Leagues': acca['leagues'],
                'Selections': legs_str,
            })
        
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"✅ Generated: {output_path}")


def generate_accumulators(csv_path: Path = None):
    """Main function"""
    if csv_path is None:
        csv_path = OUTPUT_DIR / "weekly_bets_lite.csv"
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    print("\n🎰 ACCUMULATOR FINDER")
    print("="*50)
    
    finder = AccumulatorFinder(csv_path)
    accumulators = finder.build_accumulators(min_legs=3, max_legs=6, min_total_prob=0.30)
    
    if not accumulators:
        print("❌ No suitable accumulators found")
        return
    
    print(f"\n✅ Found {len(accumulators)} opportunities")
    
    finder.generate_html_report(accumulators)
    finder.export_csv(accumulators)
    
    print(f"\n🏆 TOP 5:")
    for i, acca in enumerate(accumulators[:5], 1):
        print(f"\n{i}. {acca['num_legs']}-Fold - Odds: {acca['total_odds']:.2f} - Prob: {acca['total_probability']:.1%} - EV: {acca['expected_value']:+.1f}%")
        for leg in acca['legs']:
            print(f"   • {leg['selection']} ({leg['probability']:.0%})")



if __name__ == "__main__":
    generate_accumulators()