"""
FINAL VALIDATION: 60-Day Data Analysis
Focus on Au Privave's predictive power
Then create comprehensive project summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp, binomtest
import warnings
warnings.filterwarnings('ignore')

def dtw_distance(s1, s2):
    """Calculate DTW distance"""
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                          dtw_matrix[i, j-1],
                                          dtw_matrix[i-1, j-1])
    
    return dtw_matrix[n, m]

def normalize_pattern(arr):
    """Normalize array to 0-1"""
    arr = np.array(arr)
    if arr.max() == arr.min():
        return arr * 0
    return (arr - arr.min()) / (arr.max() - arr.min())

print("="*80)
print("🎺 FINAL VALIDATION - 60 DAYS OF DATA")
print("="*80)

# Load 60-day 5-minute SPY data
print("\n📈 Loading 60-day SPY data...")
spy_raw = pd.read_csv('/mnt/user-data/uploads/SPY_intraday_5m_60d.csv', skiprows=2)
spy_raw.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
spy_raw['Datetime'] = pd.to_datetime(spy_raw['Datetime'])
spy_raw = spy_raw.sort_values('Datetime').reset_index(drop=True)

for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
    spy_raw[col] = pd.to_numeric(spy_raw[col], errors='coerce')

spy_raw = spy_raw.dropna()

# Calculate features
spy_raw['price_range'] = spy_raw['High'] - spy_raw['Low']
spy_raw['volume_norm'] = (spy_raw['Volume'] - spy_raw['Volume'].min()) / (spy_raw['Volume'].max() - spy_raw['Volume'].min())
spy_raw['price_range_norm'] = (spy_raw['price_range'] - spy_raw['price_range'].min()) / (spy_raw['price_range'].max() - spy_raw['price_range'].min())

spy = spy_raw.dropna().reset_index(drop=True)

print(f"SPY data: {len(spy):,} 5-minute bars")
print(f"Date range: {spy['Datetime'].min()} to {spy['Datetime'].max()}")
print(f"Trading days: {spy['Datetime'].dt.date.nunique()}")

# Load jazz data
jazz_df = pd.read_csv('/mnt/user-data/uploads/jazz_timeseries_with_patterns.csv')

# =================================================================
# FOCUS: AU PRIVAVE ANALYSIS
# =================================================================

print("\n" + "="*80)
print("🎯 AU PRIVAVE - DEEP DIVE VALIDATION")
print("="*80)

# Au Privave parameters
au_privave = jazz_df[jazz_df['song'] == 'au_privave'].reset_index(drop=True)
duration_minutes = au_privave['time'].max() / 60

print(f"\nAu Privave:")
print(f"  Duration: {duration_minutes:.2f} minutes")
print(f"  Data points: {len(au_privave):,}")

# Create signature pattern
# Since we have 5-min data, we need to adjust window size
# Au Privave is 2.6 minutes, so we'll use a 1-bar window (5 minutes)
# But let's create a 2-bar pattern for better matching

window_size_bars = 2  # 2 bars = 10 minutes

jazz_energy = au_privave['rms_energy_norm'].values
jazz_centroid = au_privave['spectral_centroid_norm'].values

# Downsample to match 5-min resolution
# Au Privave is ~2.6 minutes, so we want to capture the energy progression
downsample_factor = len(jazz_energy) // window_size_bars
jazz_signature = jazz_energy[::downsample_factor][:window_size_bars]
jazz_signature_norm = normalize_pattern(jazz_signature)

print(f"\nSignature pattern (normalized): {jazz_signature_norm}")
print(f"Interpretation: {' → '.join([f'{v:.2f}' for v in jazz_signature_norm])}")

# =================================================================
# FIND ALL AU PRIVAVE PATTERNS IN 60 DAYS
# =================================================================

print(f"\n🔍 Scanning {len(spy)-window_size_bars:,} windows for Au Privave patterns...")

all_matches = []

for i in range(len(spy) - window_size_bars - 12):  # Need 12 bars (60 min) for prediction
    # Get volume window
    window_volume = spy['Volume'].iloc[i:i+window_size_bars].values
    window_volume_norm = normalize_pattern(window_volume)
    
    # Calculate DTW
    dtw = dtw_distance(jazz_signature_norm, window_volume_norm)
    
    # Calculate correlation
    if len(window_volume_norm) == len(jazz_signature_norm):
        corr, _ = pearsonr(jazz_signature_norm, window_volume_norm)
    else:
        corr = 0
    
    # Check if rising pattern
    is_rising = window_volume_norm[-1] > window_volume_norm[0]
    
    all_matches.append({
        'position': i,
        'start_time': spy['Datetime'].iloc[i],
        'dtw': dtw,
        'correlation': corr,
        'is_rising': is_rising
    })

matches_df = pd.DataFrame(all_matches)

# Get best matches
best_matches = matches_df.nsmallest(100, 'dtw')

print(f"\nTop 10 best matches (by DTW):")
print(best_matches[['start_time', 'dtw', 'correlation', 'is_rising']].head(10).to_string(index=False))

# =================================================================
# PREDICTION TESTING - MULTIPLE HORIZONS
# =================================================================

print("\n" + "="*80)
print("🔮 PREDICTIVE POWER ANALYSIS")
print("="*80)

# Test horizons in 5-min bars: 1 bar = 5 min
horizons_bars = [2, 4, 6, 8, 12]  # 10, 20, 30, 40, 60 minutes
horizons_minutes = [h*5 for h in horizons_bars]

all_predictions = []

for horizon_bars, horizon_min in zip(horizons_bars, horizons_minutes):
    predictions = []
    
    for idx, row in best_matches.head(50).iterrows():  # Top 50 matches
        pos = int(row['position'])
        
        if pos + window_size_bars + horizon_bars < len(spy):
            # Price at end of pattern
            price_start = spy['Close'].iloc[pos + window_size_bars - 1]
            
            # Price after horizon
            price_end = spy['Close'].iloc[pos + window_size_bars + horizon_bars - 1]
            
            # Calculate return
            price_return = ((price_end - price_start) / price_start) * 100
            
            # Volume change
            volume_during = spy['Volume'].iloc[pos:pos+window_size_bars].mean()
            volume_after = spy['Volume'].iloc[pos+window_size_bars:pos+window_size_bars+horizon_bars].mean()
            volume_change = ((volume_after - volume_during) / volume_during) * 100
            
            predictions.append({
                'horizon_min': horizon_min,
                'start_time': row['start_time'],
                'dtw': row['dtw'],
                'correlation': row['correlation'],
                'price_return': price_return,
                'volume_change': volume_change,
                'direction': 'UP' if price_return > 0 else 'DOWN'
            })
    
    if predictions:
        all_predictions.extend(predictions)

pred_df = pd.DataFrame(all_predictions)

# Summary by horizon
print(f"\n📊 Results by Prediction Horizon:")
print(f"\n{'Horizon':<12} {'Avg Return':<15} {'Win Rate':<12} {'Sharpe':<10} {'Sample'}")
print("-" * 70)

horizon_stats = []

for horizon_min in horizons_minutes:
    data = pred_df[pred_df['horizon_min'] == horizon_min]
    
    if len(data) > 0:
        avg_return = data['price_return'].mean()
        median_return = data['price_return'].median()
        std_return = data['price_return'].std()
        win_rate = (data['direction'] == 'UP').mean() * 100
        sharpe = avg_return / std_return if std_return > 0 else 0
        count = len(data)
        
        # Statistical test
        t_stat, p_value = ttest_1samp(data['price_return'], 0)
        
        # Binomial test for win rate
        binom_result = binomtest((data['direction'] == 'UP').sum(), count, 0.5, alternative='greater')
        binom_p = binom_result.pvalue
        
        horizon_stats.append({
            'horizon': horizon_min,
            'avg_return': avg_return,
            'median_return': median_return,
            'std': std_return,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'count': count,
            't_pvalue': p_value,
            'binom_pvalue': binom_p
        })
        
        sig_return = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
        sig_winrate = "***" if binom_p < 0.01 else "**" if binom_p < 0.05 else "*" if binom_p < 0.10 else ""
        
        print(f"{horizon_min} min{'':<6} {avg_return:+.4f}% {sig_return:<4} {win_rate:.1f}% {sig_winrate:<4} {sharpe:>6.3f}     {count}")

horizon_stats_df = pd.DataFrame(horizon_stats)

# Find best horizon
best_horizon = horizon_stats_df.loc[horizon_stats_df['avg_return'].idxmax()]

print(f"\n🏆 BEST HORIZON: {int(best_horizon['horizon'])} minutes")
print(f"   Average return: {best_horizon['avg_return']:+.4f}%")
print(f"   Win rate: {best_horizon['win_rate']:.1f}%")
print(f"   Sharpe ratio: {best_horizon['sharpe']:.3f}")
print(f"   Sample size: {int(best_horizon['count'])}")
print(f"   Return p-value: {best_horizon['t_pvalue']:.4f}")
print(f"   Win rate p-value: {best_horizon['binom_pvalue']:.4f}")

# =================================================================
# TIME SERIES ANALYSIS - WHEN DO PATTERNS OCCUR?
# =================================================================

print("\n" + "="*80)
print("📅 TEMPORAL ANALYSIS")
print("="*80)

# Add date/time features to best matches
best_matches['hour'] = pd.to_datetime(best_matches['start_time']).dt.hour
best_matches['day_of_week'] = pd.to_datetime(best_matches['start_time']).dt.day_name()
best_matches['date'] = pd.to_datetime(best_matches['start_time']).dt.date

# By hour
print(f"\n🕐 Patterns by Hour of Day (Top 50):")
hour_dist = best_matches.head(50)['hour'].value_counts().sort_index()
for hour, count in hour_dist.items():
    print(f"   {hour:02d}:00 - {count} patterns")

# By day of week
print(f"\n📆 Patterns by Day of Week (Top 50):")
dow_dist = best_matches.head(50)['day_of_week'].value_counts()
for day, count in dow_dist.items():
    print(f"   {day}: {count} patterns")

# =================================================================
# VISUALIZATIONS
# =================================================================

print("\n" + "="*80)
print("📊 CREATING FINAL VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. DTW distribution of all matches
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(matches_df['dtw'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(x=matches_df['dtw'].quantile(0.1), color='red', linestyle='--', 
           linewidth=2, label='Top 10%')
ax1.set_xlabel('DTW Distance', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('Au Privave Pattern Match Distribution\n(60 Days)', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Correlation distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(matches_df['correlation'], bins=50, alpha=0.7, color='coral', edgecolor='black')
ax2.set_xlabel('Correlation', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Pattern Correlation Distribution', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Rising vs Falling patterns
ax3 = fig.add_subplot(gs[0, 2])
rising_count = best_matches.head(50)['is_rising'].sum()
labels = ['Rising', 'Falling']
counts = [rising_count, 50-rising_count]
colors = ['green', 'red']
ax3.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax3.set_title('Pattern Direction (Top 50)', fontsize=11, fontweight='bold')

# 4. Return by horizon
ax4 = fig.add_subplot(gs[1, :])
horizons = horizon_stats_df['horizon'].values
returns = horizon_stats_df['avg_return'].values
errors = horizon_stats_df['std'].values / np.sqrt(horizon_stats_df['count'].values)

ax4.errorbar(horizons, returns, yerr=errors, fmt='o-', linewidth=3, 
            markersize=10, capsize=5, color='steelblue')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_xlabel('Prediction Horizon (minutes)', fontsize=11)
ax4.set_ylabel('Average Return (%)', fontsize=11)
ax4.set_title('Au Privave Predictive Power - 60 Day Validation', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Mark best
best_idx = horizon_stats_df['avg_return'].idxmax()
ax4.scatter([horizon_stats_df.loc[best_idx, 'horizon']], 
           [horizon_stats_df.loc[best_idx, 'avg_return']], 
           s=300, color='gold', edgecolors='black', linewidths=2, zorder=5)

# 5. Win rate by horizon
ax5 = fig.add_subplot(gs[2, 0])
win_rates = horizon_stats_df['win_rate'].values

ax5.plot(horizons, win_rates, 'o-', linewidth=3, markersize=10, color='green')
ax5.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random (50%)')
ax5.set_xlabel('Prediction Horizon (minutes)', fontsize=10)
ax5.set_ylabel('Win Rate (%)', fontsize=10)
ax5.set_title('Win Rate by Horizon', fontsize=11, fontweight='bold')
ax5.set_ylim(0, 100)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Sharpe ratio
ax6 = fig.add_subplot(gs[2, 1])
sharpes = horizon_stats_df['sharpe'].values
colors_sharpe = ['green' if s > 0 else 'red' for s in sharpes]

ax6.bar(horizons, sharpes, color=colors_sharpe, alpha=0.7, edgecolor='black')
ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax6.set_xlabel('Prediction Horizon (minutes)', fontsize=10)
ax6.set_ylabel('Sharpe Ratio', fontsize=10)
ax6.set_title('Risk-Adjusted Returns', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Sample size by horizon
ax7 = fig.add_subplot(gs[2, 2])
samples = horizon_stats_df['count'].values

ax7.bar(horizons, samples, color='steelblue', alpha=0.7, edgecolor='black')
ax7.set_xlabel('Prediction Horizon (minutes)', fontsize=10)
ax7.set_ylabel('Sample Size', fontsize=10)
ax7.set_title('Number of Patterns Tested', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 8. Return distribution at best horizon
ax8 = fig.add_subplot(gs[3, :])
best_horizon_data = pred_df[pred_df['horizon_min'] == best_horizon['horizon']]['price_return']

ax8.hist(best_horizon_data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax8.axvline(x=best_horizon_data.mean(), color='green', linestyle='--', linewidth=2,
           label=f"Mean: {best_horizon_data.mean():+.3f}%")
ax8.axvline(x=best_horizon_data.median(), color='orange', linestyle='--', linewidth=2,
           label=f"Median: {best_horizon_data.median():+.3f}%")
ax8.set_xlabel('Return (%)', fontsize=11)
ax8.set_ylabel('Frequency', fontsize=11)
ax8.set_title(f'Return Distribution at {int(best_horizon["horizon"])}-Minute Horizon', 
             fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.suptitle('Au Privave - 60 Day Validation Study', fontsize=16, fontweight='bold')
plt.savefig('/mnt/user-data/outputs/au_privave_60day_validation.png', dpi=150, bbox_inches='tight')
print("✅ Saved: au_privave_60day_validation.png")

# Save results
pred_df.to_csv('/home/claude/au_privave_60day_predictions.csv', index=False)
horizon_stats_df.to_csv('/home/claude/au_privave_horizon_stats.csv', index=False)
print("✅ Saved: au_privave_60day_predictions.csv")
print("✅ Saved: au_privave_horizon_stats.csv")

# =================================================================
# FINAL SUMMARY STATISTICS
# =================================================================

print("\n" + "="*80)
print("📋 FINAL VALIDATION SUMMARY")
print("="*80)

print(f"\n📊 Dataset:")
print(f"   Period: {spy['Datetime'].min().date()} to {spy['Datetime'].max().date()}")
print(f"   Trading days: {spy['Datetime'].dt.date.nunique()}")
print(f"   Total 5-min bars: {len(spy):,}")

print(f"\n🎺 Au Privave Pattern:")
print(f"   Signature: {jazz_signature_norm}")
print(f"   Total matches found: {len(matches_df):,}")
print(f"   Top matches analyzed: 50")

print(f"\n🏆 BEST PERFORMANCE:")
print(f"   Optimal horizon: {int(best_horizon['horizon'])} minutes")
print(f"   Average return: {best_horizon['avg_return']:+.4f}%")
print(f"   Win rate: {best_horizon['win_rate']:.1f}%")
print(f"   Sharpe ratio: {best_horizon['sharpe']:.3f}")
print(f"   Sample size: {int(best_horizon['count'])}")

if best_horizon['t_pvalue'] < 0.05:
    print(f"   ✅ Returns are statistically significant (p={best_horizon['t_pvalue']:.4f})")
elif best_horizon['t_pvalue'] < 0.10:
    print(f"   ⚠️ Returns are marginally significant (p={best_horizon['t_pvalue']:.4f})")
else:
    print(f"   ❌ Returns not statistically significant (p={best_horizon['t_pvalue']:.4f})")

if best_horizon['binom_pvalue'] < 0.05:
    print(f"   ✅ Win rate is statistically significant (p={best_horizon['binom_pvalue']:.4f})")
elif best_horizon['binom_pvalue'] < 0.10:
    print(f"   ⚠️ Win rate is marginally significant (p={best_horizon['binom_pvalue']:.4f})")
else:
    print(f"   ❌ Win rate not statistically significant (p={best_horizon['binom_pvalue']:.4f})")

# Calculate annualized performance
if best_horizon['avg_return'] > 0:
    # Assume we find 1 pattern per week (conservative)
    patterns_per_year = 52
    annual_return = best_horizon['avg_return'] * patterns_per_year
    print(f"\n💰 Hypothetical Annual Performance:")
    print(f"   If 1 pattern/week: {annual_return:+.2f}%")
    print(f"   Note: This assumes no transaction costs")

print("\n" + "="*80)
print("✨ 60-DAY VALIDATION COMPLETE")
print("="*80)
