"""
Extended Time Scale Prediction Test
Test if Blue Monk patterns predict longer-term movements (10-60 minutes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp
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
print("🔮 EXTENDED TIME SCALE PREDICTION TEST")
print("="*80)

# Load SPY data
spy_raw = pd.read_csv('/mnt/user-data/uploads/SPY_intraday_1m_7d.csv', skiprows=2)
spy_raw.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
spy_raw['Datetime'] = pd.to_datetime(spy_raw['Datetime'])
spy_raw = spy_raw.sort_values('Datetime').reset_index(drop=True)

for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
    spy_raw[col] = pd.to_numeric(spy_raw[col], errors='coerce')

spy_raw = spy_raw.dropna()

# Filter to market hours
spy_raw['hour'] = spy_raw['Datetime'].dt.hour
spy_raw['minute'] = spy_raw['Datetime'].dt.minute
spy_raw = spy_raw[
    ((spy_raw['hour'] == 9) & (spy_raw['minute'] >= 30)) |
    ((spy_raw['hour'] >= 10) & (spy_raw['hour'] < 16)) |
    ((spy_raw['hour'] == 16) & (spy_raw['minute'] == 0))
].copy()

spy = spy_raw.reset_index(drop=True)

print(f"\nSPY data: {len(spy):,} minutes")

# Load jazz data
jazz_df = pd.read_csv('/mnt/user-data/uploads/jazz_timeseries_with_patterns.csv')

# Top 5 songs
top_songs = [
    ('blue_monk', 3),
    ('au_privave', 2),
    ('bags_groove', 5),
    ('bilies_bounce', 3),
    ('nows_the_time', 3)
]

# =================================================================
# TEST MULTIPLE TIME HORIZONS
# =================================================================

print("\n" + "="*80)
print("📊 TESTING MULTIPLE PREDICTION HORIZONS")
print("="*80)

# Test horizons: 5, 10, 15, 20, 30, 45, 60 minutes
horizons = [5, 10, 15, 20, 30, 45, 60]

all_horizon_results = []

for song_name, window_size in top_songs:
    print(f"\n{'='*80}")
    print(f"🎵 {song_name.upper()}")
    print(f"{'='*80}")
    
    song_data = jazz_df[jazz_df['song'] == song_name].reset_index(drop=True)
    
    # Get signature pattern
    jazz_energy = song_data['rms_energy_norm'].values
    downsample_factor = max(1, len(jazz_energy) // window_size)
    jazz_signature = jazz_energy[::downsample_factor][:window_size]
    jazz_signature_norm = normalize_pattern(jazz_signature)
    
    # Find best matches in SPY
    best_matches = []
    
    for i in range(len(spy) - window_size - 60):  # Need 60 min buffer
        window_volume = spy['Volume'].iloc[i:i+window_size].values
        window_volume_norm = normalize_pattern(window_volume)
        
        dtw = dtw_distance(jazz_signature_norm, window_volume_norm)
        
        best_matches.append({
            'position': i,
            'dtw': dtw
        })
    
    # Get top 20 matches
    best_matches_df = pd.DataFrame(best_matches).nsmallest(20, 'dtw')
    
    print(f"Analyzing top 20 matches across {len(horizons)} time horizons...")
    
    # Test each horizon
    for horizon in horizons:
        predictions = []
        
        for _, row in best_matches_df.iterrows():
            pos = int(row['position'])
            
            # Make sure we have enough data
            if pos + window_size + horizon < len(spy):
                # Price at end of pattern
                price_start = spy['Close'].iloc[pos + window_size - 1]
                
                # Price after horizon
                price_end = spy['Close'].iloc[pos + window_size + horizon - 1]
                
                # Calculate returns
                price_change_pct = ((price_end - price_start) / price_start) * 100
                
                # Volume behavior
                volume_during = spy['Volume'].iloc[pos:pos+window_size].mean()
                volume_after = spy['Volume'].iloc[pos+window_size:pos+window_size+horizon].mean()
                volume_change_pct = ((volume_after - volume_during) / volume_during) * 100
                
                # High/Low range
                high_after = spy['High'].iloc[pos+window_size:pos+window_size+horizon].max()
                low_after = spy['Low'].iloc[pos+window_size:pos+window_size+horizon].min()
                volatility = ((high_after - low_after) / price_start) * 100
                
                # Max favorable/unfavorable excursion
                prices_after = spy['Close'].iloc[pos+window_size:pos+window_size+horizon]
                max_gain = ((prices_after.max() - price_start) / price_start) * 100
                max_loss = ((prices_after.min() - price_start) / price_start) * 100
                
                predictions.append({
                    'song': song_name,
                    'horizon': horizon,
                    'dtw': row['dtw'],
                    'price_change_pct': price_change_pct,
                    'volume_change_pct': volume_change_pct,
                    'volatility': volatility,
                    'max_gain': max_gain,
                    'max_loss': max_loss,
                    'direction': 'UP' if price_change_pct > 0 else 'DOWN'
                })
        
        if predictions:
            pred_df = pd.DataFrame(predictions)
            
            all_horizon_results.extend(predictions)
            
            # Summary for this song and horizon
            avg_return = pred_df['price_change_pct'].mean()
            win_rate = (pred_df['direction'] == 'UP').mean() * 100
            sharpe = pred_df['price_change_pct'].mean() / pred_df['price_change_pct'].std() if pred_df['price_change_pct'].std() > 0 else 0
            
            print(f"  {horizon:2d} min: Avg={avg_return:+.3f}%, Win={win_rate:.0f}%, Sharpe={sharpe:.2f}, n={len(pred_df)}")

# =================================================================
# ANALYZE RESULTS ACROSS ALL HORIZONS
# =================================================================

print("\n" + "="*80)
print("📊 CROSS-HORIZON ANALYSIS")
print("="*80)

results_df = pd.DataFrame(all_horizon_results)

# Group by horizon
horizon_summary = results_df.groupby('horizon').agg({
    'price_change_pct': ['mean', 'median', 'std', 'count'],
    'direction': lambda x: (x=='UP').sum(),
    'volatility': 'mean',
    'max_gain': 'mean',
    'max_loss': 'mean'
})

print("\n🎯 Performance by Time Horizon:")
print("\n" + "="*80)

for horizon in horizons:
    data = results_df[results_df['horizon'] == horizon]
    
    if len(data) > 0:
        avg_return = data['price_change_pct'].mean()
        median_return = data['price_change_pct'].median()
        std_return = data['price_change_pct'].std()
        count = len(data)
        up_count = (data['direction'] == 'UP').sum()
        win_rate = (up_count / count) * 100
        avg_vol = data['volatility'].mean()
        sharpe = avg_return / std_return if std_return > 0 else 0
        
        # T-test: Is mean significantly different from 0?
        t_stat, p_value = ttest_1samp(data['price_change_pct'], 0)
        
        print(f"\n{horizon}-Minute Horizon:")
        print(f"  Sample size: {count}")
        print(f"  Average return: {avg_return:+.4f}%")
        print(f"  Median return: {median_return:+.4f}%")
        print(f"  Std dev: {std_return:.4f}%")
        print(f"  Win rate: {win_rate:.1f}% ({up_count}/{count})")
        print(f"  Sharpe ratio: {sharpe:.3f}")
        print(f"  Avg volatility: {avg_vol:.3f}%")
        print(f"  T-test p-value: {p_value:.4f}", end="")
        
        if p_value < 0.05:
            print(" ✅ SIGNIFICANT!")
        elif p_value < 0.10:
            print(" ⚠️ Marginally significant")
        else:
            print("")

# Find best horizon
best_horizon_by_return = results_df.groupby('horizon')['price_change_pct'].mean().idxmax()
best_return = results_df.groupby('horizon')['price_change_pct'].mean().max()

best_horizon_by_winrate = (results_df.groupby('horizon').apply(lambda x: (x['direction']=='UP').mean()) * 100).idxmax()
best_winrate = (results_df.groupby('horizon').apply(lambda x: (x['direction']=='UP').mean()) * 100).max()

print(f"\n🏆 BEST HORIZONS:")
print(f"  Best by return: {best_horizon_by_return} min ({best_return:+.3f}%)")
print(f"  Best by win rate: {best_horizon_by_winrate} min ({best_winrate:.1f}%)")

# =================================================================
# SONG COMPARISON AT BEST HORIZON
# =================================================================

print("\n" + "="*80)
print(f"📊 SONG COMPARISON AT {best_horizon_by_return}-MIN HORIZON")
print("="*80)

best_horizon_data = results_df[results_df['horizon'] == best_horizon_by_return]

song_performance = best_horizon_data.groupby('song').agg({
    'price_change_pct': ['mean', 'std', 'count'],
    'direction': lambda x: (x=='UP').sum()
})

print(f"\nSong Performance ({best_horizon_by_return} minutes ahead):")
print(song_performance)

# =================================================================
# VISUALIZATIONS
# =================================================================

print("\n" + "="*80)
print("📊 CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# 1. Average return by horizon
ax1 = fig.add_subplot(gs[0, 0])
horizon_means = results_df.groupby('horizon')['price_change_pct'].mean()
horizon_stds = results_df.groupby('horizon')['price_change_pct'].std()

ax1.errorbar(horizon_means.index, horizon_means.values, yerr=horizon_stds.values,
            fmt='o-', linewidth=2, markersize=8, capsize=5, color='steelblue')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax1.set_xlabel('Prediction Horizon (minutes)', fontsize=11)
ax1.set_ylabel('Average Return (%)', fontsize=11)
ax1.set_title('Return by Prediction Horizon', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Mark best horizon
ax1.scatter([best_horizon_by_return], [best_return], s=200, color='gold', 
           edgecolors='black', linewidths=2, zorder=5, label=f'Best: {best_horizon_by_return}min')
ax1.legend()

# 2. Win rate by horizon
ax2 = fig.add_subplot(gs[0, 1])
win_rates = (results_df.groupby('horizon').apply(lambda x: (x['direction']=='UP').mean()) * 100)

ax2.plot(win_rates.index, win_rates.values, 'o-', linewidth=2, markersize=8, color='coral')
ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random (50%)')
ax2.set_xlabel('Prediction Horizon (minutes)', fontsize=11)
ax2.set_ylabel('Win Rate (%)', fontsize=11)
ax2.set_title('Win Rate by Prediction Horizon', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Sharpe ratio by horizon
ax3 = fig.add_subplot(gs[1, 0])
sharpe_ratios = results_df.groupby('horizon').apply(
    lambda x: x['price_change_pct'].mean() / x['price_change_pct'].std() if x['price_change_pct'].std() > 0 else 0
)

colors = ['green' if s > 0 else 'red' for s in sharpe_ratios.values]
ax3.bar(sharpe_ratios.index, sharpe_ratios.values, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Prediction Horizon (minutes)', fontsize=11)
ax3.set_ylabel('Sharpe Ratio', fontsize=11)
ax3.set_title('Risk-Adjusted Returns by Horizon', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Volatility by horizon
ax4 = fig.add_subplot(gs[1, 1])
volatilities = results_df.groupby('horizon')['volatility'].mean()

ax4.plot(volatilities.index, volatilities.values, 'o-', linewidth=2, markersize=8, color='purple')
ax4.set_xlabel('Prediction Horizon (minutes)', fontsize=11)
ax4.set_ylabel('Average Volatility (%)', fontsize=11)
ax4.set_title('Price Volatility by Prediction Horizon', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Distribution of returns at best horizon
ax5 = fig.add_subplot(gs[2, :])
best_data = results_df[results_df['horizon'] == best_horizon_by_return]['price_change_pct']

ax5.hist(best_data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax5.axvline(x=best_data.mean(), color='green', linestyle='--', linewidth=2, 
           label=f'Mean: {best_data.mean():+.3f}%')
ax5.axvline(x=best_data.median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {best_data.median():+.3f}%')
ax5.set_xlabel('Return (%)', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title(f'Return Distribution at {best_horizon_by_return}-Minute Horizon', 
             fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.suptitle('Extended Time Scale Prediction Analysis', fontsize=15, fontweight='bold')
plt.savefig('/mnt/user-data/outputs/extended_horizon_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: extended_horizon_analysis.png")

# Save results
results_df.to_csv('/home/claude/extended_horizon_results.csv', index=False)
print("✅ Saved: extended_horizon_results.csv")

# =================================================================
# FINAL CONCLUSIONS
# =================================================================

print("\n" + "="*80)
print("✨ EXTENDED HORIZON CONCLUSIONS")
print("="*80)

print(f"\n🔍 KEY FINDINGS:")

# Find if any horizon is significantly better
significant_horizons = []
for horizon in horizons:
    data = results_df[results_df['horizon'] == horizon]['price_change_pct']
    if len(data) > 0:
        t_stat, p_value = ttest_1samp(data, 0)
        if p_value < 0.10:
            significant_horizons.append((horizon, data.mean(), p_value))

if significant_horizons:
    print(f"\n✅ Statistically significant horizons found:")
    for h, mean, p in significant_horizons:
        print(f"   {h} min: {mean:+.3f}% (p={p:.4f})")
else:
    print(f"\n⚠️ No statistically significant predictive power found")

# Compare to 5-min baseline
baseline_5min = results_df[results_df['horizon'] == 5]['price_change_pct'].mean()
best_horizon_return = results_df[results_df['horizon'] == best_horizon_by_return]['price_change_pct'].mean()

print(f"\n📊 Comparison to 5-min baseline:")
print(f"   5-min return: {baseline_5min:+.4f}%")
print(f"   {best_horizon_by_return}-min return: {best_horizon_return:+.4f}%")
print(f"   Improvement: {best_horizon_return - baseline_5min:+.4f}%")

if abs(best_horizon_return) > abs(baseline_5min) * 1.5:
    print(f"   ✅ Meaningful improvement at longer horizon!")
elif abs(best_horizon_return) > abs(baseline_5min):
    print(f"   ⚠️ Slight improvement")
else:
    print(f"   ❌ No improvement")

print(f"\n💡 RECOMMENDATION:")
if len(significant_horizons) > 0 and best_return > 0.02:
    print(f"   ✅ {best_horizon_by_return}-minute horizon shows promise!")
    print(f"   Consider testing with more data (months instead of days)")
elif best_winrate > 55:
    print(f"   ⚠️ Win rate of {best_winrate:.1f}% is encouraging")
    print(f"   But returns are small - may not be tradeable")
else:
    print(f"   ❌ No strong predictive edge found")
    print(f"   Pattern matching is descriptive, not predictive")

print("\n" + "="*80)
print("✅ EXTENDED ANALYSIS COMPLETE")
print("="*80)
