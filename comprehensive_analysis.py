"""
Comprehensive Pattern Analysis:
1. Find other Blue Monk-like patterns in SPY data
2. Reverse test: When SPY shows this pattern, what happens next?
3. Extend to other top-performing songs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr
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
print("🔍 COMPREHENSIVE PATTERN ANALYSIS")
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

print(f"\nSPY data: {len(spy):,} minutes across {spy['Datetime'].dt.date.nunique()} days")

# =================================================================
# TASK 1: FIND OTHER BLUE MONK PATTERNS IN SPY DATA
# =================================================================

print("\n" + "="*80)
print("TASK 1: FIND BLUE MONK-LIKE PATTERNS IN SPY")
print("="*80)

# The Blue Monk signature pattern (normalized volume progression)
blue_monk_signature = np.array([0.0000, 0.1021, 0.1312])
blue_monk_signature_norm = normalize_pattern(blue_monk_signature)

print(f"\nBlue Monk signature pattern: {blue_monk_signature}")
print(f"Characteristics: Low → Medium → High (gradual rise)")

# Scan all 3-minute windows in SPY
window_size = 3
all_matches = []

print(f"\nScanning {len(spy)-window_size} windows...")

for i in range(len(spy) - window_size):
    window_volume = spy['Volume'].iloc[i:i+window_size].values
    window_volume_norm = normalize_pattern(window_volume)
    
    # Calculate DTW distance to Blue Monk pattern
    dtw = dtw_distance(blue_monk_signature_norm, window_volume_norm)
    
    # Calculate correlation
    if len(window_volume_norm) == len(blue_monk_signature_norm):
        corr, _ = pearsonr(blue_monk_signature_norm, window_volume_norm)
    else:
        corr = 0
    
    # Check if it's a "gradual rise" pattern
    is_rising = (window_volume_norm[1] > window_volume_norm[0]) and \
                (window_volume_norm[2] > window_volume_norm[1])
    
    all_matches.append({
        'position': i,
        'start_time': spy['Datetime'].iloc[i],
        'dtw': dtw,
        'correlation': corr,
        'is_rising': is_rising,
        'volume_pattern': window_volume_norm.tolist(),
        'actual_volumes': window_volume.tolist()
    })

matches_df = pd.DataFrame(all_matches)

# Find best matches (low DTW, high correlation, rising pattern)
blue_monk_like = matches_df[
    (matches_df['dtw'] < 0.15) & 
    (matches_df['correlation'] > 0.8) &
    (matches_df['is_rising'] == True)
].sort_values('dtw')

print(f"\n🎯 Found {len(blue_monk_like)} Blue Monk-like patterns (DTW<0.15, r>0.8, rising)")
print(f"\nTop 10 matches:")
print(blue_monk_like[['start_time', 'dtw', 'correlation']].head(10).to_string(index=False))

# =================================================================
# TASK 2: REVERSE TEST - WHAT HAPPENS AFTER THESE PATTERNS?
# =================================================================

print("\n" + "="*80)
print("TASK 2: REVERSE TEST - PREDICTION VALIDATION")
print("="*80)

print(f"\nQuestion: When we see a Blue Monk pattern, what happens in the next 5 minutes?")

# For each Blue Monk-like pattern, check what happened next
predictions = []

for idx, row in blue_monk_like.head(20).iterrows():  # Top 20 patterns
    pos = row['position']
    
    # Get the 5 minutes after the pattern
    if pos + window_size + 5 < len(spy):
        after_window = spy.iloc[pos+window_size:pos+window_size+5]
        
        # Calculate what happened
        price_before = spy['Close'].iloc[pos+window_size-1]
        price_after = after_window['Close'].iloc[-1]
        price_change_pct = ((price_after - price_before) / price_before) * 100
        
        volume_before = spy['Volume'].iloc[pos+window_size-1]
        volume_after = after_window['Volume'].mean()
        volume_change_pct = ((volume_after - volume_before) / volume_before) * 100
        
        # High/Low range
        price_high = after_window['High'].max()
        price_low = after_window['Low'].min()
        price_range_pct = ((price_high - price_low) / price_before) * 100
        
        predictions.append({
            'start_time': row['start_time'],
            'dtw': row['dtw'],
            'correlation': row['correlation'],
            'price_change_pct': price_change_pct,
            'volume_change_pct': volume_change_pct,
            'price_range_pct': price_range_pct,
            'direction': 'UP' if price_change_pct > 0 else 'DOWN'
        })

predictions_df = pd.DataFrame(predictions)

if len(predictions_df) > 0:
    print(f"\n📊 Analysis of {len(predictions_df)} patterns:")
    print(f"\nPrice movement (next 5 min):")
    print(f"   Average: {predictions_df['price_change_pct'].mean():+.3f}%")
    print(f"   Median: {predictions_df['price_change_pct'].median():+.3f}%")
    print(f"   Direction: {(predictions_df['direction']=='UP').sum()}/{len(predictions_df)} went UP")
    
    print(f"\nVolume change (next 5 min):")
    print(f"   Average: {predictions_df['volume_change_pct'].mean():+.1f}%")
    print(f"   Median: {predictions_df['volume_change_pct'].median():+.1f}%")
    
    print(f"\nVolatility (next 5 min):")
    print(f"   Average range: {predictions_df['price_range_pct'].mean():.3f}%")
    
    # Save predictions
    predictions_df.to_csv('/home/claude/blue_monk_predictions.csv', index=False)
    print(f"\n✅ Saved: blue_monk_predictions.csv")

# =================================================================
# TASK 3: EXTEND TO OTHER TOP SONGS
# =================================================================

print("\n" + "="*80)
print("TASK 3: ANALYZE OTHER TOP-PERFORMING SONGS")
print("="*80)

# Load jazz data
jazz_df = pd.read_csv('/mnt/user-data/uploads/jazz_timeseries_with_patterns.csv')

# Top 5 songs from our analysis
top_songs = [
    ('blue_monk', 3, 0.045),
    ('au_privave', 2, 0.052),
    ('bags_groove', 5, 0.184),
    ('bilies_bounce', 3, 0.207),
    ('nows_the_time', 3, 0.291)
]

all_song_patterns = []

for song_name, window_size, known_dtw in top_songs:
    print(f"\n{'='*80}")
    print(f"🎵 {song_name.upper()}")
    print(f"{'='*80}")
    
    song_data = jazz_df[jazz_df['song'] == song_name].reset_index(drop=True)
    
    # Get signature pattern
    jazz_energy = song_data['rms_energy_norm'].values
    downsample_factor = max(1, len(jazz_energy) // window_size)
    jazz_signature = jazz_energy[::downsample_factor][:window_size]
    jazz_signature_norm = normalize_pattern(jazz_signature)
    
    print(f"Signature pattern: {jazz_signature_norm}")
    print(f"Window size: {window_size} minutes")
    
    # Find matches in SPY
    song_matches = []
    
    for i in range(len(spy) - window_size):
        window_volume = spy['Volume'].iloc[i:i+window_size].values
        window_volume_norm = normalize_pattern(window_volume)
        
        dtw = dtw_distance(jazz_signature_norm, window_volume_norm)
        
        if len(window_volume_norm) == len(jazz_signature_norm):
            corr, _ = pearsonr(jazz_signature_norm, window_volume_norm)
        else:
            corr = 0
        
        song_matches.append({
            'song': song_name,
            'position': i,
            'start_time': spy['Datetime'].iloc[i],
            'dtw': dtw,
            'correlation': corr,
            'window_size': window_size
        })
    
    song_matches_df = pd.DataFrame(song_matches)
    
    # Find best matches
    best_matches = song_matches_df.nsmallest(10, 'dtw')
    
    print(f"\n🎯 Top 5 matches in SPY:")
    print(best_matches[['start_time', 'dtw', 'correlation']].head(5).to_string(index=False))
    
    # Check predictions for top matches
    song_predictions = []
    
    for _, row in best_matches.head(10).iterrows():
        pos = row['position']
        
        if pos + window_size + 5 < len(spy):
            after_window = spy.iloc[pos+window_size:pos+window_size+5]
            
            price_before = spy['Close'].iloc[pos+window_size-1]
            price_after = after_window['Close'].iloc[-1]
            price_change_pct = ((price_after - price_before) / price_before) * 100
            
            song_predictions.append({
                'song': song_name,
                'start_time': row['start_time'],
                'dtw': row['dtw'],
                'price_change_pct': price_change_pct,
                'direction': 'UP' if price_change_pct > 0 else 'DOWN'
            })
    
    if song_predictions:
        pred_df = pd.DataFrame(song_predictions)
        print(f"\n📈 Prediction results (next 5 min):")
        print(f"   Avg price change: {pred_df['price_change_pct'].mean():+.3f}%")
        print(f"   Direction: {(pred_df['direction']=='UP').sum()}/{len(pred_df)} went UP")
        
        all_song_patterns.extend(song_predictions)

# =================================================================
# SUMMARY COMPARISON
# =================================================================

print("\n" + "="*80)
print("📊 OVERALL SUMMARY - ALL 5 SONGS")
print("="*80)

if all_song_patterns:
    all_patterns_df = pd.DataFrame(all_song_patterns)
    
    print(f"\nTotal patterns analyzed: {len(all_patterns_df)}")
    
    # Group by song
    by_song = all_patterns_df.groupby('song').agg({
        'price_change_pct': ['mean', 'median', 'std'],
        'direction': lambda x: (x=='UP').sum()
    })
    
    print(f"\n🎺 Performance by Song:")
    print(by_song)
    
    # Overall
    print(f"\n🎯 OVERALL PREDICTIVE POWER:")
    print(f"   Average price change: {all_patterns_df['price_change_pct'].mean():+.3f}%")
    print(f"   Median price change: {all_patterns_df['price_change_pct'].median():+.3f}%")
    print(f"   Up direction: {(all_patterns_df['direction']=='UP').sum()}/{len(all_patterns_df)} ({(all_patterns_df['direction']=='UP').mean()*100:.1f}%)")
    
    # Save
    all_patterns_df.to_csv('/home/claude/all_songs_predictions.csv', index=False)
    print(f"\n✅ Saved: all_songs_predictions.csv")

# =================================================================
# VISUALIZATIONS
# =================================================================

print("\n" + "="*80)
print("📊 CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

# 1. Blue Monk pattern occurrences
ax1 = fig.add_subplot(gs[0, :])
if len(blue_monk_like) > 0:
    blue_monk_times = blue_monk_like['start_time'].head(20)
    blue_monk_dtws = blue_monk_like['dtw'].head(20)
    
    ax1.scatter(range(len(blue_monk_times)), blue_monk_dtws, s=100, c=blue_monk_dtws, 
               cmap='RdYlGn_r', alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Pattern Instance', fontsize=11)
    ax1.set_ylabel('DTW Distance', fontsize=11)
    ax1.set_title('Blue Monk-like Patterns Found in SPY (Top 20)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(ax1.collections[0], ax=ax1, label='DTW Distance')

# 2. Prediction results - Price change distribution
ax2 = fig.add_subplot(gs[1, 0])
if len(predictions_df) > 0:
    ax2.hist(predictions_df['price_change_pct'], bins=15, alpha=0.7, 
            color='steelblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax2.axvline(x=predictions_df['price_change_pct'].mean(), color='green', 
               linestyle='--', linewidth=2, label=f'Mean: {predictions_df["price_change_pct"].mean():+.3f}%')
    ax2.set_xlabel('Price Change (%)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Blue Monk Pattern → Price Movement (Next 5 min)', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# 3. Volume change distribution
ax3 = fig.add_subplot(gs[1, 1])
if len(predictions_df) > 0:
    ax3.hist(predictions_df['volume_change_pct'], bins=15, alpha=0.7, 
            color='coral', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=predictions_df['volume_change_pct'].mean(), color='green', 
               linestyle='--', linewidth=2, label=f'Mean: {predictions_df["volume_change_pct"].mean():+.1f}%')
    ax3.set_xlabel('Volume Change (%)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Blue Monk Pattern → Volume Change (Next 5 min)', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# 4. All songs comparison
ax4 = fig.add_subplot(gs[2, :])
if all_song_patterns:
    song_names = []
    song_means = []
    song_stds = []
    
    for song_name, _, _ in top_songs:
        song_preds = [p['price_change_pct'] for p in all_song_patterns if p['song'] == song_name]
        if song_preds:
            song_names.append(song_name.replace('_', ' ').title())
            song_means.append(np.mean(song_preds))
            song_stds.append(np.std(song_preds))
    
    x_pos = np.arange(len(song_names))
    colors = ['green' if m > 0 else 'red' for m in song_means]
    
    ax4.bar(x_pos, song_means, yerr=song_stds, alpha=0.7, color=colors, 
           edgecolor='black', capsize=5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(song_names, rotation=15)
    ax4.set_ylabel('Avg Price Change (%)', fontsize=11)
    ax4.set_title('Predictive Power by Song (Next 5 min)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

# 5. Win rate comparison
ax5 = fig.add_subplot(gs[3, :])
if all_song_patterns:
    song_names = []
    win_rates = []
    
    for song_name, _, _ in top_songs:
        song_preds = [p for p in all_song_patterns if p['song'] == song_name]
        if song_preds:
            song_names.append(song_name.replace('_', ' ').title())
            up_count = sum(1 for p in song_preds if p['direction'] == 'UP')
            win_rates.append(up_count / len(song_preds) * 100)
    
    x_pos = np.arange(len(song_names))
    colors = ['green' if w > 50 else 'red' for w in win_rates]
    
    ax5.bar(x_pos, win_rates, alpha=0.7, color=colors, edgecolor='black')
    ax5.axhline(y=50, color='black', linestyle='--', linewidth=2, label='50% (Random)')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(song_names, rotation=15)
    ax5.set_ylabel('Win Rate (%)', fontsize=11)
    ax5.set_title('Direction Prediction Accuracy (% Up Moves)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 100)

plt.suptitle('Comprehensive Pattern Analysis - Predictive Testing', fontsize=15, fontweight='bold')
plt.savefig('/mnt/user-data/outputs/comprehensive_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: comprehensive_analysis.png")

# =================================================================
# FINAL CONCLUSIONS
# =================================================================

print("\n" + "="*80)
print("✨ FINAL CONCLUSIONS")
print("="*80)

print(f"\n🎯 TASK 1 - Pattern Discovery:")
print(f"   Found {len(blue_monk_like)} Blue Monk-like patterns in SPY data")
print(f"   These are 'gradual volume rise' patterns similar to Blue Monk's energy curve")

print(f"\n🔮 TASK 2 - Predictive Power (Blue Monk):")
if len(predictions_df) > 0:
    avg_change = predictions_df['price_change_pct'].mean()
    up_pct = (predictions_df['direction']=='UP').mean() * 100
    
    print(f"   Average price change: {avg_change:+.3f}%")
    print(f"   Win rate (up moves): {up_pct:.1f}%")
    
    if abs(avg_change) > 0.01:
        print(f"   ✅ Shows directional bias!")
    else:
        print(f"   ⚠️ No strong directional bias")
    
    if up_pct > 60:
        print(f"   ✅ Strong upward bias!")
    elif up_pct > 55:
        print(f"   ⚠️ Slight upward bias")
    else:
        print(f"   ⚠️ No clear directional edge")

print(f"\n🎺 TASK 3 - Multi-Song Analysis:")
if all_song_patterns:
    overall_avg = all_patterns_df['price_change_pct'].mean()
    overall_win = (all_patterns_df['direction']=='UP').mean() * 100
    
    print(f"   Overall average: {overall_avg:+.3f}%")
    print(f"   Overall win rate: {overall_win:.1f}%")
    
    best_song = all_patterns_df.groupby('song')['price_change_pct'].mean().idxmax()
    best_avg = all_patterns_df.groupby('song')['price_change_pct'].mean().max()
    
    print(f"   Best performer: {best_song} ({best_avg:+.3f}%)")

print("\n" + "="*80)
print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
print("="*80)
