"""
Complete Intraday DTW Analysis - ALL 8 JAZZ SONGS vs SPY
Using SPY (no after-hours data) for cleaner analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def dtw_distance(s1, s2):
    """Calculate DTW distance between two sequences"""
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

def find_matching_windows(jazz_pattern, spy_series, window_size, top_n=5):
    """Find best matching windows in SPY data"""
    matches = []
    
    for i in range(len(spy_series) - window_size + 1):
        spy_window = spy_series[i:i+window_size]
        
        if len(spy_window) == window_size:
            distance = dtw_distance(jazz_pattern, spy_window)
            matches.append((i, distance))
    
    matches.sort(key=lambda x: x[1])
    return matches[:top_n]

print("="*80)
print("🎺 COMPLETE INTRADAY DTW ANALYSIS - ALL 8 SONGS vs SPY")
print("="*80)

# Load SPY intraday data
print("\n📈 Loading SPY intraday data...")
spy_raw = pd.read_csv('/mnt/user-data/uploads/SPY_intraday_1m_7d.csv', skiprows=2)
spy_raw.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
spy_raw['Datetime'] = pd.to_datetime(spy_raw['Datetime'])
spy_raw = spy_raw.sort_values('Datetime').reset_index(drop=True)

# Clean data
for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
    spy_raw[col] = pd.to_numeric(spy_raw[col], errors='coerce')

spy_raw = spy_raw.dropna()

# Filter to market hours only (9:30 - 16:00 EST)
spy_raw['hour'] = spy_raw['Datetime'].dt.hour
spy_raw['minute'] = spy_raw['Datetime'].dt.minute
spy_raw = spy_raw[
    ((spy_raw['hour'] == 9) & (spy_raw['minute'] >= 30)) |
    ((spy_raw['hour'] >= 10) & (spy_raw['hour'] < 16)) |
    ((spy_raw['hour'] == 16) & (spy_raw['minute'] == 0))
].copy()

print(f"SPY market hours data: {len(spy_raw):,} minutes")
print(f"Date range: {spy_raw['Datetime'].min()} to {spy_raw['Datetime'].max()}")

# Calculate SPY features
print("\n🔧 Calculating SPY features...")
spy_raw['price_range'] = spy_raw['High'] - spy_raw['Low']
spy_raw['price_change'] = spy_raw['Close'].diff().abs()
spy_raw['returns'] = spy_raw['Close'].pct_change()

# Normalize
spy_raw['volume_norm'] = (spy_raw['Volume'] - spy_raw['Volume'].min()) / (spy_raw['Volume'].max() - spy_raw['Volume'].min())
spy_raw['price_range_norm'] = (spy_raw['price_range'] - spy_raw['price_range'].min()) / (spy_raw['price_range'].max() - spy_raw['price_range'].min())
spy_raw['returns_abs'] = spy_raw['returns'].abs()
spy_raw['returns_norm'] = (spy_raw['returns_abs'] - spy_raw['returns_abs'].min()) / (spy_raw['returns_abs'].max() - spy_raw['returns_abs'].min())

spy = spy_raw.dropna().reset_index(drop=True)

print(f"Clean data points: {len(spy):,}")
print(f"Trading days: {spy['Datetime'].dt.date.nunique()}")

# Load all jazz songs
print("\n🎺 Loading all jazz songs...")
jazz_df = pd.read_csv('/mnt/user-data/uploads/jazz_timeseries_with_patterns.csv')
songs = sorted(jazz_df['song'].unique())

print(f"Songs: {len(songs)}")
for song in songs:
    duration = jazz_df[jazz_df['song'] == song]['time'].max()
    print(f"  - {song:20s} ({duration/60:5.2f} min)")

# =================================================================
# ANALYZE ALL SONGS
# =================================================================

all_results = []

for song_name in songs:
    print(f"\n{'='*80}")
    print(f"🎵 {song_name.upper()}")
    print(f"{'='*80}")
    
    song_data = jazz_df[jazz_df['song'] == song_name].reset_index(drop=True)
    
    duration_seconds = song_data['time'].max()
    duration_minutes = duration_seconds / 60
    duration_datapoints = len(song_data)
    
    print(f"Duration: {duration_minutes:.2f} minutes ({duration_datapoints:,} data points)")
    
    # Calculate SPY window size in minutes
    spy_window_size = max(2, int(duration_minutes))  # At least 2 minutes
    
    print(f"SPY window: {spy_window_size} minutes")
    
    # Downsample jazz to 1-minute resolution
    jazz_energy = song_data['rms_energy_norm'].values
    jazz_centroid = song_data['spectral_centroid_norm'].values
    
    downsample_factor = max(1, len(jazz_energy) // spy_window_size)
    jazz_energy_downsampled = jazz_energy[::downsample_factor][:spy_window_size]
    jazz_centroid_downsampled = jazz_centroid[::downsample_factor][:spy_window_size]
    
    print(f"Downsampled: {len(jazz_energy_downsampled)} points")
    
    # Match energy to volume
    print(f"\n🔍 Energy → Volume matching...")
    energy_matches = find_matching_windows(
        jazz_energy_downsampled,
        spy['volume_norm'].values,
        window_size=spy_window_size,
        top_n=10
    )
    
    best_energy_match = energy_matches[0]
    
    # Match centroid to price range
    print(f"🔍 Centroid → Price Range matching...")
    centroid_matches = find_matching_windows(
        jazz_centroid_downsampled,
        spy['price_range_norm'].values,
        window_size=spy_window_size,
        top_n=10
    )
    
    best_centroid_match = centroid_matches[0]
    
    # Determine best overall match
    if best_energy_match[1] < best_centroid_match[1]:
        best_match = best_energy_match
        best_type = 'energy'
    else:
        best_match = best_centroid_match
        best_type = 'centroid'
    
    pos, dist = best_match
    start_time = spy['Datetime'].iloc[pos]
    end_time = spy['Datetime'].iloc[min(pos + spy_window_size - 1, len(spy)-1)]
    
    print(f"\n✨ BEST MATCH ({best_type}):")
    print(f"   DTW Distance: {dist:.4f}")
    print(f"   Time: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}")
    print(f"   Position: {pos}")
    
    # Show top 3 energy matches
    print(f"\n📊 Top 3 Energy→Volume matches:")
    for rank, (p, d) in enumerate(energy_matches[:3], 1):
        t = spy['Datetime'].iloc[p]
        print(f"   {rank}. {t.strftime('%m-%d %H:%M')} | DTW: {d:.4f}")
    
    # Show top 3 centroid matches
    print(f"\n📊 Top 3 Centroid→PriceRange matches:")
    for rank, (p, d) in enumerate(centroid_matches[:3], 1):
        t = spy['Datetime'].iloc[p]
        print(f"   {rank}. {t.strftime('%m-%d %H:%M')} | DTW: {d:.4f}")
    
    # Store results
    all_results.append({
        'song': song_name,
        'duration_minutes': duration_minutes,
        'window_size': spy_window_size,
        'best_type': best_type,
        'best_dtw': dist,
        'best_position': pos,
        'start_time': start_time,
        'end_time': end_time,
        'energy_dtw': best_energy_match[1],
        'centroid_dtw': best_centroid_match[1]
    })

# =================================================================
# SAVE RESULTS
# =================================================================

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('best_dtw')
results_df.to_csv('/home/claude/spy_intraday_all_songs.csv', index=False)
print(f"\n✅ Saved: spy_intraday_all_songs.csv")

# =================================================================
# SUMMARY STATISTICS
# =================================================================

print("\n" + "="*80)
print("📊 SUMMARY - ALL 8 SONGS")
print("="*80)

print(f"\n🏆 RANKING BY BEST DTW DISTANCE:")
print(results_df[['song', 'best_type', 'best_dtw', 'start_time', 'duration_minutes']].to_string(index=False))

print(f"\n📈 STATISTICS:")
print(f"   Mean DTW: {results_df['best_dtw'].mean():.4f}")
print(f"   Median DTW: {results_df['best_dtw'].median():.4f}")
print(f"   Min DTW: {results_df['best_dtw'].min():.4f} ({results_df.iloc[0]['song']})")
print(f"   Max DTW: {results_df['best_dtw'].max():.4f}")

print(f"\n⚡ FEATURE COMPARISON:")
print(f"   Avg Energy DTW: {results_df['energy_dtw'].mean():.4f}")
print(f"   Avg Centroid DTW: {results_df['centroid_dtw'].mean():.4f}")

energy_better = (results_df['energy_dtw'] < results_df['centroid_dtw']).sum()
print(f"   Energy better: {energy_better}/8 songs")
print(f"   Centroid better: {8-energy_better}/8 songs")

# =================================================================
# VISUALIZATIONS
# =================================================================

print("\n" + "="*80)
print("📊 CREATING VISUALIZATIONS")
print("="*80)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Plot best matches for all 8 songs
for idx, (_, row) in enumerate(results_df.iterrows()):
    song_name = row['song']
    
    song_data = jazz_df[jazz_df['song'] == song_name].reset_index(drop=True)
    
    window_size = int(row['window_size'])
    pos = int(row['best_position'])
    
    # Get jazz pattern
    jazz_energy = song_data['rms_energy_norm'].values
    downsample_factor = max(1, len(jazz_energy) // window_size)
    jazz_downsampled = jazz_energy[::downsample_factor][:window_size]
    
    # Get SPY pattern
    if row['best_type'] == 'energy':
        spy_pattern = spy['volume_norm'].iloc[pos:pos+window_size].values
        ylabel = 'Volume'
    else:
        spy_pattern = spy['price_range_norm'].iloc[pos:pos+window_size].values
        ylabel = 'Price Range'
    
    # Plot
    ax = fig.add_subplot(gs[idx//2, idx%2])
    
    x = range(max(len(jazz_downsampled), len(spy_pattern)))
    ax.plot(jazz_downsampled, 'b-', linewidth=2.5, alpha=0.7, label='Jazz Energy')
    ax.plot(spy_pattern, 'r--', linewidth=2.5, alpha=0.7, label=f'SPY {ylabel}')
    
    ax.set_title(f"{song_name.replace('_', ' ').title()}\nDTW: {row['best_dtw']:.4f} | {row['start_time'].strftime('%m-%d %H:%M')}", 
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Minutes', fontsize=10)
    ax.set_ylabel('Normalized', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

plt.suptitle('All 8 Jazz Songs - Best Intraday Matches with SPY', fontsize=14, fontweight='bold')
plt.savefig('/mnt/user-data/outputs/spy_all_songs_matches.png', dpi=150, bbox_inches='tight')
print("✅ Saved: spy_all_songs_matches.png")

# Bar chart comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# DTW distances
ax = axes[0]
x = range(len(results_df))
colors = ['green' if d < 0.5 else 'orange' if d < 1.0 else 'red' for d in results_df['best_dtw']]
ax.bar(x, results_df['best_dtw'], color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels([s.replace('_', '\n') for s in results_df['song']], rotation=0, fontsize=10)
ax.set_ylabel('DTW Distance', fontsize=11)
ax.set_title('DTW Distance - All Songs (Lower is Better)', fontsize=12, fontweight='bold')
ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Excellent (<0.5)')
ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Good (<1.0)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Energy vs Centroid
ax = axes[1]
x_pos = np.arange(len(results_df))
width = 0.35

ax.bar(x_pos - width/2, results_df['energy_dtw'], width, label='Energy→Volume', 
      alpha=0.7, color='steelblue', edgecolor='black')
ax.bar(x_pos + width/2, results_df['centroid_dtw'], width, label='Centroid→PriceRange', 
      alpha=0.7, color='coral', edgecolor='black')

ax.set_xticks(x_pos)
ax.set_xticklabels([s.replace('_', '\n') for s in results_df['song']], rotation=0, fontsize=10)
ax.set_ylabel('DTW Distance', fontsize=11)
ax.set_title('Energy vs Centroid - Which Feature Matches Better?', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/spy_dtw_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: spy_dtw_comparison.png")

# =================================================================
# FINAL SUMMARY
# =================================================================

print("\n" + "="*80)
print("✨ ANALYSIS COMPLETE - KEY FINDINGS")
print("="*80)

excellent = results_df[results_df['best_dtw'] < 0.5]
good = results_df[(results_df['best_dtw'] >= 0.5) & (results_df['best_dtw'] < 1.0)]
moderate = results_df[results_df['best_dtw'] >= 1.0]

print(f"\n🏆 Match Quality:")
print(f"   Excellent (DTW < 0.5): {len(excellent)} songs")
if len(excellent) > 0:
    print(f"      {', '.join(excellent['song'].values)}")

print(f"   Good (0.5 ≤ DTW < 1.0): {len(good)} songs")
if len(good) > 0:
    print(f"      {', '.join(good['song'].values)}")

print(f"   Moderate (DTW ≥ 1.0): {len(moderate)} songs")
if len(moderate) > 0:
    print(f"      {', '.join(moderate['song'].values)}")

print(f"\n🎯 Best Match Overall:")
best = results_df.iloc[0]
print(f"   Song: {best['song']}")
print(f"   DTW: {best['best_dtw']:.4f}")
print(f"   Type: {best['best_type']}")
print(f"   Time: {best['start_time'].strftime('%Y-%m-%d %H:%M')}")

print(f"\n💡 Conclusion:")
if len(excellent) >= 3:
    print("   ✅ Strong evidence! Multiple songs show excellent matches!")
elif len(excellent) + len(good) >= 5:
    print("   ✅ Promising! Most songs show good matches!")
else:
    print("   ⚠️ Mixed results. Some songs match well, others don't.")

print("\n" + "="*80)
