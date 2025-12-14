"""
DTW Pattern Matching with INTRADAY SPX Data
Now we have matching time scales!
Jazz: ~3 minutes vs SPX: ~3 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime, timedelta
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

def find_matching_windows(jazz_pattern, spx_series, window_size, top_n=5):
    """Find best matching windows in SPX data"""
    matches = []
    
    for i in range(len(spx_series) - window_size + 1):
        spx_window = spx_series[i:i+window_size]
        
        if len(spx_window) == window_size:
            distance = dtw_distance(jazz_pattern, spx_window)
            matches.append((i, distance))
    
    matches.sort(key=lambda x: x[1])
    return matches[:top_n]

print("="*70)
print("🎺 INTRADAY DTW ANALYSIS")
print("="*70)

# Load intraday SPX data
print("\n📈 Loading intraday SPX data...")
spx_raw = pd.read_csv('/mnt/user-data/uploads/_GSPC_intraday_1m_7d.csv', skiprows=2)
spx_raw.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
spx_raw['Datetime'] = pd.to_datetime(spx_raw['Datetime'])
spx_raw = spx_raw.sort_values('Datetime').reset_index(drop=True)

# Clean data
spx_raw['Close'] = pd.to_numeric(spx_raw['Close'], errors='coerce')
spx_raw['Volume'] = pd.to_numeric(spx_raw['Volume'], errors='coerce')
spx_raw['High'] = pd.to_numeric(spx_raw['High'], errors='coerce')
spx_raw['Low'] = pd.to_numeric(spx_raw['Low'], errors='coerce')
spx_raw = spx_raw.dropna()

print(f"SPX intraday data: {len(spx_raw):,} minutes")
print(f"Date range: {spx_raw['Datetime'].min()} to {spx_raw['Datetime'].max()}")
print(f"Total duration: {(spx_raw['Datetime'].max() - spx_raw['Datetime'].min()).total_seconds()/3600:.1f} hours")

# Calculate features
print("\n🔧 Calculating SPX features...")
spx_raw['price_range'] = spx_raw['High'] - spx_raw['Low']
spx_raw['price_change'] = spx_raw['Close'].diff().abs()
spx_raw['returns'] = spx_raw['Close'].pct_change()

# Normalize
spx_raw['volume_norm'] = (spx_raw['Volume'] - spx_raw['Volume'].min()) / (spx_raw['Volume'].max() - spx_raw['Volume'].min())
spx_raw['price_range_norm'] = (spx_raw['price_range'] - spx_raw['price_range'].min()) / (spx_raw['price_range'].max() - spx_raw['price_range'].min())
spx_raw['price_change_norm'] = (spx_raw['price_change'] - spx_raw['price_change'].min()) / (spx_raw['price_change'].max() - spx_raw['price_change'].min())

spx = spx_raw.dropna()

print(f"Clean data points: {len(spx):,}")

# Load jazz data
print("\n🎺 Loading jazz songs...")
jazz_df = pd.read_csv('/mnt/user-data/uploads/jazz_timeseries_with_patterns.csv')
songs = jazz_df['song'].unique()

print(f"Songs available: {len(songs)}")

# Select songs to analyze
# Use the best performers from daily analysis
priority_songs = ['tenor_madness', 'au_privave', 'blue_monk', 'bilies_bounce']
songs_to_analyze = [s for s in priority_songs if s in songs]

print(f"\nAnalyzing: {songs_to_analyze}")

# =================================================================
# MATCH EACH SONG TO INTRADAY DATA
# =================================================================

all_results = []

for song_name in songs_to_analyze[:2]:  # Start with top 2
    print(f"\n{'='*70}")
    print(f"🎵 Processing: {song_name}")
    print(f"{'='*70}")
    
    song_data = jazz_df[jazz_df['song'] == song_name].reset_index(drop=True)
    
    duration_minutes = song_data['time'].max() / 60
    duration_datapoints = len(song_data)
    
    print(f"Duration: {duration_minutes:.2f} minutes ({duration_datapoints:,} data points)")
    
    # Calculate how many SPX minutes correspond to song length
    # Jazz is sampled at ~43 points/second
    # SPX is sampled at 1 point/minute
    # So we need to downsample jazz or think in terms of time
    
    # Strategy: Use song duration in minutes to match SPX window
    spx_window_size = int(duration_minutes)  # minutes
    
    print(f"SPX window size: {spx_window_size} minutes")
    
    # Get jazz energy pattern (full song)
    jazz_energy = song_data['rms_energy_norm'].values
    
    # Downsample jazz to match 1-minute resolution
    downsample_factor = len(jazz_energy) // spx_window_size
    jazz_energy_downsampled = jazz_energy[::downsample_factor][:spx_window_size]
    
    print(f"Jazz energy downsampled: {len(jazz_energy_downsampled)} points")
    
    # Find matching windows in SPX
    print(f"\n🔍 Searching for matches in SPX data...")
    matches = find_matching_windows(
        jazz_energy_downsampled,
        spx['volume_norm'].values,
        window_size=spx_window_size,
        top_n=10
    )
    
    print(f"\nTop 5 matches:")
    for rank, (pos, dist) in enumerate(matches[:5], 1):
        start_time = spx['Datetime'].iloc[pos]
        end_time = spx['Datetime'].iloc[min(pos + spx_window_size, len(spx)-1)]
        print(f"  {rank}. Position {pos:4d} | Distance: {dist:6.3f} | {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}")
        
        all_results.append({
            'song': song_name,
            'rank': rank,
            'spx_position': pos,
            'dtw_distance': dist,
            'start_time': start_time,
            'end_time': end_time,
            'duration_minutes': duration_minutes
        })

# Save results
results_df = pd.DataFrame(all_results)
results_df.to_csv('/home/claude/intraday_dtw_results.csv', index=False)
print(f"\n✅ Saved: intraday_dtw_results.csv")

# =================================================================
# VISUALIZATION
# =================================================================

print("\n" + "="*70)
print("📊 CREATING VISUALIZATIONS")
print("="*70)

# Visualize best matches
fig, axes = plt.subplots(len(songs_to_analyze[:2]), 2, figsize=(18, 6*len(songs_to_analyze[:2])))
if len(songs_to_analyze[:2]) == 1:
    axes = axes.reshape(1, -1)

for idx, song_name in enumerate(songs_to_analyze[:2]):
    song_data = jazz_df[jazz_df['song'] == song_name].reset_index(drop=True)
    
    duration_minutes = song_data['time'].max() / 60
    spx_window_size = int(duration_minutes)
    
    # Get downsampled jazz
    jazz_energy = song_data['rms_energy_norm'].values
    downsample_factor = len(jazz_energy) // spx_window_size
    jazz_energy_downsampled = jazz_energy[::downsample_factor][:spx_window_size]
    
    # Get best match
    best_matches = results_df[(results_df['song'] == song_name) & (results_df['rank'] == 1)]
    
    if len(best_matches) > 0:
        best_match = best_matches.iloc[0]
        pos = int(best_match['spx_position'])
        
        # Get SPX window
        spx_window = spx['volume_norm'].iloc[pos:pos+spx_window_size].values
        spx_times = spx['Datetime'].iloc[pos:pos+spx_window_size].values
        
        # Left: Jazz pattern
        ax = axes[idx, 0]
        ax.plot(range(len(jazz_energy_downsampled)), jazz_energy_downsampled, 
               'b-', linewidth=2, label='Jazz Energy')
        ax.set_title(f'{song_name.replace("_", " ").title()}\n({duration_minutes:.1f} minutes)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Normalized Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Right: SPX match
        ax = axes[idx, 1]
        ax.plot(range(len(spx_window)), spx_window, 'r-', linewidth=2, label='SPX Volume')
        ax.set_title(f'SPX Match (DTW: {best_match["dtw_distance"]:.3f})\n{best_match["start_time"].strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Normalized Volume')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/intraday_dtw_matches.png', dpi=150, bbox_inches='tight')
print("✅ Saved: intraday_dtw_matches.png")

# Overlay comparison
fig, axes = plt.subplots(len(songs_to_analyze[:2]), 1, figsize=(16, 5*len(songs_to_analyze[:2])))
if len(songs_to_analyze[:2]) == 1:
    axes = [axes]

for idx, song_name in enumerate(songs_to_analyze[:2]):
    song_data = jazz_df[jazz_df['song'] == song_name].reset_index(drop=True)
    
    duration_minutes = song_data['time'].max() / 60
    spx_window_size = int(duration_minutes)
    
    jazz_energy = song_data['rms_energy_norm'].values
    downsample_factor = len(jazz_energy) // spx_window_size
    jazz_energy_downsampled = jazz_energy[::downsample_factor][:spx_window_size]
    
    best_matches = results_df[(results_df['song'] == song_name) & (results_df['rank'] == 1)]
    
    if len(best_matches) > 0:
        best_match = best_matches.iloc[0]
        pos = int(best_match['spx_position'])
        spx_window = spx['volume_norm'].iloc[pos:pos+spx_window_size].values
        
        ax = axes[idx]
        x = range(max(len(jazz_energy_downsampled), len(spx_window)))
        
        ax.plot(jazz_energy_downsampled, 'b-', linewidth=2.5, alpha=0.7, label='Jazz Energy')
        ax.plot(spx_window, 'r--', linewidth=2.5, alpha=0.7, label='SPX Volume')
        
        ax.set_title(f'{song_name.replace("_", " ").title()} vs SPX (DTW: {best_match["dtw_distance"]:.3f})', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Minutes', fontsize=11)
        ax.set_ylabel('Normalized Value', fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/intraday_dtw_overlay.png', dpi=150, bbox_inches='tight')
print("✅ Saved: intraday_dtw_overlay.png")

# =================================================================
# SUMMARY
# =================================================================

print("\n" + "="*70)
print("📋 SUMMARY")
print("="*70)

print(f"\n✨ INTRADAY ANALYSIS COMPLETE")
print(f"\nSongs analyzed: {len(songs_to_analyze[:2])}")
print(f"SPX data: {len(spx):,} minutes across {(spx['Datetime'].max() - spx['Datetime'].min()).days} days")

print(f"\n🏆 Best Matches:")
best_results = results_df[results_df['rank'] == 1].sort_values('dtw_distance')
print(best_results[['song', 'dtw_distance', 'start_time', 'duration_minutes']].to_string(index=False))

print(f"\n🎯 Key Advantage of Intraday Data:")
print("  - TRUE time scale matching (minutes vs minutes)")
print("  - No need for complex interpolation")
print("  - Can see actual intraday patterns")

print("\n💡 Next Steps:")
print("  1. Analyze remaining songs")
print("  2. Check if matches occur during specific market conditions")
print("  3. Build predictive model based on patterns")
