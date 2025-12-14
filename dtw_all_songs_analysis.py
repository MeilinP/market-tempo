"""
DTW Pattern Matching - ALL 8 JAZZ SONGS
Test if patterns consistently match to similar SPX periods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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

def find_similar_patterns(jazz_segment, spx_series, window_size=20, top_n=3):
    """Find SPX windows most similar to jazz segment"""
    similarities = []
    
    for i in range(len(spx_series) - window_size + 1):
        spx_window = spx_series[i:i+window_size]
        distance = dtw_distance(jazz_segment, spx_window)
        similarities.append((i, distance))
    
    similarities.sort(key=lambda x: x[1])
    return similarities[:top_n]

def extract_patterns_from_song(jazz_data, song_name, segment_length=30, max_patterns=10):
    """Extract interesting patterns from a jazz song"""
    
    jazz_energy = jazz_data['rms_energy_norm'].values
    jazz_centroid = jazz_data['spectral_centroid_norm'].values
    
    patterns = []
    
    # Find energy peaks
    peaks, _ = find_peaks(jazz_energy, height=0.6, distance=50)
    
    for peak in peaks[:max_patterns]:
        start = max(0, peak - segment_length//2)
        end = min(len(jazz_energy), peak + segment_length//2)
        
        if end - start == segment_length:
            patterns.append({
                'song': song_name,
                'position': peak,
                'time': jazz_data['time'].iloc[peak],
                'pattern': jazz_energy[start:end],
                'type': 'energy'
            })
    
    # Find centroid peaks
    peaks, _ = find_peaks(jazz_centroid, height=0.6, distance=50)
    
    for peak in peaks[:max_patterns]:
        start = max(0, peak - segment_length//2)
        end = min(len(jazz_centroid), peak + segment_length//2)
        
        if end - start == segment_length:
            patterns.append({
                'song': song_name,
                'position': peak,
                'time': jazz_data['time'].iloc[peak],
                'pattern': jazz_centroid[start:end],
                'type': 'centroid'
            })
    
    return patterns

# Load SPX data
print("="*70)
print("🎺 DTW ANALYSIS - ALL 8 JAZZ SONGS")
print("="*70)

spx = pd.read_csv('/home/claude/spx_processed.csv')
spx_volume = spx['volume_norm'].values
spx_price_range = spx['price_range_norm'].values

print(f"\nSPX data loaded: {len(spx)} trading days")

# Load all jazz songs
jazz_df = pd.read_csv('/mnt/user-data/uploads/jazz_timeseries_with_patterns.csv')
songs = jazz_df['song'].unique()

print(f"Jazz songs: {len(songs)}")
for song in songs:
    print(f"  - {song}")

# Process each song
all_results = []
all_patterns = []

segment_length = 30

print("\n" + "="*70)
print("🔍 EXTRACTING PATTERNS FROM ALL SONGS")
print("="*70)

for song in songs:
    print(f"\n{'='*70}")
    print(f"Processing: {song}")
    print(f"{'='*70}")
    
    # Get song data
    song_data = jazz_df[jazz_df['song'] == song].copy()
    song_data = song_data.reset_index(drop=True)
    
    print(f"Duration: {song_data['time'].max():.2f}s ({song_data['time'].max()/60:.2f} min)")
    print(f"Data points: {len(song_data):,}")
    
    # Extract patterns
    patterns = extract_patterns_from_song(song_data, song, segment_length=segment_length)
    print(f"Patterns extracted: {len(patterns)}")
    
    all_patterns.extend(patterns)
    
    # Match to SPX
    for idx, pattern in enumerate(patterns[:5]):  # Top 5 patterns per song
        feature_type = pattern['type']
        
        if feature_type == 'energy':
            spx_feature = spx_volume
            spx_feature_name = 'volume'
        else:
            spx_feature = spx_price_range
            spx_feature_name = 'price_range'
        
        matches = find_similar_patterns(pattern['pattern'], spx_feature, 
                                        window_size=segment_length, top_n=3)
        
        best_match_pos, best_match_dist = matches[0]
        
        all_results.append({
            'song': song,
            'pattern_idx': idx,
            'jazz_time': pattern['time'],
            'jazz_type': feature_type,
            'spx_position': best_match_pos,
            'spx_feature': spx_feature_name,
            'dtw_distance': best_match_dist,
            'similarity_score': 1 / (1 + best_match_dist)
        })
        
        if idx == 0:  # Print best match for each song
            print(f"  Best match: Position {best_match_pos}, Distance: {best_match_dist:.3f}")

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

print("\n" + "="*70)
print("📊 OVERALL STATISTICS")
print("="*70)

print(f"\nTotal patterns analyzed: {len(all_patterns)}")
print(f"Total matches found: {len(results_df)}")

print(f"\nDTW Distance Statistics:")
print(f"  Mean: {results_df['dtw_distance'].mean():.4f}")
print(f"  Median: {results_df['dtw_distance'].median():.4f}")
print(f"  Min: {results_df['dtw_distance'].min():.4f}")
print(f"  Max: {results_df['dtw_distance'].max():.4f}")

print(f"\nSimilarity Score Statistics:")
print(f"  Mean: {results_df['similarity_score'].mean():.4f}")
print(f"  Median: {results_df['similarity_score'].median():.4f}")

# Save results
results_df.to_csv('/home/claude/dtw_all_songs_results.csv', index=False)
print(f"\n✅ Saved: dtw_all_songs_results.csv")

# =================================================================
# ANALYZE SPX POSITION CLUSTERING
# =================================================================

print("\n" + "="*70)
print("📍 SPX POSITION CLUSTERING ANALYSIS")
print("="*70)

print("\nWhere do patterns cluster in SPX timeline?")

# Create histogram of SPX positions
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.hist(results_df['spx_position'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
plt.xlabel('SPX Position (Trading Day)', fontsize=12)
plt.ylabel('Number of Matches', fontsize=12)
plt.title('Distribution of Pattern Matches Across SPX Timeline', fontsize=13, fontweight='bold')
plt.axvline(x=results_df['spx_position'].mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {results_df["spx_position"].mean():.0f}')
plt.axvline(x=results_df['spx_position'].median(), color='green', linestyle='--', 
           linewidth=2, label=f'Median: {results_df["spx_position"].median():.0f}')
plt.legend()
plt.grid(True, alpha=0.3)

# By song
plt.subplot(2, 1, 2)
for song in songs:
    song_results = results_df[results_df['song'] == song]
    plt.scatter(song_results['spx_position'], [song]*len(song_results), 
               alpha=0.6, s=100, label=song)

plt.ylabel('Song', fontsize=12)
plt.xlabel('SPX Position (Trading Day)', fontsize=12)
plt.title('Pattern Matches by Song', fontsize=13, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/dtw_all_songs_clustering.png', dpi=150, bbox_inches='tight')
print("✅ Saved: dtw_all_songs_clustering.png")

# =================================================================
# BEST MATCHES PER SONG
# =================================================================

print("\n" + "="*70)
print("🏆 BEST MATCH PER SONG")
print("="*70)

best_per_song = results_df.loc[results_df.groupby('song')['dtw_distance'].idxmin()]
best_per_song = best_per_song.sort_values('dtw_distance')

print("\n" + best_per_song[['song', 'jazz_type', 'spx_position', 'dtw_distance', 'similarity_score']].to_string(index=False))

# Save best matches
best_per_song.to_csv('/home/claude/dtw_best_per_song.csv', index=False)
print("\n✅ Saved: dtw_best_per_song.csv")

# =================================================================
# COMPARE ENERGY VS CENTROID
# =================================================================

print("\n" + "="*70)
print("⚡ ENERGY VS CENTROID COMPARISON")
print("="*70)

energy_results = results_df[results_df['jazz_type'] == 'energy']
centroid_results = results_df[results_df['jazz_type'] == 'centroid']

print(f"\nEnergy patterns: {len(energy_results)}")
print(f"  Mean DTW distance: {energy_results['dtw_distance'].mean():.4f}")
print(f"  Best match: {energy_results['dtw_distance'].min():.4f}")

print(f"\nCentroid patterns: {len(centroid_results)}")
print(f"  Mean DTW distance: {centroid_results['dtw_distance'].mean():.4f}")
print(f"  Best match: {centroid_results['dtw_distance'].min():.4f}")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_ind(energy_results['dtw_distance'], 
                                   centroid_results['dtw_distance'])

print(f"\nT-test: Energy vs Centroid")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    if energy_results['dtw_distance'].mean() < centroid_results['dtw_distance'].mean():
        print("  ✅ Energy patterns match significantly better!")
    else:
        print("  ✅ Centroid patterns match significantly better!")
else:
    print("  ⚠️ No significant difference")

# =================================================================
# VISUALIZATION: BEST MATCHES ACROSS ALL SONGS
# =================================================================

print("\n" + "="*70)
print("📊 CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(4, 2, figsize=(16, 14))
fig.suptitle('Best DTW Matches - All 8 Jazz Songs', fontsize=14, fontweight='bold')

for idx, (_, row) in enumerate(best_per_song.iterrows()):
    if idx >= 8:
        break
    
    # Get the pattern
    song_patterns = [p for p in all_patterns if p['song'] == row['song']]
    if len(song_patterns) > row['pattern_idx']:
        pattern = song_patterns[int(row['pattern_idx'])]['pattern']
    else:
        continue
    
    # Get SPX match
    spx_pos = int(row['spx_position'])
    if row['jazz_type'] == 'energy':
        spx_pattern = spx_volume[spx_pos:spx_pos+segment_length]
    else:
        spx_pattern = spx_price_range[spx_pos:spx_pos+segment_length]
    
    # Plot
    ax = axes[idx//2, idx%2]
    ax.plot(pattern, 'b-', linewidth=2, label='Jazz', alpha=0.7)
    ax.plot(spx_pattern, 'r--', linewidth=2, label='SPX', alpha=0.7)
    ax.set_title(f"{row['song']}\nDist: {row['dtw_distance']:.2f}, Pos: {spx_pos}", 
                fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/dtw_all_songs_best_matches.png', dpi=150, bbox_inches='tight')
print("✅ Saved: dtw_all_songs_best_matches.png")

# =================================================================
# SUMMARY REPORT
# =================================================================

print("\n" + "="*70)
print("📋 SUMMARY REPORT")
print("="*70)

print(f"\n🎺 Songs Analyzed: {len(songs)}")
print(f"📊 Total Patterns: {len(all_patterns)}")
print(f"🎯 Total Matches: {len(results_df)}")

print(f"\n🏆 Best Overall Match:")
best = results_df.loc[results_df['dtw_distance'].idxmin()]
print(f"  Song: {best['song']}")
print(f"  Type: {best['jazz_type']}")
print(f"  SPX Position: {best['spx_position']:.0f}")
print(f"  DTW Distance: {best['dtw_distance']:.4f}")

print(f"\n📍 SPX Position Clustering:")
print(f"  Mean position: {results_df['spx_position'].mean():.0f} (Day {results_df['spx_position'].mean():.0f})")
print(f"  Median position: {results_df['spx_position'].median():.0f}")
print(f"  Most common range: {results_df['spx_position'].mode().values[0] if len(results_df['spx_position'].mode()) > 0 else 'N/A'}")

# Check if clustering exists
position_std = results_df['spx_position'].std()
position_range = results_df['spx_position'].max() - results_df['spx_position'].min()

print(f"  Standard deviation: {position_std:.1f} days")
print(f"  Range: {position_range:.0f} days")

if position_std < 50:
    print("  ✅ Strong clustering detected!")
elif position_std < 80:
    print("  ⚠️ Moderate clustering")
else:
    print("  ❌ Patterns spread across timeline")

print("\n" + "="*70)
print("✨ ANALYSIS COMPLETE")
print("="*70)
print("\nKey Questions Answered:")
print("1. Do all songs match to similar SPX periods? → Check clustering chart")
print("2. Which feature works better? → Energy vs Centroid comparison")
print("3. What's the best overall match? → See best matches visualization")
