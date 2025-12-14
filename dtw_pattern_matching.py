"""
Dynamic Time Warping (DTW) Pattern Matching
Find visually similar patterns between Jazz and SPX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Simple DTW implementation (in case dtaidistance has issues)
def dtw_distance(s1, s2):
    """
    Calculate DTW distance between two sequences
    """
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                          dtw_matrix[i, j-1],    # deletion
                                          dtw_matrix[i-1, j-1])  # match
    
    return dtw_matrix[n, m]

def find_similar_patterns(jazz_segment, spx_series, window_size=20, top_n=5):
    """
    Find SPX windows most similar to jazz segment using DTW
    """
    similarities = []
    
    # Slide window across SPX data
    for i in range(len(spx_series) - window_size + 1):
        spx_window = spx_series[i:i+window_size]
        
        # Calculate DTW distance
        distance = dtw_distance(jazz_segment, spx_window)
        
        # Store (position, distance)
        similarities.append((i, distance))
    
    # Sort by distance (lower is more similar)
    similarities.sort(key=lambda x: x[1])
    
    return similarities[:top_n]

# Load data
print("="*70)
print("🎵 PATTERN MATCHING: Jazz vs SPX")
print("="*70)

jazz = pd.read_csv('/home/claude/nows_the_time.csv')
spx = pd.read_csv('/home/claude/spx_processed.csv')

print(f"\nJazz data: {len(jazz):,} points")
print(f"SPX data: {len(spx)} points")

# =================================================================
# STRATEGY: Find interesting jazz patterns and match them to SPX
# =================================================================

print("\n" + "="*70)
print("📊 EXTRACTING JAZZ PATTERNS")
print("="*70)

# Use normalized energy as the main feature
jazz_energy = jazz['rms_energy_norm'].values
jazz_centroid = jazz['spectral_centroid_norm'].values

# Find peaks in jazz energy (interesting moments)
peaks, properties = find_peaks(jazz_energy, height=0.7, distance=50)

print(f"\nFound {len(peaks)} high-energy peaks in jazz")
print(f"Peak locations (in data points): {peaks[:10]}...")

# Extract segments around peaks
segment_length = 30  # data points
jazz_patterns = []

for peak in peaks[:10]:  # Take first 10 peaks
    start = max(0, peak - segment_length//2)
    end = min(len(jazz_energy), peak + segment_length//2)
    
    if end - start == segment_length:
        pattern = jazz_energy[start:end]
        jazz_patterns.append({
            'position': peak,
            'time': jazz['time'].iloc[peak],
            'pattern': pattern,
            'type': 'energy'
        })

print(f"Extracted {len(jazz_patterns)} patterns from jazz")

# Also extract centroid patterns
centroid_peaks, _ = find_peaks(jazz_centroid, height=0.7, distance=50)

for peak in centroid_peaks[:10]:
    start = max(0, peak - segment_length//2)
    end = min(len(jazz_centroid), peak + segment_length//2)
    
    if end - start == segment_length:
        pattern = jazz_centroid[start:end]
        jazz_patterns.append({
            'position': peak,
            'time': jazz['time'].iloc[peak],
            'pattern': pattern,
            'type': 'centroid'
        })

print(f"Total patterns: {len(jazz_patterns)}")

# =================================================================
# MATCH PATTERNS TO SPX
# =================================================================

print("\n" + "="*70)
print("🔍 MATCHING TO SPX DATA")
print("="*70)

spx_volume = spx['volume_norm'].values
spx_price_range = spx['price_range_norm'].values

# Match each jazz pattern to SPX
results = []

for idx, jazz_pat in enumerate(jazz_patterns[:5]):  # First 5 patterns
    pattern = jazz_pat['pattern']
    feature_type = jazz_pat['type']
    
    # Match to corresponding SPX feature
    if feature_type == 'energy':
        spx_feature = spx_volume
        spx_feature_name = 'volume'
    else:
        spx_feature = spx_price_range
        spx_feature_name = 'price_range'
    
    # Find similar windows in SPX
    window_size = segment_length
    matches = find_similar_patterns(pattern, spx_feature, window_size=window_size, top_n=3)
    
    print(f"\nJazz Pattern #{idx+1} ({feature_type} at {jazz_pat['time']:.2f}s):")
    for rank, (pos, dist) in enumerate(matches, 1):
        date = spx.index[pos] if hasattr(spx, 'index') else f"Day {pos}"
        print(f"  Match #{rank}: Position {pos}, Distance: {dist:.4f}")
    
    # Store best match
    best_match_pos, best_match_dist = matches[0]
    results.append({
        'jazz_pattern_idx': idx,
        'jazz_time': jazz_pat['time'],
        'jazz_type': feature_type,
        'spx_position': best_match_pos,
        'spx_feature': spx_feature_name,
        'dtw_distance': best_match_dist,
        'similarity_score': 1 / (1 + best_match_dist)  # Convert to similarity
    })

results_df = pd.DataFrame(results)
results_df.to_csv('/home/claude/dtw_matches.csv', index=False)
print(f"\n✅ Saved: dtw_matches.csv")

# =================================================================
# VISUALIZE BEST MATCHES
# =================================================================

print("\n" + "="*70)
print("📈 CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("DTW Pattern Matching: Jazz vs SPX - Best Matches", 
             fontsize=14, fontweight='bold')

for idx in range(min(3, len(jazz_patterns))):
    jazz_pat = jazz_patterns[idx]
    result = results[idx]
    
    # Left: Jazz pattern
    ax = axes[idx, 0]
    ax.plot(jazz_pat['pattern'], 'b-', linewidth=2)
    ax.set_title(f"Jazz Pattern #{idx+1} ({jazz_pat['type']} at {jazz_pat['time']:.2f}s)", 
                fontsize=11)
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('Data Point')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Right: Matched SPX pattern
    ax = axes[idx, 1]
    
    # Get SPX pattern
    spx_pos = result['spx_position']
    if result['jazz_type'] == 'energy':
        spx_pattern = spx_volume[spx_pos:spx_pos+segment_length]
    else:
        spx_pattern = spx_price_range[spx_pos:spx_pos+segment_length]
    
    ax.plot(spx_pattern, 'r-', linewidth=2)
    ax.set_title(f"SPX Match (Position {spx_pos}, Distance: {result['dtw_distance']:.3f})", 
                fontsize=11)
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('Trading Day')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/dtw_pattern_matches.png', dpi=150, bbox_inches='tight')
print("✅ Saved: dtw_pattern_matches.png")

# Create overlay comparison
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("DTW: Overlaying Jazz Patterns on SPX Data", fontsize=14, fontweight='bold')

# Top: Energy patterns
ax = axes[0]
ax.plot(spx_volume, 'gray', alpha=0.5, linewidth=1, label='SPX Volume')

energy_patterns = [p for p in jazz_patterns if p['type'] == 'energy'][:3]
for idx, pat in enumerate(energy_patterns):
    if idx < len(results):
        pos = results[idx]['spx_position']
        ax.plot(range(pos, pos+segment_length), pat['pattern'], 
               linewidth=2, label=f"Jazz Pattern {idx+1}", alpha=0.7)

ax.set_title('Energy/Volume Pattern Matching', fontsize=12)
ax.set_xlabel('Trading Day')
ax.set_ylabel('Normalized Value')
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom: Centroid patterns
ax = axes[1]
ax.plot(spx_price_range, 'gray', alpha=0.5, linewidth=1, label='SPX Price Range')

centroid_patterns = [p for p in jazz_patterns if p['type'] == 'centroid'][:3]
for idx, pat in enumerate(centroid_patterns):
    matching_result = [r for r in results if r['jazz_type'] == 'centroid']
    if idx < len(matching_result):
        pos = matching_result[idx]['spx_position']
        ax.plot(range(pos, pos+segment_length), pat['pattern'], 
               linewidth=2, label=f"Jazz Pattern {idx+1}", alpha=0.7)

ax.set_title('Centroid/Price Range Pattern Matching', fontsize=12)
ax.set_xlabel('Trading Day')
ax.set_ylabel('Normalized Value')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/dtw_overlay.png', dpi=150, bbox_inches='tight')
print("✅ Saved: dtw_overlay.png")

# =================================================================
# SUMMARY STATISTICS
# =================================================================

print("\n" + "="*70)
print("📊 DTW ANALYSIS SUMMARY")
print("="*70)

print(f"\nAverage DTW distance: {results_df['dtw_distance'].mean():.4f}")
print(f"Min DTW distance: {results_df['dtw_distance'].min():.4f}")
print(f"Max DTW distance: {results_df['dtw_distance'].max():.4f}")
print(f"\nAverage similarity score: {results_df['similarity_score'].mean():.4f}")

print("\nBest matches:")
print(results_df.sort_values('dtw_distance')[['jazz_time', 'jazz_type', 'spx_position', 'dtw_distance']].head())

print("\n" + "="*70)
print("✨ PATTERN MATCHING COMPLETE")
print("="*70)
print("\nInterpretation:")
print("- Lower DTW distance = More similar patterns")
print("- Check visualizations to see if patterns truly look alike")
print("- Next step: Validate if similar patterns predict market moves")
