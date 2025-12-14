"""
Visualize Jazz vs SPX comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load aligned data
aligned = pd.read_csv('/home/claude/jazz_spx_aligned.csv')

# Create comprehensive visualization
fig, axes = plt.subplots(4, 1, figsize=(16, 12))
fig.suptitle("Now's the Time (Jazz) vs SPX 2024 - Pattern Comparison", 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Energy vs Volume
ax = axes[0]
ax2 = ax.twinx()
ax.plot(aligned['jazz_energy'], 'b-', alpha=0.7, linewidth=1.5, label='Jazz Energy')
ax2.plot(aligned['spx_volume'], 'r-', alpha=0.7, linewidth=1.5, label='SPX Volume')
ax.set_ylabel('Jazz RMS Energy (norm)', color='b', fontsize=11)
ax2.set_ylabel('SPX Volume (norm)', color='r', fontsize=11)
ax.set_title('Energy vs Volume (Correlation: -0.0048)', fontsize=12, pad=10)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 2. Centroid vs Price Range  
ax = axes[1]
ax2 = ax.twinx()
ax.plot(aligned['jazz_centroid'], 'b-', alpha=0.7, linewidth=1.5, label='Jazz Spectral Centroid')
ax2.plot(aligned['spx_price_range'], 'r-', alpha=0.7, linewidth=1.5, label='SPX Price Range')
ax.set_ylabel('Jazz Centroid (norm)', color='b', fontsize=11)
ax2.set_ylabel('SPX Price Range (norm)', color='r', fontsize=11)
ax.set_title('Spectral Centroid vs Price Range (Correlation: 0.2610) ⭐ Strongest', 
             fontsize=12, pad=10)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 3. Onset vs Momentum
ax = axes[2]
ax2 = ax.twinx()
ax.plot(aligned['jazz_onset'], 'b-', alpha=0.7, linewidth=1.5, label='Jazz Onset Strength')
ax2.plot(aligned['spx_momentum'], 'r-', alpha=0.7, linewidth=1.5, label='SPX Price Momentum')
ax.set_ylabel('Jazz Onset (norm)', color='b', fontsize=11)
ax2.set_ylabel('SPX Momentum (norm)', color='r', fontsize=11)
ax.set_title('Onset Strength vs Price Momentum (Correlation: 0.0774)', fontsize=12, pad=10)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 4. Scatter plot of strongest correlation
ax = axes[3]
ax.scatter(aligned['jazz_centroid'], aligned['spx_price_range'], 
          alpha=0.5, s=30, c=np.arange(len(aligned)), cmap='viridis')
ax.set_xlabel('Jazz Spectral Centroid (normalized)', fontsize=11)
ax.set_ylabel('SPX Price Range (normalized)', fontsize=11)
ax.set_title('Scatter: Centroid vs Price Range (r=0.2610)', fontsize=12, pad=10)

# Add trend line
z = np.polyfit(aligned['jazz_centroid'], aligned['spx_price_range'], 1)
p = np.poly1d(z)
ax.plot(aligned['jazz_centroid'], p(aligned['jazz_centroid']), 
       "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/jazz_spx_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: jazz_spx_comparison.png")

# Create summary statistics plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Statistical Analysis Summary", fontsize=14, fontweight='bold')

# Correlation bar chart
correlations = pd.read_csv('/home/claude/jazz_spx_correlations.csv')
ax = axes[0, 0]
colors = ['green' if x > 0 else 'red' for x in correlations['Correlation']]
ax.barh(correlations['Metric'], correlations['Correlation'], color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Correlation Coefficient')
ax.set_title('Feature Correlations')
ax.grid(True, alpha=0.3, axis='x')

# Distribution comparison - Energy vs Volume
ax = axes[0, 1]
ax.hist(aligned['jazz_energy'], bins=30, alpha=0.6, label='Jazz Energy', color='blue')
ax.hist(aligned['spx_volume'], bins=30, alpha=0.6, label='SPX Volume', color='red')
ax.set_xlabel('Normalized Value')
ax.set_ylabel('Frequency')
ax.set_title('Distribution: Energy vs Volume')
ax.legend()
ax.grid(True, alpha=0.3)

# Distribution comparison - Centroid vs Price Range
ax = axes[1, 0]
ax.hist(aligned['jazz_centroid'], bins=30, alpha=0.6, label='Jazz Centroid', color='blue')
ax.hist(aligned['spx_price_range'], bins=30, alpha=0.6, label='SPX Price Range', color='red')
ax.set_xlabel('Normalized Value')
ax.set_ylabel('Frequency')
ax.set_title('Distribution: Centroid vs Price Range')
ax.legend()
ax.grid(True, alpha=0.3)

# Key statistics table
ax = axes[1, 1]
ax.axis('off')
stats_data = [
    ['Metric', 'Jazz', 'SPX'],
    ['Data Points', '8,151', '247'],
    ['Duration', '3.15 min', '247 days'],
    ['Energy Cycles', '627', '~49'],
    ['Volume Cycle', '-', '5 days'],
    ['Momentum Cycle', '-', '17 days'],
    ['Best Correlation', 'Centroid', 'Price Range'],
    ['Correlation Value', '0.2610', '0.2610'],
]

table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Summary Statistics', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/jazz_spx_statistics.png', dpi=150, bbox_inches='tight')
print("✅ Saved: jazz_spx_statistics.png")

print("\n" + "="*70)
print("📊 VISUALIZATION COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  1. jazz_spx_comparison.png - Main comparison charts")
print("  2. jazz_spx_statistics.png - Statistical summary")
