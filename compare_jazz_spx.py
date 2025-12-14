"""
Compare "Now's the Time" jazz features with SPX market data
Find correlations and patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from datetime import datetime

# Load data
print("="*70)
print("🎺 Loading Now's the Time data...")
print("="*70)

jazz = pd.read_csv('/home/claude/nows_the_time.csv')
print(f"Jazz data points: {len(jazz):,}")
print(f"Duration: {jazz['time'].max():.2f} seconds")

print("\n" + "="*70)
print("📈 Loading SPX data...")
print("="*70)

# Load SPX data
spx_raw = pd.read_csv('/mnt/user-data/uploads/_GSPC_historical_data.csv', skiprows=2)
spx_raw.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']  # Rename columns
spx_raw['Date'] = pd.to_datetime(spx_raw['Date'])
spx_raw = spx_raw.sort_values('Date')
spx_raw = spx_raw.set_index('Date')

# Clean and convert to float
spx = pd.DataFrame()
spx['Close'] = pd.to_numeric(spx_raw['Close'], errors='coerce')
spx['Volume'] = pd.to_numeric(spx_raw['Volume'], errors='coerce')
spx['High'] = pd.to_numeric(spx_raw['High'], errors='coerce')
spx['Low'] = pd.to_numeric(spx_raw['Low'], errors='coerce')
spx = spx.dropna()

print(f"SPX data points: {len(spx)}")
print(f"Date range: {spx.index[0].date()} to {spx.index[-1].date()}")

# Calculate SPX features (analogous to jazz features)
print("\n" + "="*70)
print("🔧 Calculating SPX features...")
print("="*70)

# 1. Volume (analogous to RMS Energy)
spx['volume_norm'] = (spx['Volume'] - spx['Volume'].min()) / (spx['Volume'].max() - spx['Volume'].min())

# 2. Price volatility (analogous to Spectral Centroid - market sentiment)
spx['price_range'] = spx['High'] - spx['Low']
spx['price_range_norm'] = (spx['price_range'] - spx['price_range'].min()) / (spx['price_range'].max() - spx['price_range'].min())

# 3. Price momentum (analogous to Onset Strength)
spx['price_change'] = spx['Close'].diff()
spx['price_momentum'] = spx['price_change'].rolling(window=3).mean()
spx['momentum_norm'] = (spx['price_momentum'] - spx['price_momentum'].min()) / (spx['price_momentum'].max() - spx['price_momentum'].min())

# 4. Volatility (analogous to Spectral Flux)
spx['returns'] = spx['Close'].pct_change()
spx['volatility'] = spx['returns'].rolling(window=5).std()
spx['volatility_norm'] = (spx['volatility'] - spx['volatility'].min()) / (spx['volatility'].max() - spx['volatility'].min())

spx = spx.dropna()

print("SPX features calculated:")
print("  - Volume (~ RMS Energy)")
print("  - Price Range (~ Spectral Centroid)")  
print("  - Price Momentum (~ Onset Strength)")
print("  - Volatility (~ Spectral Flux)")

# =================================================================
# TIME SCALE MAPPING
# =================================================================
print("\n" + "="*70)
print("⏰ TIME SCALE CONSIDERATIONS")
print("="*70)

print(f"\nJazz:")
print(f"  Duration: 189 seconds = 3.15 minutes")
print(f"  Data points: 8,151")
print(f"  Sampling: ~43 points/second")

print(f"\nSPX:")
print(f"  Duration: {len(spx)} trading days")
print(f"  Data points: {len(spx)}")
print(f"  Sampling: 1 point/day")

print(f"\nTime scale ratio:")
print(f"  If 1 jazz second = 1 SPX day:")
print(f"    189 seconds ≈ 189 trading days ≈ ~9 months")

# =================================================================
# PATTERN ANALYSIS
# =================================================================
print("\n" + "="*70)
print("🔍 PATTERN ANALYSIS")
print("="*70)

# Jazz patterns
jazz_energy_cycle = 0.30  # seconds (from summary)
jazz_num_cycles = 627

# SPX - let's look for similar cycles
# If 1 second = 1 day, then 0.30 seconds ≈ same-day patterns
# Or we can look at different time scales

print(f"\nJazz Energy Cycle: {jazz_energy_cycle:.2f} seconds")
print(f"Number of cycles in song: {jazz_num_cycles:.0f}")

# Try to find cycles in SPX data
def find_cycles(data, min_distance=5):
    """Find repeating patterns using autocorrelation"""
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    peaks, properties = signal.find_peaks(autocorr, height=autocorr.max()*0.3, distance=min_distance)
    
    if len(peaks) > 0:
        return peaks[0], len(data) / peaks[0]
    return None, 0

spx_volume_cycle, spx_volume_num = find_cycles(spx['volume_norm'].values)
spx_momentum_cycle, spx_momentum_num = find_cycles(spx['momentum_norm'].values)

print(f"\nSPX Volume Cycle: {spx_volume_cycle if spx_volume_cycle else 'N/A'} days")
print(f"SPX Momentum Cycle: {spx_momentum_cycle if spx_momentum_cycle else 'N/A'} days")

# =================================================================
# CORRELATION ANALYSIS
# =================================================================
print("\n" + "="*70)
print("📊 FEATURE CORRELATION")
print("="*70)

# Resample to same length for correlation
# Take a subset of SPX data with similar number of points to jazz
spx_subset = spx.iloc[:len(jazz)].copy() if len(spx) > len(jazz) else spx.copy()

# Or downsample jazz to match SPX length
jazz_downsampled = jazz.iloc[::int(len(jazz)/len(spx))].copy() if len(jazz) > len(spx) else jazz.copy()
jazz_downsampled = jazz_downsampled.iloc[:len(spx)].copy()

print(f"\nResampled lengths:")
print(f"  Jazz: {len(jazz_downsampled)} points")
print(f"  SPX: {len(spx)} points")

# Calculate correlations
correlations = {
    'Energy vs Volume': stats.pearsonr(jazz_downsampled['rms_energy_norm'].values, 
                                       spx['volume_norm'].values[:len(jazz_downsampled)])[0],
    'Centroid vs Price Range': stats.pearsonr(jazz_downsampled['spectral_centroid_norm'].values,
                                              spx['price_range_norm'].values[:len(jazz_downsampled)])[0],
    'Onset vs Momentum': stats.pearsonr(jazz_downsampled['onset_strength_norm'].values,
                                       spx['momentum_norm'].values[:len(jazz_downsampled)])[0],
    'Flux vs Volatility': stats.pearsonr(jazz_downsampled['spectral_flux_norm'].values,
                                        spx['volatility_norm'].values[:len(jazz_downsampled)])[0],
}

print("\nCorrelation coefficients:")
for pair, corr in correlations.items():
    print(f"  {pair}: {corr:.4f}")

# =================================================================
# SAVE RESULTS
# =================================================================
print("\n" + "="*70)
print("💾 SAVING RESULTS")
print("="*70)

# Save processed SPX data
spx.to_csv('/home/claude/spx_processed.csv')
print("✅ Saved: spx_processed.csv")

# Save comparison summary
summary = pd.DataFrame({
    'Metric': list(correlations.keys()),
    'Correlation': list(correlations.values())
})
summary.to_csv('/home/claude/jazz_spx_correlations.csv', index=False)
print("✅ Saved: jazz_spx_correlations.csv")

# Save aligned data for visualization
aligned = pd.DataFrame({
    'jazz_energy': jazz_downsampled['rms_energy_norm'].values,
    'spx_volume': spx['volume_norm'].values[:len(jazz_downsampled)],
    'jazz_centroid': jazz_downsampled['spectral_centroid_norm'].values,
    'spx_price_range': spx['price_range_norm'].values[:len(jazz_downsampled)],
    'jazz_onset': jazz_downsampled['onset_strength_norm'].values,
    'spx_momentum': spx['momentum_norm'].values[:len(jazz_downsampled)],
})
aligned.to_csv('/home/claude/jazz_spx_aligned.csv', index=False)
print("✅ Saved: jazz_spx_aligned.csv")

print("\n" + "="*70)
print("✨ ANALYSIS COMPLETE")
print("="*70)
print("\nKey findings:")
print(f"1. Jazz has {jazz_num_cycles:.0f} energy cycles in 3.15 minutes")
print(f"2. SPX volume shows {'cycles' if spx_volume_cycle else 'no clear cycles'}")
print(f"3. Strongest correlation: {max(correlations, key=correlations.get)} ({max(correlations.values()):.4f})")
print(f"4. Weakest correlation: {min(correlations, key=correlations.get)} ({min(correlations.values()):.4f})")
