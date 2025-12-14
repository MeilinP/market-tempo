"""
Jazz Feature Extraction with PATTERN Detection - FIXED VERSION
Works with all scipy versions!
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def detect_cycles(time_series, sr, hop_length):
    """
    Detect periodic patterns using autocorrelation
    """
    # Use autocorrelation to find periodicity
    autocorr = np.correlate(time_series, time_series, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Find peaks
    peaks, properties = signal.find_peaks(autocorr, height=autocorr.max()*0.3, distance=10)
    
    if len(peaks) > 0:
        primary_cycle_frames = peaks[0]
        primary_cycle_seconds = primary_cycle_frames * hop_length / sr
        num_cycles = len(time_series) / primary_cycle_frames if primary_cycle_frames > 0 else 0
        periodicity_strength = properties['peak_heights'][0] / autocorr.max()
    else:
        primary_cycle_frames = 0
        primary_cycle_seconds = 0
        num_cycles = 0
        periodicity_strength = 0
    
    return {
        'cycle_length_frames': int(primary_cycle_frames),
        'cycle_length_seconds': float(primary_cycle_seconds),
        'num_cycles': float(num_cycles),
        'periodicity_strength': float(periodicity_strength)
    }


def extract_features_with_patterns(audio_file):
    """
    Extract features without using beat_track (which has the scipy bug)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {audio_file.name}")
    print(f"{'='*60}")
    
    # Load audio
    y, sr = librosa.load(str(audio_file), sr=22050)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Parameters
    frame_length = 2048
    hop_length = 512
    
    # =================================================================
    # CORE FEATURES
    # =================================================================
    print("\n[1/6] Extracting RMS Energy...")
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    print("[2/6] Extracting Spectral Centroid...")
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    
    print("[3/6] Extracting Tempo (alternative method)...")
    # Use tempo without beat_track to avoid scipy.signal.hann bug
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)[0]
    
    # Estimate number of beats from tempo and duration
    num_beats = int((tempo / 60.0) * duration)
    
    # =================================================================
    # PATTERN/CYCLE FEATURES
    # =================================================================
    print("[4/6] Extracting Spectral Flux...")
    spec_flux = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, hop_length=hop_length)
    
    print("[5/6] Detecting Energy Cycles...")
    energy_cycles = detect_cycles(rms, sr, hop_length)
    
    print("[6/6] Detecting Beat Patterns...")
    beat_cycles = detect_cycles(onset_env, sr, hop_length)
    
    # Create time grid
    n_frames = len(rms)
    time_grid = np.linspace(0, duration, n_frames)
    
    # Ensure all arrays same length
    centroid = np.interp(time_grid, np.linspace(0, duration, len(centroid)), centroid)
    onset_env = np.interp(time_grid, np.linspace(0, duration, len(onset_env)), onset_env)
    spec_flux = np.interp(time_grid, np.linspace(0, duration, len(spec_flux)), spec_flux)
    
    # =================================================================
    # TIME-SERIES DATAFRAME
    # =================================================================
    time_series_df = pd.DataFrame({
        'time': time_grid,
        'rms_energy': rms,
        'spectral_centroid': centroid,
        'onset_strength': onset_env,
        'spectral_flux': spec_flux,
    })
    
    # Normalize all features to 0-1
    for col in ['rms_energy', 'spectral_centroid', 'onset_strength', 'spectral_flux']:
        if time_series_df[col].std() > 0:
            time_series_df[col + '_norm'] = (
                (time_series_df[col] - time_series_df[col].min()) / 
                (time_series_df[col].max() - time_series_df[col].min())
            )
    
    # =================================================================
    # SUMMARY STATISTICS
    # =================================================================
    summary = {
        'filename': audio_file.name,
        'duration_seconds': float(duration),
        'duration_minutes': float(duration / 60),
        
        # Tempo/Rhythm
        'tempo_bpm': float(tempo),
        'num_beats_estimated': num_beats,
        'avg_beat_interval': float(60.0 / tempo),
        
        # RMS Energy stats
        'rms_mean': float(rms.mean()),
        'rms_std': float(rms.std()),
        'rms_max': float(rms.max()),
        'rms_range': float(rms.max() - rms.min()),
        
        # Spectral Centroid stats
        'centroid_mean_hz': float(centroid.mean()),
        'centroid_std_hz': float(centroid.std()),
        'centroid_range_hz': float(centroid.max() - centroid.min()),
        
        # Pattern/Cycle stats - ENERGY
        'energy_cycle_length_sec': energy_cycles['cycle_length_seconds'],
        'energy_num_cycles': energy_cycles['num_cycles'],
        'energy_periodicity_strength': energy_cycles['periodicity_strength'],
        
        # Pattern/Cycle stats - BEATS
        'beat_cycle_length_sec': beat_cycles['cycle_length_seconds'],
        'beat_num_cycles': beat_cycles['num_cycles'],
        'beat_periodicity_strength': beat_cycles['periodicity_strength'],
        
        # Spectral flux stats
        'spectral_flux_mean': float(spec_flux.mean()),
        'spectral_flux_std': float(spec_flux.std()),
    }
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Tempo: {tempo:.1f} BPM")
    print(f"  Avg Energy: {rms.mean():.4f}")
    print(f"  Energy Cycle: {energy_cycles['cycle_length_seconds']:.2f} sec ({energy_cycles['num_cycles']:.1f} cycles)")
    print(f"  Beat Pattern: {beat_cycles['cycle_length_seconds']:.2f} sec ({beat_cycles['num_cycles']:.1f} cycles)")
    print("="*60)
    
    return time_series_df, summary


def main():
    AUDIO_DIR = '/Users/meilinpan/Desktop/market-tempo'
    OUTPUT_DIR = '/Users/meilinpan/Desktop/market-tempo/jazz_with_patterns'
    
    print("\n" + "🎺"*30)
    print("JAZZ FEATURE EXTRACTION - FIXED VERSION")
    print("Core Features + Pattern Detection")
    print("🎺"*30)
    
    audio_dir = Path(AUDIO_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    mp3_files = sorted(audio_dir.glob('*.mp3'))
    print(f"\n🎵 Found {len(mp3_files)} jazz recordings")
    
    all_summaries = []
    all_time_series = {}
    
    for mp3_file in mp3_files:
        try:
            ts_df, summary = extract_features_with_patterns(mp3_file)
            all_summaries.append(summary)
            all_time_series[mp3_file.stem] = ts_df
            print("✅ Success!")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_summaries) == 0:
        print("\n❌ No files processed successfully!")
        return
    
    # Save results
    summary_df = pd.DataFrame(all_summaries)
    
    # 1. Summary
    summary_file = output_dir / 'jazz_summary_with_patterns.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✅ Summary saved: {summary_file}")
    
    # 2. Combined time-series
    all_ts = []
    for song, df in all_time_series.items():
        df_copy = df.copy()
        df_copy['song'] = song
        all_ts.append(df_copy)
    
    combined_ts = pd.concat(all_ts, ignore_index=True)
    time_series_file = output_dir / 'jazz_timeseries_with_patterns.csv'
    combined_ts.to_csv(time_series_file, index=False)
    print(f"✅ Time-series saved: {time_series_file}")
    
    # 3. Individual files
    ts_dir = output_dir / 'by_song'
    ts_dir.mkdir(exist_ok=True)
    for song, df in all_time_series.items():
        df.to_csv(ts_dir / f'{song}.csv', index=False)
    print(f"✅ Individual files: {ts_dir}/")
    
    print("\n" + "="*60)
    print(f"✨ SUCCESS! Processed {len(summary_df)} songs")
    print("="*60)
    
    if len(summary_df) > 0:
        print("\nQUICK OVERVIEW:")
        display_cols = ['filename', 'tempo_bpm', 'energy_num_cycles', 'beat_num_cycles']
        print(summary_df[display_cols].to_string(index=False))
        
        print("\nPATTERN STATISTICS:")
        pattern_cols = ['filename', 'energy_cycle_length_sec', 'beat_cycle_length_sec']
        print(summary_df[pattern_cols].to_string(index=False))
    
    print("\n" + "="*60)
    print("📤 UPLOAD TO CLAUDE:")
    print("="*60)
    print(f"1. {summary_file.name}")
    print(f"2. {time_series_file.name}")
    print("\n🎯 Ready for SPX comparison!")


if __name__ == '__main__':
    main()
