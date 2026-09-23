from pathlib import Path
from src.dsp.filters import load_and_preprocess_audio, compute_shannon_energy_envelope

# Grab the first available .wav file in training-a
sample_file = next(Path('data/raw').glob('**/*.wav'), None)

if sample_file is None:
    print('No .wav files found! Check data/raw directory.')
else:
    print(f'Testing on sample: {sample_file}')
    raw, filtered, sr = load_and_preprocess_audio(str(sample_file))
    envelope = compute_shannon_energy_envelope(filtered, fs=sr)
    
    print(f'Sample rate: {sr} Hz')
    print(f'Duration: {len(filtered) / sr:.2f} seconds')
    print(f'Filtered signal range: [{filtered.min():.3f}, {filtered.max():.3f}]')
    print(f'Envelope range: [{envelope.min():.3f}, {envelope.max():.3f}]')
    print('DSP pipeline verification successful!')
