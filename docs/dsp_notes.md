# DSP Notes

## Initial conventions

- Preserve the original WAV files in `data/raw/`.
- Convert audio to mono using a channel mean when a recording has multiple channels.
- Use zero-phase SOS filtering (`scipy.signal.sosfiltfilt`) for offline analysis to avoid phase distortion.
- Validate that cutoff frequencies are below the Nyquist frequency before designing a filter.

Filter cutoff choices should be justified against the observed spectrum and documented here as the dataset evolves.
