//! Decode an inline PCM16 WAV and deliver mono PCM at the engine's reference rate.
//! Check duration at the source rate, then compensate for resampler delay so the
//! normalized clip spans the same interval as the input.

use std::fmt::Display;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use omni_engine::Invalid;
use omni_engine::ReferenceAudioInfo;
use omni_engine::ReferencePcm;
use rubato::Fft;
use rubato::FixedSync;
use rubato::Resampler as _;
use rubato::audioadapter_buffers::direct::InterleavedSlice;

const WAV_PREFIX: &str = "data:audio/wav;base64,";
const PCM_SUBFORMAT: [u8; 16] =
    [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71];

struct PcmFormat {
    channels: usize,
    sample_rate: u32,
}

pub(crate) fn decode(uri: &str, info: &ReferenceAudioInfo) -> Result<ReferencePcm, Invalid> {
    let encoded = uri.strip_prefix(WAV_PREFIX).ok_or_else(|| invalid(format!("expected the `{WAV_PREFIX}` prefix")))?;
    let wav = STANDARD.decode(encoded).map_err(|err| invalid(format!("invalid base64 ({err})")))?;
    let (source, spec) = wav_pcm(&wav)?;
    let channels = spec.channels;
    let frames = source.len() / (channels * 2);
    let sample_rate = info.sample_rate;
    info.check_duration(frames, spec.sample_rate)?;

    let pcm = if channels == 1 && spec.sample_rate == sample_rate {
        Bytes::copy_from_slice(source)
    } else {
        let samples = source.as_chunks::<2>().0.chunks_exact(channels).map(|frame| {
            let sum: f32 = frame.iter().map(|&sample| f32::from(i16::from_le_bytes(sample))).sum();
            sum / channels as f32
        });
        if spec.sample_rate == sample_rate {
            samples.flat_map(quantize).collect()
        } else {
            resample(samples.collect(), spec.sample_rate, sample_rate)?
        }
    };
    Ok(ReferencePcm { pcm, sample_rate })
}

fn invalid(message: impl Display) -> Invalid {
    Invalid { param: "ref_audio", message: format!("invalid `ref_audio` ({message})") }
}

fn wav_pcm(wav: &[u8]) -> Result<(&[u8], PcmFormat), Invalid> {
    let (header, mut chunks) = wav.split_first_chunk::<12>().ok_or_else(|| invalid("truncated RIFF header"))?;
    if &header[..4] != b"RIFF" || &header[8..] != b"WAVE" {
        return Err(invalid("expected a RIFF WAVE header"));
    }
    let declared_len = u64::from(u32::from_le_bytes(header[4..8].try_into().expect("RIFF length field"))) + 8;
    if declared_len != wav.len() as u64 {
        return Err(invalid(format!("RIFF declares {declared_len} bytes, got {}", wav.len())));
    }
    // Bound every chunk against the actual RIFF buffer, including pads and chunks after data.
    let mut format = None;
    let mut data = None;
    while !chunks.is_empty() {
        let (header, rest) = chunks.split_first_chunk::<8>().ok_or_else(|| invalid("truncated WAV chunk header"))?;
        let size = u32::from_le_bytes(*header.last_chunk::<4>().expect("chunk length field")) as usize;
        let (payload, tail) = rest.split_at_checked(size).ok_or_else(|| invalid("truncated WAV chunk payload"))?;
        match &header[..4] {
            b"fmt " if format.is_none() => format = Some(payload),
            b"data" if format.is_some() && data.is_none() => data = Some(payload),
            b"fmt " | b"data" => return Err(invalid("expected one fmt chunk followed by one data chunk")),
            _ => {}
        }
        chunks = tail.get(size % 2..).ok_or_else(|| invalid("missing WAV chunk pad byte"))?;
    }

    let format = format.ok_or_else(|| invalid("missing WAV fmt chunk"))?;
    let pcm = data.ok_or_else(|| invalid("missing WAV data chunk"))?;
    let spec = pcm_format(format)?;
    let frame_bytes = spec.channels * 2;
    if !pcm.len().is_multiple_of(frame_bytes) {
        return Err(invalid(format!("expected {frame_bytes} bytes per frame, got {} bytes of PCM", pcm.len())));
    }
    Ok((pcm, spec))
}

fn pcm_format(format: &[u8]) -> Result<PcmFormat, Invalid> {
    let (fields, extension) = format.split_first_chunk::<16>().ok_or_else(|| invalid("truncated WAV fmt chunk"))?;
    let u16_at = |offset| u16::from_le_bytes(fields[offset..offset + 2].try_into().expect("WAV fmt field"));
    let u32_at = |offset| u32::from_le_bytes(fields[offset..offset + 4].try_into().expect("WAV fmt field"));
    match u16_at(0) {
        // PCM also occurs in the longer format headers, where cbSize is ignored for tag 1.
        1 if matches!(format.len(), 16 | 18 | 40) => {}
        1 => return Err(invalid(format!("expected 16, 18 or 40 bytes in PCM fmt, got {}", format.len()))),
        0xfffe => {
            let extension: &[u8; 24] = extension
                .try_into()
                .map_err(|_| invalid(format!("expected 40 bytes in extensible WAV fmt, got {}", format.len())))?;
            let size = u16::from_le_bytes([extension[0], extension[1]]);
            if size != 22 {
                return Err(invalid(format!("expected extensible WAV cbSize of 22, got {size}")));
            }
            let valid_bits = u16::from_le_bytes([extension[2], extension[3]]);
            // Some encoders leave valid bits unset and use the container width.
            if !matches!(valid_bits, 0 | 16) {
                return Err(invalid(format!("expected 16 valid bits per sample, got {valid_bits}")));
            }
            if extension[8..] != PCM_SUBFORMAT {
                return Err(invalid(format!("expected the PCM subformat GUID, got {:02x?}", &extension[8..])));
            }
        }
        tag => return Err(invalid(format!("expected PCM or extensible WAV fmt, got format tag {tag:#06x}"))),
    }

    let channels = u16_at(2);
    let sample_rate = u32_at(4);
    let byte_rate = u32_at(8);
    let align = u16_at(12);
    let bits = u16_at(14);
    if !matches!(channels, 1 | 2) || bits != 16 {
        return Err(invalid(format!(
            "expected mono or stereo PCM16 WAV, got {bits}-bit WAV with {channels} channel(s)"
        )));
    }
    let frame_bytes = channels * 2;
    if align != frame_bytes {
        return Err(invalid(format!("expected {frame_bytes} bytes per frame, got {align}")));
    }
    if !matches!(sample_rate, 16_000 | 24_000 | 44_100 | 48_000) {
        return Err(invalid(format!("expected WAV at 16000, 24000, 44100 or 48000 Hz, got {sample_rate} Hz")));
    }
    let expected_byte_rate = sample_rate * u32::from(align);
    if byte_rate != expected_byte_rate {
        return Err(invalid(format!("expected {expected_byte_rate} bytes per second, got {byte_rate}")));
    }
    Ok(PcmFormat { channels: usize::from(channels), sample_rate })
}

fn quantize(sample: f32) -> [u8; 2] {
    (sample.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16).to_le_bytes()
}

fn resample(mut samples: Vec<f32>, from: u32, to: u32) -> Result<Bytes, Invalid> {
    let mut fft = Fft::new(from as usize, to as usize, from as usize / 100, 1, FixedSync::Both).map_err(invalid)?;
    let take = fft.input_frames_next();
    let give = fft.output_frames_next();
    let delay = fft.output_delay();
    let frames = (samples.len() as u64 * u64::from(to)).div_ceil(u64::from(from)) as usize;
    let end = delay + frames;
    // Run whole blocks, padding with enough zeros to flush the filter's tail past `end`.
    let blocks = end.div_ceil(give);
    samples.resize(blocks * take, 0.0);
    let mut output = vec![0.0; blocks * give];
    for (input, output) in samples.chunks_exact(take).zip(output.chunks_exact_mut(give)) {
        let input = InterleavedSlice::new(input, 1, take).expect("a mono chunk");
        let mut output = InterleavedSlice::new_mut(output, 1, give).expect("a mono chunk");
        fft.process_into_buffer(&input, &mut output, None).map_err(invalid)?;
    }
    Ok(output[delay..end].iter().copied().flat_map(quantize).collect())
}
