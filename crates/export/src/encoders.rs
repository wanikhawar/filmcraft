use super::convert::{
    Matrix, Planes444, rgba_to_yuv420_8, rgba_to_yuv420_10, rgba_to_yuv422_10, rgbf_to_yuv420_8, rgbf_to_yuv420_10, rgbf_to_yuv422_10, to_yuv444_10,
};
use super::format::{ColorSignal, DEFAULT_CRF, ExportSettings, Format, H264Pass};
use super::registry::{AudioEncoder, EncodedPacket, EncoderFrame, PrepareAhead, VideoEncoder};
use super::stats::{Stage, timed};
use super::{BitrateMode, ExportError, FrameRate, H264Profile, Result, SampleEntry, SampleFormat};
use std::sync::{Arc, Mutex, PoisonError};

struct MjpegEncoder {
    w: u16,
    h: u16,
    quality: u8,
    rate: FrameRate,
}

impl VideoEncoder for MjpegEncoder {
    fn sample_entry(&self) -> SampleEntry {
        SampleEntry::jpeg(self.w, self.h)
    }
    fn timescale(&self) -> u32 {
        self.rate.num as u32
    }
    fn encode(&mut self, f: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
        let rgb: Vec<u8> = f.rgba.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        let mut out = Vec::new();
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, self.quality);
        enc.encode(&rgb, f.width, f.height, image::ExtendedColorType::Rgb8).map_err(|e| ExportError::Encode(e.to_string()))?;
        Ok(vec![EncodedPacket { data: out, key: true, duration: self.rate.den as u32, composition_offset: 0 }])
    }
    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        Ok(Vec::new())
    }
}

pub(crate) fn mjpeg_factory(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Option<Result<Box<dyn VideoEncoder>>> {
    (format == Format::Mjpeg).then(|| Ok(Box::new(MjpegEncoder { w: w as u16, h: h as u16, quality: s.quality.clamp(1, 100), rate }) as Box<dyn VideoEncoder>))
}

/// AAC-LC (our encoder), 320 kbps stereo by default.
struct AacEncoder {
    enc: filmcraft_aac::Encoder,
    rate: u32,
    channels: u32,
}

impl AudioEncoder for AacEncoder {
    fn sample_entry(&self) -> SampleEntry {
        SampleEntry::aac(self.enc.audio_specific_config(), self.channels, self.rate)
    }
    fn encode(&mut self, planar: &[Vec<f32>]) -> Result<Vec<Vec<u8>>> {
        let refs: Vec<&[f32]> = if planar.len() == 6 {
            // ours: L, R, C, LFE, Ls, Rs → AAC channel configuration 6: C, L, R, Ls, Rs, LFE
            [2usize, 0, 1, 4, 5, 3].iter().map(|&c| planar[c].as_slice()).collect()
        } else {
            planar.iter().map(Vec::as_slice).collect()
        };
        Ok(self.enc.encode(&refs))
    }
    fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        Ok(self.enc.flush())
    }
    fn priming(&self) -> u32 {
        self.enc.priming_samples()
    }
    fn frame_size(&self) -> u32 {
        1024
    }
}

/// FLAC (our encoder) in MP4 / QuickTime: one frame of 4096 samples per access unit.
pub(crate) struct FlacAudio {
    enc: filmcraft_flac::Encoder,
    format: SampleFormat,
    /// Samples of the frame `flush` returns (shorter than a full block).
    tail: u32,
}

impl FlacAudio {
    pub(crate) fn new(sample_rate: u32, channels: u32, s: &ExportSettings) -> Result<Self> {
        let format = s.audio.sample_format();
        let ch = u8::try_from(channels).map_err(|_| ExportError::Unsupported("FLAC: too many audio channels".into()))?;
        let mut cfg = filmcraft_flac::EncoderConfig::new(sample_rate, ch, format.bits() as u8);
        cfg.level = s.audio.flac_level.min(8);
        let enc = filmcraft_flac::Encoder::new(cfg).map_err(|e| ExportError::Unsupported(format!("FLAC: {e}")))?;
        Ok(FlacAudio { enc, format, tail: 0 })
    }
}

impl AudioEncoder for FlacAudio {
    fn sample_entry(&self) -> SampleEntry {
        SampleEntry::flac(&self.enc.streaminfo())
    }
    fn encode(&mut self, planar: &[Vec<f32>]) -> Result<Vec<Vec<u8>>> {
        let ints: Vec<Vec<i32>> = planar.iter().map(|c| c.iter().map(|&s| self.format.quantise(s)).collect()).collect();
        let refs: Vec<&[i32]> = ints.iter().map(Vec::as_slice).collect();
        self.enc.encode_frames(&refs).map_err(|e| ExportError::Encode(format!("FLAC: {e}")))
    }
    fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        Ok(self.flush_timed()?.into_iter().map(|(au, _)| au).collect())
    }
    fn flush_timed(&mut self) -> Result<Vec<(Vec<u8>, u32)>> {
        self.tail = self.enc.pending_samples() as u32;
        let last = self.enc.finish();
        Ok(if last.is_empty() { Vec::new() } else { vec![(last, self.tail)] })
    }
    fn priming(&self) -> u32 {
        0
    }
    fn frame_size(&self) -> u32 {
        4096
    }
    fn final_sample_entry(&self) -> Option<SampleEntry> {
        Some(SampleEntry::flac(&self.enc.streaminfo()))
    }
}

pub(crate) fn aac_factory(_format: Format, sample_rate: u32, channels: u32, s: &ExportSettings) -> Option<Result<Box<dyn AudioEncoder>>> {
    // AAC caps a frame at 6144 bits per channel (ISO/IEC 14496-3 §4.5.3.2): 6 bits per sample, so
    // 264.6 kbps for stereo at 22.05 kHz; a higher setting is capped instead of refused
    let max = (6 * sample_rate as u64 * channels.max(1) as u64).min(u32::MAX as u64) as u32;
    let bps = (s.audio.bitrate_kbps.clamp(32, 512) * 1000).min(max);
    Some(
        filmcraft_aac::Encoder::new(filmcraft_aac::EncoderConfig::cbr(sample_rate, channels as usize, bps))
            .map(|enc| Box::new(AacEncoder { enc, rate: sample_rate, channels }) as Box<dyn AudioEncoder>)
            .map_err(|e| ExportError::Encode(e.to_string())),
    )
}

/// ProRes 422 encoder (HQ unless the settings pick another flavour): sRGB/709 RGBA8 → 10-bit limited-range BT.709 4:2:2.
struct ProResEncoder {
    enc: filmcraft_prores::Encoder,
    profile: filmcraft_prores::Profile,
    /// Code the frames' alpha channel (4444 profiles with Include Alpha Channel).
    alpha: bool,
    w: u32,
    h: u32,
    rate: FrameRate,
    signal: ColorSignal,
    /// The picture converted into, kept between frames: every conversion overwrites all of its samples,
    /// so only a change of size, chroma or alpha needs a new one.
    frame: Option<filmcraft_prores::Frame>,
}

impl VideoEncoder for ProResEncoder {
    fn sample_entry(&self) -> SampleEntry {
        let mut e = SampleEntry::prores(filmcraft_isobmff::FourCc(self.profile.fourcc()), self.w as u16, self.h as u16);
        self.signal.apply_to(&mut e, true);
        e
    }
    fn timescale(&self) -> u32 {
        self.rate.num as u32
    }
    fn encode(&mut self, f: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
        whole_picture(f)?;
        let chroma = self.profile.chroma();
        let wanted_alpha = self.alpha;
        let mut fr = match self.frame.take() {
            Some(fr) if fr.width == f.width && fr.height == f.height && fr.chroma == chroma && fr.alpha.is_some() == wanted_alpha => fr,
            _ => filmcraft_prores::Frame::new(f.width, f.height, chroma, 10, wanted_alpha),
        };
        if f.hdr.is_some()
            && let Some(a) = fr.alpha.as_mut()
        {
            // HDR input carries no alpha and `to_yuv444_10` leaves the plane alone for it: a fresh frame is
            // opaque, but a reused one would otherwise keep the previous frame's alpha
            a.fill(1023);
        }
        let (w, h) = (f.width as usize, f.height as usize);
        timed(Stage::Convert, || match (chroma, f.hdr) {
            (filmcraft_prores::ChromaFormat::Yuv444, hdr) => {
                let (kr, kb) = if hdr.is_some() { self.signal.kr_kb() } else { (0.2126, 0.0722) };
                to_yuv444_10(f.rgba, hdr, w, h, Matrix::new(kr, kb), Planes444 { y: &mut fr.y, cb: &mut fr.cb, cr: &mut fr.cr, alpha: fr.alpha.as_deref_mut() })
            }
            (_, Some(rgb)) => {
                let (kr, kb) = self.signal.kr_kb();
                rgbf_to_yuv422_10(rgb, w, h, Matrix::new(kr, kb), &mut fr.y, &mut fr.cb, &mut fr.cr)
            }
            (_, None) => rgba_to_yuv422_10(f.rgba, w, h, &mut fr.y, &mut fr.cb, &mut fr.cr),
        });
        let encoded = self.enc.encode(&fr);
        self.frame = Some(fr);
        let data = encoded.map_err(|e| ExportError::Encode(e.to_string()))?;
        Ok(vec![EncodedPacket { data, key: true, duration: self.rate.den as u32, composition_offset: 0 }])
    }
    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        Ok(Vec::new())
    }
}

/// Refuse a picture whose pixels (the float pixels of an HDR frame, else the RGBA8) do not cover
/// its size. The ProRes and DNxHR encoders keep their frame between pictures and the 4:2:2
/// conversions skip rows the source does not cover, so a short source would otherwise be coded
/// with the previous picture's rows.
fn whole_picture(f: &EncoderFrame) -> Result<()> {
    let px = (f.width as usize).checked_mul(f.height as usize);
    let covered = match f.hdr {
        Some(rgb) => px.and_then(|n| n.checked_mul(3)).is_some_and(|n| rgb.len() >= n),
        None => px.and_then(|n| n.checked_mul(4)).is_some_and(|n| f.rgba.len() >= n),
    };
    if covered { Ok(()) } else { Err(ExportError::Encode(format!("a {}x{} picture arrived without all of its pixels", f.width, f.height))) }
}

/// The ProRes profile named by [`ExportSettings::prores_profile`].
pub fn prores_profile(name: &str) -> filmcraft_prores::Profile {
    use filmcraft_prores::Profile;
    match name.to_ascii_lowercase().as_str() {
        "proxy" => Profile::Proxy,
        "lt" => Profile::Lt,
        "standard" | "422" => Profile::Standard,
        "4444" => Profile::P4444,
        "4444xq" | "4444 xq" | "4444-xq" | "xq" => Profile::P4444Xq,
        _ => Profile::Hq,
    }
}

fn prores_encoder(w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> ProResEncoder {
    let profile = prores_profile(&s.prores_profile);
    let mut cfg = filmcraft_prores::EncoderConfig::new(profile, w, h);
    if s.signal.is_hdr() {
        cfg.color = filmcraft_prores::ColorInfo { primaries: s.signal.primaries, transfer: s.signal.transfer, matrix: s.signal.matrix };
    }
    let alpha = s.keeps_alpha();
    cfg.encode_alpha = alpha;
    let enc = filmcraft_prores::Encoder::with_config(cfg);
    ProResEncoder { enc, profile, alpha, w, h, rate, signal: s.signal, frame: None }
}

pub(crate) fn prores_factory(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Option<Result<Box<dyn VideoEncoder>>> {
    (format == Format::ProRes).then(|| Ok(Box::new(prores_encoder(w, h, rate, s)) as Box<dyn VideoEncoder>))
}

/// The DNxHR profile named by [`ExportSettings::dnx_profile`].
pub fn dnx_profile(name: &str) -> filmcraft_dnx::Profile {
    use filmcraft_dnx::Profile;
    match name.to_ascii_lowercase().as_str() {
        "lb" => Profile::Lb,
        "sq" => Profile::Sq,
        "hqx" => Profile::Hqx,
        _ => Profile::Hq,
    }
}

/// DNxHR encoder (RGBA8 or HDR floats → BT.709 / BT.2020 limited-range 4:2:2; 8-bit for
/// LB/SQ/HQ, 10-bit for HQX).
struct DnxEncoder {
    enc: filmcraft_dnx::Encoder,
    w: u32,
    h: u32,
    rate: FrameRate,
    signal: ColorSignal,
    /// The picture converted into, kept between frames: the conversions overwrite every sample.
    frame: Option<filmcraft_dnx::Frame>,
}

impl VideoEncoder for DnxEncoder {
    fn sample_entry(&self) -> SampleEntry {
        let mut e = SampleEntry::dnx(filmcraft_isobmff::FourCc(*b"AVdh"), self.w as u16, self.h as u16);
        self.signal.apply_to(&mut e, true);
        e
    }
    fn timescale(&self) -> u32 {
        self.rate.num as u32
    }
    fn encode(&mut self, f: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
        whole_picture(f)?;
        let chroma = filmcraft_dnx::ChromaFormat::Yuv422;
        let mut fr = match self.frame.take() {
            Some(fr) if fr.width == f.width && fr.height == f.height && fr.chroma == chroma && fr.alpha.is_none() => fr,
            _ => filmcraft_dnx::Frame::new(f.width, f.height, chroma, 10, false),
        };
        match f.hdr {
            Some(rgb) => {
                let (kr, kb) = self.signal.kr_kb();
                rgbf_to_yuv422_10(rgb, f.width as usize, f.height as usize, Matrix::new(kr, kb), &mut fr.y, &mut fr.cb, &mut fr.cr)
            }
            None => rgba_to_yuv422_10(f.rgba, f.width as usize, f.height as usize, &mut fr.y, &mut fr.cb, &mut fr.cr),
        }
        // the encoder rescales 10-bit input to its coded depth
        let encoded = self.enc.encode(&fr);
        self.frame = Some(fr);
        let data = encoded.map_err(|e| ExportError::Encode(e.to_string()))?;
        Ok(vec![EncodedPacket { data, key: true, duration: self.rate.den as u32, composition_offset: 0 }])
    }
    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        Ok(Vec::new())
    }
}

fn dnx_encoder(w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Result<DnxEncoder> {
    let mut cfg = filmcraft_dnx::EncoderConfig::new(dnx_profile(&s.dnx_profile), w, h);
    if s.signal.primaries == 9 {
        cfg.color_volume = filmcraft_dnx::ColorVolume::Bt2020Ncl;
    }
    let enc = filmcraft_dnx::Encoder::with_config(cfg).map_err(|e| ExportError::Encode(e.to_string()))?;
    Ok(DnxEncoder { enc, w, h, rate, signal: s.signal, frame: None })
}

pub(crate) fn dnx_factory(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Option<Result<Box<dyn VideoEncoder>>> {
    (format == Format::DnxHr).then(|| dnx_encoder(w, h, rate, s).map(|e| Box::new(e) as Box<dyn VideoEncoder>))
}

/// The APV profile named by [`ExportSettings::apv_profile`].
pub fn apv_profile(name: &str) -> filmcraft_apv::Profile {
    use filmcraft_apv::Profile;
    match name.to_ascii_lowercase().replace([' ', '_'], "-").trim_start_matches("apv-").trim_start_matches("apv") {
        "422-12" | "42212" => Profile::P422_12,
        "444-10" | "44410" => Profile::P444_10,
        "444-12" | "44412" => Profile::P444_12,
        "4444-10" | "444410" => Profile::P4444_10,
        "4444-12" | "444412" => Profile::P4444_12,
        "400-10" | "40010" => Profile::P400_10,
        _ => Profile::P422_10,
    }
}

/// APV encoder (RFC 9924: 10/12-bit 4:2:2 or 4:4:4 intra).
struct ApvEncoder {
    enc: filmcraft_apv::Encoder,
    chroma: filmcraft_apv::ChromaFormat,
    w: u32,
    h: u32,
    rate: FrameRate,
    signal: ColorSignal,
}

impl VideoEncoder for ApvEncoder {
    fn sample_entry(&self) -> SampleEntry {
        let apvc = filmcraft_isobmff::ApvConfig::parse(&self.enc.decoder_config_record()).unwrap_or_default();
        let mut e = SampleEntry::apv(apvc, self.w as u16, self.h as u16);
        self.signal.apply_to(&mut e, true);
        e
    }
    fn timescale(&self) -> u32 {
        self.rate.num as u32
    }
    fn encode(&mut self, f: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
        let (w, h) = (f.width as usize, f.height as usize);
        let mut fr = filmcraft_apv::Frame::new(f.width, f.height, self.chroma, 10, self.chroma == filmcraft_apv::ChromaFormat::Yuv4444);
        let mut y422 = vec![0u16; w * h];
        let cw = w.div_ceil(2);
        let mut cb422 = vec![0u16; cw * h];
        let mut cr422 = vec![0u16; cw * h];
        match f.hdr {
            Some(rgb) => {
                let (kr, kb) = self.signal.kr_kb();
                rgbf_to_yuv422_10(rgb, w, h, Matrix::new(kr, kb), &mut y422, &mut cb422, &mut cr422);
            }
            None => rgba_to_yuv422_10(f.rgba, w, h, &mut y422, &mut cb422, &mut cr422),
        }
        fr.y = y422;
        match self.chroma {
            filmcraft_apv::ChromaFormat::Monochrome => {
                fr.cb.clear();
                fr.cr.clear();
            }
            filmcraft_apv::ChromaFormat::Yuv422 => {
                fr.cb = cb422;
                fr.cr = cr422;
            }
            filmcraft_apv::ChromaFormat::Yuv444 | filmcraft_apv::ChromaFormat::Yuv4444 => {
                for row in 0..h {
                    for x in 0..w {
                        fr.cb[row * w + x] = cb422[row * cw + x / 2];
                        fr.cr[row * w + x] = cr422[row * cw + x / 2];
                    }
                }
                if let Some(a) = fr.alpha.as_mut() {
                    for (i, dst) in a.iter_mut().enumerate() {
                        *dst = f.rgba.get(i * 4 + 3).map_or(1023, |&v| ((v as u32 * 1023 + 127) / 255) as u16);
                    }
                }
            }
        }
        let data = self.enc.encode_raw_au(&fr).map_err(|e| ExportError::Encode(e.to_string()))?;
        Ok(vec![EncodedPacket { data, key: true, duration: self.rate.den as u32, composition_offset: 0 }])
    }
    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        Ok(Vec::new())
    }
}

pub(crate) fn apv_factory(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Option<Result<Box<dyn VideoEncoder>>> {
    (format == Format::Apv).then(|| {
        let profile = apv_profile(&s.apv_profile);
        let mut cfg = filmcraft_apv::EncoderConfig::new(profile, w, h);
        if s.signal.is_hdr() {
            cfg.color = filmcraft_apv::ColorInfo { primaries: s.signal.primaries, transfer: s.signal.transfer, matrix: s.signal.matrix, full_range: false };
        }
        let chroma = cfg.chroma;
        let enc = filmcraft_apv::Encoder::with_config(cfg).map_err(|e| ExportError::Encode(e.to_string()))?;
        Ok(Box::new(ApvEncoder { enc, chroma, w, h, rate, signal: s.signal }) as Box<dyn VideoEncoder>)
    })
}

/// H.264 (our encoder): sRGB/709 RGBA8 → limited-range BT.709 4:2:0, 8-bit, or 10-bit for High 10
/// (HDR frames from their float pixels), length-prefixed samples with the `avcC` in the sample entry.
struct H264Encoder {
    enc: filmcraft_h264enc::Encoder,
    w: u32,
    h: u32,
    rate: FrameRate,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    /// The 10-bit planes of High 10.
    y16: Vec<u16>,
    u16: Vec<u16>,
    v16: Vec<u16>,
    signal: ColorSignal,
    /// 8-bit 4:2:0 planes the render threads converted ahead ([`VideoEncoder::prepare_ahead`]), by
    /// picture index.
    ready: Arc<Mutex<std::collections::HashMap<u64, Yuv420>>>,
    /// Plane sets no longer in use, for [`VideoEncoder::prepare_ahead`] to convert into again
    /// instead of allocating a set per picture.
    spare: Arc<Mutex<Vec<Yuv420>>>,
}

type Yuv420 = (Vec<u8>, Vec<u8>, Vec<u8>);

/// How many spare plane sets [`H264Encoder`] keeps. The sets in use are bounded by the pictures in
/// flight between the render threads and the encoder; the cap keeps memory bounded if rendering and
/// encoding ever stop pairing up (pictures rendered and never encoded).
const SPARE_PLANES: usize = 8;

/// Keep `planes` for reuse unless the pool is full (or they hold no memory).
fn recycle(spare: &Mutex<Vec<Yuv420>>, planes: Yuv420) {
    if planes.0.capacity() == 0 {
        return;
    }
    let mut spare = spare.lock().unwrap_or_else(PoisonError::into_inner);
    if spare.len() < SPARE_PLANES {
        spare.push(planes);
    }
}

/// The encoder input of `f`: 8-bit limited-range 4:2:0 in the export's colour signal.
fn h264_input(f: &EncoderFrame, signal: ColorSignal, y: &mut Vec<u8>, u: &mut Vec<u8>, v: &mut Vec<u8>) {
    match f.hdr {
        Some(rgb) => {
            let (kr, kb) = signal.kr_kb();
            rgbf_to_yuv420_8(rgb, f.width as usize, f.height as usize, Matrix::new(kr, kb), y, u, v)
        }
        None => rgba_to_yuv420_8(f.rgba, f.width as usize, f.height as usize, y, u, v),
    }
}

/// The High 10 encoder input of `f`: 10-bit limited-range 4:2:0 in the export's colour signal
/// (HDR frames from their float pixels).
fn h264_input10(f: &EncoderFrame, signal: ColorSignal, y: &mut Vec<u16>, u: &mut Vec<u16>, v: &mut Vec<u16>) -> Result<()> {
    let (w, h) = (f.width as usize, f.height as usize);
    match f.hdr {
        Some(rgb) => {
            let (kr, kb) = signal.kr_kb();
            rgbf_to_yuv420_10(rgb, w, h, Matrix::new(kr, kb), y, u, v)
        }
        None => rgba_to_yuv420_10(f.rgba, w, h, y, u, v),
    }
}

impl H264Encoder {
    fn deep(&self) -> bool {
        self.enc.config().bit_depth > 8
    }

    /// Encode `f` at 10 bits (High 10), converting it on the encoder's threads.
    fn encode10(&mut self, f: &EncoderFrame) -> Result<Vec<filmcraft_h264enc::Packet>> {
        let (enc, y, u, v, signal) = (&self.enc, &mut self.y16, &mut self.u16, &mut self.v16, self.signal);
        timed(Stage::Convert, || enc.install(|| h264_input10(f, signal, y, u, v)))?;
        let cw = (f.width as usize).div_ceil(2);
        let frame = filmcraft_h264enc::YuvFrame16 { y: &self.y16, u: &self.u16, v: &self.v16, y_stride: f.width as usize, uv_stride: cw };
        self.enc.encode16(&frame, f.index as i64 * self.rate.den).map_err(|e| ExportError::Encode(e.to_string()))
    }

    fn packets(&self, ps: Vec<filmcraft_h264enc::Packet>) -> Vec<EncodedPacket> {
        ps.into_iter()
            .map(|p| EncodedPacket { data: p.data, key: p.keyframe, duration: self.rate.den as u32, composition_offset: (p.pts - p.dts) as i32 })
            .collect()
    }
}

impl VideoEncoder for H264Encoder {
    fn sample_entry(&self) -> SampleEntry {
        let cfg = filmcraft_isobmff::AvcConfig::parse(&self.enc.avcc()).unwrap_or_else(|_| {
            let (sps, pps) = self.enc.sps_pps();
            filmcraft_isobmff::AvcConfig::new(vec![sps], vec![pps], 4)
        });
        let mut e = SampleEntry::avc(cfg, self.w as u16, self.h as u16);
        self.signal.apply_to(&mut e, false);
        e
    }
    fn timescale(&self) -> u32 {
        self.rate.num as u32
    }
    fn encode(&mut self, f: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
        if self.deep() {
            let ps = self.encode10(f)?;
            return Ok(self.packets(ps));
        }
        let (ready, stale) = {
            let mut ready = self.ready.lock().unwrap_or_else(PoisonError::into_inner);
            let mine = ready.remove(&f.index);
            // anything older belongs to a batch that was rendered and then not encoded
            let old: Vec<u64> = ready.keys().copied().filter(|&i| i <= f.index).collect();
            let stale: Vec<Yuv420> = old.into_iter().filter_map(|i| ready.remove(&i)).collect();
            (mine, stale)
        };
        // the ready lock is released: hand the unused planes back to the pool
        for planes in stale {
            recycle(&self.spare, planes);
        }
        let (w, h) = (f.width as usize, f.height as usize);
        let chroma = w.div_ceil(2) * h.div_ceil(2);
        let usable = ready.and_then(|planes| {
            if planes.0.len() == w * h && planes.1.len() == chroma && planes.2.len() == chroma {
                Some(planes)
            } else {
                recycle(&self.spare, planes);
                None
            }
        });
        match usable {
            Some((y, u, v)) => {
                let old = (std::mem::replace(&mut self.y, y), std::mem::replace(&mut self.u, u), std::mem::replace(&mut self.v, v));
                recycle(&self.spare, old);
            }
            None => {
                let (enc, y, u, v, signal) = (&self.enc, &mut self.y, &mut self.u, &mut self.v, self.signal);
                // on the encoder's own threads: the global pool is busy rendering the next batch
                timed(Stage::Convert, || enc.install(|| h264_input(f, signal, y, u, v)));
            }
        }
        let cw = (f.width as usize).div_ceil(2);
        let frame = filmcraft_h264enc::YuvFrame { y: &self.y, u: &self.u, v: &self.v, y_stride: f.width as usize, uv_stride: cw };
        let ps = self.enc.try_encode(&frame, f.index as i64 * self.rate.den).map_err(|e| ExportError::Encode(e.to_string()))?;
        Ok(self.packets(ps))
    }
    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        let ps = self.enc.flush();
        Ok(self.packets(ps))
    }
    fn media_start(&self) -> Option<i64> {
        // With B-frames the first DTS is one frame before the first PTS.
        (self.enc.delay() > 0).then_some(self.rate.den)
    }
    fn pass_stats(&self) -> Option<filmcraft_h264enc::PassStats> {
        self.enc.pass_stats()
    }
    fn prepare_ahead(&self) -> Option<PrepareAhead> {
        // High 10 converts in `encode`: the planes prepared ahead are 8-bit
        if self.deep() {
            return None;
        }
        let (ready, spare, signal) = (self.ready.clone(), self.spare.clone(), self.signal);
        Some(Arc::new(move |f: &EncoderFrame| {
            // the conversion resizes the planes in place, whatever size the set had
            let popped = spare.lock().unwrap_or_else(PoisonError::into_inner).pop();
            let (mut y, mut u, mut v) = popped.unwrap_or_default();
            timed(Stage::Convert, || h264_input(f, signal, &mut y, &mut u, &mut v));
            let replaced = ready.lock().unwrap_or_else(PoisonError::into_inner).insert(f.index, (y, u, v));
            if let Some(planes) = replaced {
                recycle(&spare, planes);
            }
        }))
    }
}

/// Slices per H.264 picture: one per four macroblock rows. The encoder's default follows the core
/// count, which would make the stream (and every decoded picture) depend on the machine; this is
/// what it picks on a machine with enough cores.
pub(crate) fn h264_slices(height: u32) -> usize {
    (height.div_ceil(16) as usize).div_ceil(4).max(1)
}

pub(crate) fn h264_factory(format: Format, w: u32, h: u32, rate: FrameRate, s: &ExportSettings) -> Option<Result<Box<dyn VideoEncoder>>> {
    if format != Format::H264 {
        return None;
    }
    let mut cfg = filmcraft_h264enc::EncoderConfig::new(w, h, rate.num as u32, rate.den as u32);
    // MXF carries the Annex B byte stream (ST 381-3) with in-band parameter sets
    cfg.format = if s.format.is_mxf() { filmcraft_h264enc::PacketFormat::AnnexB } else { filmcraft_h264enc::PacketFormat::LengthPrefixed };
    cfg.aud = s.format.is_mxf();
    cfg.keyint = s.keyframe_distance.filter(|k| *k > 0).unwrap_or_else(|| (rate.num as f64 / rate.den as f64 * 2.0).round().max(1.0) as u32);
    cfg.slices = h264_slices(h);
    let kbps = s.bitrate_kbps.max(100);
    let max = s.max_bitrate_kbps.filter(|m| *m >= kbps).unwrap_or_else(|| (u64::from(kbps) * 3 / 2).min(u64::from(u32::MAX)) as u32);
    cfg.rate = match s.bitrate_mode {
        BitrateMode::Cbr => filmcraft_h264enc::RateControl::Cbr { kbps },
        // the encoder clamps to 0..=51; a NaN from a hand-edited preset becomes the default
        BitrateMode::Crf => filmcraft_h264enc::RateControl::Crf(if s.crf.is_finite() { s.crf } else { DEFAULT_CRF }),
        _ => filmcraft_h264enc::RateControl::Vbr { target_kbps: kbps, max_kbps: max },
    };
    cfg.pass = match &s.h264_pass {
        H264Pass::Single => filmcraft_h264enc::Pass::Single,
        H264Pass::First => filmcraft_h264enc::Pass::First,
        H264Pass::Second(st) => filmcraft_h264enc::Pass::Second(st.clone()),
    };
    cfg.profile = match s.h264_profile {
        H264Profile::Baseline => filmcraft_h264enc::Profile::Baseline,
        H264Profile::Main => filmcraft_h264enc::Profile::Main,
        H264Profile::High => filmcraft_h264enc::Profile::High,
        H264Profile::High10 => filmcraft_h264enc::Profile::High10,
    };
    cfg.bit_depth = s.h264_profile.bit_depth();
    cfg.intra_only = s.h264_intra_only;
    cfg.level = s.h264_level;
    if let Some((n, d)) = s.pixel_aspect {
        cfg.sar = (n.clamp(1, 65535) as u16, d.clamp(1, 65535) as u16);
    }
    if s.signal.is_hdr() {
        cfg.color = filmcraft_h264enc::ColorConfig { primaries: s.signal.primaries, transfer: s.signal.transfer, matrix: s.signal.matrix, full_range: false };
        if let Some((md, cll)) = s.signal.static_metadata() {
            let mut b = [0u8; 24];
            b.copy_from_slice(&md.to_bytes());
            cfg.mastering_display = Some(b);
            cfg.content_light = Some(cll);
        }
    }
    Some(
        filmcraft_h264enc::Encoder::new(cfg)
            .map(|enc| {
                let (y, u, v, y16, u16, v16) = Default::default();
                Box::new(H264Encoder { enc, w, h, rate, y, u, v, y16, u16, v16, signal: s.signal, ready: Default::default(), spare: Default::default() })
                    as Box<dyn VideoEncoder>
            })
            .map_err(|e| ExportError::Encode(e.to_string())),
    )
}

#[cfg(test)]
mod reuse_tests {
    use super::*;

    const W: u32 = 96;
    const H: u32 = 64;

    fn rgba(t: u32, alpha: u8) -> Vec<u8> {
        (0..W * H)
            .flat_map(|i| {
                let (x, y) = (i % W, i / W);
                [(x * 3 + t * 37) as u8, (y * 5 + t * 11) as u8, ((x ^ y) * 2 + t * 3) as u8, alpha]
            })
            .collect()
    }

    fn hdr(t: u32) -> Vec<f32> {
        (0..W * H * 3).map(|i| ((i * 7 + t * 131) % 1000) as f32 / 900.0 - 0.05).collect()
    }

    /// One input picture: straight RGBA8 with a flat alpha, or HDR floats.
    struct Input {
        rgba: Vec<u8>,
        hdr: Option<Vec<f32>>,
    }

    fn encode(enc: &mut dyn VideoEncoder, inputs: &[Input], index: u64) -> Vec<u8> {
        let f = &inputs[index as usize];
        let frame = EncoderFrame { width: W, height: H, rgba: &f.rgba, hdr: f.hdr.as_deref(), index };
        enc.encode(&frame).expect("encodes").into_iter().flat_map(|p| p.data).collect()
    }

    /// The pictures one frame at a time through one encoder that keeps its frame must give the packets
    /// of an identical encoder given a fresh frame every time (the encoders themselves may carry state).
    fn prores_reuse_matches_fresh(s: &ExportSettings, inputs: &[Input]) {
        let (mut reused, mut fresh) = (prores_encoder(W, H, FrameRate::FPS_24, s), prores_encoder(W, H, FrameRate::FPS_24, s));
        for i in 0..inputs.len() as u64 {
            fresh.frame = None;
            assert!(encode(&mut reused, inputs, i) == encode(&mut fresh, inputs, i), "frame {i}");
        }
    }

    fn sdr(t: u32, alpha: u8) -> Input {
        Input { rgba: rgba(t, alpha), hdr: None }
    }

    fn hdr_input(t: u32) -> Input {
        Input { rgba: rgba(t, 255), hdr: Some(hdr(t)) }
    }

    #[test]
    fn prores_422_reuses_its_frame_with_identical_packets() {
        let s = ExportSettings { format: Format::ProRes, ..Default::default() };
        let inputs: Vec<Input> = (0..4).map(|t| sdr(t, 255)).collect();
        prores_reuse_matches_fresh(&s, &inputs);
        let s = ExportSettings { format: Format::ProRes, signal: ColorSignal::PQ, ..Default::default() };
        let inputs: Vec<Input> = (0..4).map(|t| if t % 2 == 0 { hdr_input(t) } else { sdr(t, 255) }).collect();
        prores_reuse_matches_fresh(&s, &inputs);
    }

    #[test]
    fn prores_4444_alpha_does_not_leak_into_the_next_hdr_frame() {
        let s = ExportSettings { format: Format::ProRes, prores_profile: "4444".into(), alpha: true, signal: ColorSignal::PQ, ..Default::default() };
        assert!(s.keeps_alpha());
        // an SDR frame with partial alpha, then an HDR frame (opaque), then SDR again
        let inputs = vec![sdr(0, 40), hdr_input(1), sdr(2, 200), sdr(3, 0), hdr_input(4)];
        prores_reuse_matches_fresh(&s, &inputs);
    }

    #[test]
    fn prores_keeps_its_frame_between_encodes() {
        let s = ExportSettings { format: Format::ProRes, ..Default::default() };
        let mut enc = prores_encoder(W, H, FrameRate::FPS_24, &s);
        let inputs: Vec<Input> = (0..3).map(|t| sdr(t, 255)).collect();
        assert!(enc.frame.is_none());
        encode(&mut enc, &inputs, 0);
        let ptr = enc.frame.as_ref().map(|f| f.y.as_ptr());
        assert!(ptr.is_some());
        encode(&mut enc, &inputs, 1);
        assert_eq!(enc.frame.as_ref().map(|f| f.y.as_ptr()), ptr, "the second frame converted into a new allocation");
        encode(&mut enc, &inputs, 2);
        assert_eq!(enc.frame.as_ref().map(|f| f.y.as_ptr()), ptr);
    }

    #[test]
    fn dnxhr_hqx_reuses_its_frame_with_identical_packets() {
        let s = ExportSettings { format: Format::DnxHr, dnx_profile: "hqx".into(), ..Default::default() };
        let inputs: Vec<Input> = (0..4).map(|t| if t == 2 { hdr_input(t) } else { sdr(t, 255) }).collect();
        let (mut reused, mut fresh) = (dnx_encoder(W, H, FrameRate::FPS_24, &s).expect("builds"), dnx_encoder(W, H, FrameRate::FPS_24, &s).expect("builds"));
        for i in 0..inputs.len() as u64 {
            fresh.frame = None;
            assert!(encode(&mut reused, &inputs, i) == encode(&mut fresh, &inputs, i), "frame {i}");
        }
        let ptr = reused.frame.as_ref().map(|f| f.y.as_ptr());
        assert!(ptr.is_some());
        encode(&mut reused, &inputs, 0);
        assert_eq!(reused.frame.as_ref().map(|f| f.y.as_ptr()), ptr, "the frame was reallocated");
    }

    fn h264_encoder() -> H264Encoder {
        let cfg = filmcraft_h264enc::EncoderConfig::new(W, H, 24, 1);
        H264Encoder {
            enc: filmcraft_h264enc::Encoder::new(cfg).expect("builds"),
            w: W,
            h: H,
            rate: FrameRate::FPS_24,
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            y16: Vec::new(),
            u16: Vec::new(),
            v16: Vec::new(),
            signal: ColorSignal::default(),
            ready: Default::default(),
            spare: Default::default(),
        }
    }

    fn frame_of(rgba: &[u8], index: u64) -> EncoderFrame<'_> {
        EncoderFrame { width: W, height: H, rgba, hdr: None, index }
    }

    fn ready_y_ptr(enc: &H264Encoder, index: u64) -> Option<*const u8> {
        enc.ready.lock().unwrap().get(&index).map(|p| p.0.as_ptr())
    }

    #[test]
    fn h264_prepared_planes_come_from_the_ones_the_encoder_finished_with() {
        let mut enc = h264_encoder();
        let prepare = enc.prepare_ahead().expect("8-bit converts ahead");
        let pics: Vec<Vec<u8>> = (0..4).map(|t| rgba(t, 255)).collect();
        prepare(&frame_of(&pics[0], 0));
        let first = ready_y_ptr(&enc, 0).expect("prepared");
        enc.encode(&frame_of(&pics[0], 0)).expect("encodes");
        assert_eq!(enc.y.as_ptr(), first, "the prepared planes are the encoder's input");
        assert!(enc.spare.lock().unwrap().is_empty(), "nothing to give back yet");
        prepare(&frame_of(&pics[1], 1));
        let second = ready_y_ptr(&enc, 1).expect("prepared");
        assert_ne!(second, first, "the first set is still in use");
        enc.encode(&frame_of(&pics[1], 1)).expect("encodes");
        assert_eq!(enc.spare.lock().unwrap().len(), 1, "frame 0's planes went to the pool");
        // the next conversion reuses them
        prepare(&frame_of(&pics[2], 2));
        assert_eq!(ready_y_ptr(&enc, 2), Some(first));
        assert!(enc.spare.lock().unwrap().is_empty());
        // and the packets are what converting in `encode` gives
        let mut plain = h264_encoder();
        let a: Vec<u8> = enc.encode(&frame_of(&pics[2], 2)).expect("encodes").into_iter().flat_map(|p| p.data).collect();
        for t in 0..2 {
            plain.encode(&frame_of(&pics[t], t as u64)).expect("encodes");
        }
        let b: Vec<u8> = plain.encode(&frame_of(&pics[2], 2)).expect("encodes").into_iter().flat_map(|p| p.data).collect();
        assert!(a == b);
    }

    #[test]
    fn h264_spare_pool_stays_bounded() {
        let mut enc = h264_encoder();
        let prepare = enc.prepare_ahead().expect("8-bit converts ahead");
        let pic = rgba(0, 255);
        // many pictures rendered and then skipped over: encode only the last, so the others are stale
        for i in 0..(3 * SPARE_PLANES as u64) {
            prepare(&frame_of(&pic, i));
        }
        let last = 3 * SPARE_PLANES as u64;
        prepare(&frame_of(&pic, last));
        enc.encode(&frame_of(&pic, last)).expect("encodes");
        assert_eq!(enc.spare.lock().unwrap().len(), SPARE_PLANES);
        assert!(enc.ready.lock().unwrap().is_empty());
        // more rounds never grow it
        for round in 1..4u64 {
            let base = last + round * 100;
            for i in 0..(2 * SPARE_PLANES as u64) {
                prepare(&frame_of(&pic, base + i));
            }
            enc.encode(&frame_of(&pic, base + 2 * SPARE_PLANES as u64 - 1)).expect("encodes");
            assert!(enc.spare.lock().unwrap().len() <= SPARE_PLANES);
        }
    }

    /// A picture whose pixels are shorter than its size is an error (it used to index past the end
    /// of the source and panic), and the encoder carries on with the next whole picture.
    #[test]
    fn a_short_picture_is_an_error_not_a_panic() {
        let inputs = [sdr(0, 255)];
        let whole = &inputs[0];
        let short_rgba = &whole.rgba[..whole.rgba.len() / 2];
        let short_hdr = hdr(0);
        let short_hdr = &short_hdr[..short_hdr.len() - 3];
        let shorts = [
            EncoderFrame { width: W, height: H, rgba: short_rgba, hdr: None, index: 0 },
            EncoderFrame { width: W, height: H, rgba: &whole.rgba, hdr: Some(short_hdr), index: 0 },
        ];
        let s = ExportSettings { format: Format::ProRes, ..Default::default() };
        let mut prores = prores_encoder(W, H, FrameRate::FPS_24, &s);
        let s = ExportSettings { format: Format::DnxHr, ..Default::default() };
        let mut dnx = dnx_encoder(W, H, FrameRate::FPS_24, &s).expect("builds");
        for enc in [&mut prores as &mut dyn VideoEncoder, &mut dnx] {
            for f in &shorts {
                assert!(enc.encode(f).is_err());
            }
            assert!(!encode(enc, &inputs, 0).is_empty());
        }
    }
}
