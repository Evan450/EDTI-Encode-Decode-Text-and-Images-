#!/usr/bin/env python
"""
 Copyright (C) 2025 Discover Interactive

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see https://choosealicense.com/licenses/gpl-3.0/ or https://www.gnu.org/licenses/gpl-3.0.en.html.

-- Updates and Secondary Header --

Name: Encode Decode Text and Images (EDTI)
Author: Discover Interactive
Version: 6.1.0b
Description:
  - Added FDM support (automatic/manual), must be activated first to work.
  - Fixed miscellaneous bugs and issues that were found.
  - Added memory warning and chunking to reduce chance of memory overhead.
"""

import numpy as np
import argparse, sys, wave, struct, logging, hmac, hashlib, codecs, math, os
from PIL import Image

# Logging configuration
logging.basicConfig(
    filename='edti.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# CONFIGURATION
CONFIG = {
    'SAMPLE_RATE': 44100,
    'BIT_DURATION': 0.04,
    'FREQ0': 800,
    'FREQ1': 3500,
    'PREAMBLE': b'EDTI',
    'HEADER_SIZE': 32,
    'MIN_HMAC_SIG_LEN': 32,
    'TEXT_ERROR_THRESHOLD': 0.3,
    'MAX_IMAGE_DIMENSION': 452,
    'MAX_WAV_BYTES': (1 << 32) - 1
}
SAMPLE_RATE  = CONFIG['SAMPLE_RATE']
BIT_DURATION = CONFIG['BIT_DURATION']
FREQ0        = CONFIG['FREQ0']
FREQ1        = CONFIG['FREQ1']
PREAMBLE     = CONFIG['PREAMBLE']
HEADER_SIZE  = CONFIG['HEADER_SIZE']
MAX_WAV_BYTES= CONFIG['MAX_WAV_BYTES']

# FDM Parameters
MAX_AUTO_CHANNELS = 16
MIN_FREQ_GAP = 500
SEARCH_STEP = 10
FREQ_SEARCH_RANGE = (200, 10000)

# Part watermark constants (for multi-file ordering)
PART_INFO_MARKER = 0x7FFF

# Audio processing check
if CONFIG['BIT_DURATION'] < 0.01:
    logging.warning(f"BIT_DURATION of {BIT_DURATION}s is very short and may cause decoding issues.")
elif CONFIG['BIT_DURATION'] > 0.5:
    logging.warning(f"BIT_DURATION of {BIT_DURATION}s is very long and will result in large files.")

# Custom UTF-8 error handler
def hash_replacement(error):
    return ('#', error.end)
codecs.register_error('hashreplace', hash_replacement)

# Utility Functions
def wait_and_exit(code=1):
    input("Press Enter to exit...")
    sys.exit(code)

def ensure_wav_extension(fname):
    return fname if fname.lower().endswith('.wav') else fname + '.wav'

def generate_tone(freq, duration, amplitude=1.0):
    N = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, N, endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
    window = np.hanning(N)
    return sine_wave * window

def save_wave_mono(fname, data):
    raw_bytes = data.tobytes()
    if len(raw_bytes) > MAX_WAV_BYTES:
        logging.error("WAV data exceeds 4GB limit! Aborting.")
        wait_and_exit(1)
    with wave.open(fname, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)
    logging.info(f"Saved WAV file => '{fname}' (size: {len(raw_bytes)} bytes)")

def read_wave_mono(fname):
    with wave.open(fname, 'rb') as wf:
        if wf.getnchannels() != 1:
            logging.error("Expected mono WAV.")
            wait_and_exit(1)
        frames = wf.readframes(wf.getnframes())
        arr = np.frombuffer(frames, dtype=np.int16)
    logging.info(f"Read WAV file => '{fname}', frames={wf.getnframes()}")
    return arr

def hmac_sign(data, passphrase):
    return hmac.new(passphrase.encode('utf-8'), data, hashlib.sha256).digest()

def hmac_verify(data, signature, passphrase):
    return hmac.compare_digest(hmac_sign(data, passphrase), signature)

def crc16_ccitt(data, poly=0x1021, init=0xFFFF):
    crc = init
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc & 0xFFFF

# Payload Construction
def build_payload(text=None, image=None, image_mode=None):
    payload = bytearray()
    TYPE_TEXT  = 1
    TYPE_IMAGE = 2
    TYPE_END   = 255

    if text:
        seg_data = text.encode('utf-8', errors='hashreplace')
        seg_crc = crc16_ccitt(seg_data)
        seg_len = len(seg_data) + 2
        payload.append(TYPE_TEXT)
        payload += struct.pack('>I', seg_len)
        payload += seg_data
        payload += struct.pack('>H', seg_crc)

    if image:
        try:
            im = Image.open(image)
            if im.width > CONFIG['MAX_IMAGE_DIMENSION'] or im.height > CONFIG['MAX_IMAGE_DIMENSION']:
                im.thumbnail((CONFIG['MAX_IMAGE_DIMENSION'], CONFIG['MAX_IMAGE_DIMENSION']))
            if image_mode == 'grayscale':
                im = im.convert('L')
            elif image_mode == 'rgb':
                im = im.convert('RGB')
            else:
                im = im.convert('RGB' if im.mode in ['RGB', 'RGBA', 'P'] else 'L')
            w, h = im.size
            img_data = np.array(im).tobytes()
            meta = struct.pack('>HH', w, h)
            seg_data = meta + img_data
            seg_crc = crc16_ccitt(seg_data)
            seg_len = len(seg_data) + 2
            payload.append(TYPE_IMAGE)
            payload += struct.pack('>I', seg_len)
            payload += seg_data
            payload += struct.pack('>H', seg_crc)
        except Exception as e:
            logging.error(f"Error processing image: {e}")

    seg_crc = 0
    seg_len = 2
    payload.append(TYPE_END)
    payload += struct.pack('>I', seg_len)
    payload += struct.pack('>H', seg_crc)
    return bytes(payload)

def prepare_for_transmission(payload, passphrase=None):
    data = struct.pack('>I', len(payload)) + payload
    if passphrase:
        sig = hmac_sign(data, passphrase)
        data += struct.pack('>I', len(sig)) + sig
    header = PREAMBLE + struct.pack('>I', len(data))
    return header + data

# FDM Frequency Allocation
def allocate_fdm_frequencies(n_channels, min_gap=MIN_FREQ_GAP, search_step=SEARCH_STEP, search_range=FREQ_SEARCH_RANGE):
    allocated = []
    allocated.append((FREQ0, FREQ1))
    if n_channels == 1:
        return allocated, None
    for ch in range(1, n_channels):
        candidate0 = FREQ0 + ch * min_gap
        candidate1 = FREQ1 + ch * min_gap
        if candidate0 < search_range[0] or candidate1 < search_range[0]:
            candidate0 = search_range[0]
            candidate1 = search_range[0] + (FREQ1 - FREQ0)
        def is_valid(c0, c1):
            for (f0, f1) in allocated:
                if abs(c0 - f0) < min_gap or abs(c0 - f1) < min_gap:
                    return False
                if abs(c1 - f0) < min_gap or abs(c1 - f1) < min_gap:
                    return False
            return (abs(c1 - c0) >= min_gap)
        if not is_valid(candidate0, candidate1):
            found = False
            search_limit = search_range[1] - min_gap
            while candidate0 < search_limit and candidate1 < search_range[1]:
                candidate0 += search_step
                candidate1 += search_step
                if is_valid(candidate0, candidate1):
                    found = True
                    break
            if not found:
                logging.warning(f"Could not allocate frequencies for channel {ch}; using {len(allocated)} channels instead.")
                return allocated, ch
        allocated.append((candidate0, candidate1))
    return allocated, None

# Modulation/Demodulation
def bits_from_bytes(data_bytes):
    return ''.join(f'{b:08b}' for b in data_bytes)

def modulate_bits(bitstream):
    tone0 = generate_tone(FREQ0, BIT_DURATION).astype(np.float32)
    tone1 = generate_tone(FREQ1, BIT_DURATION).astype(np.float32)
    samples = [tone1.copy() if bit == '1' else tone0.copy() for bit in bitstream]
    return np.concatenate(samples)

def modulate_bits_fdm(bitstream, desired_channels):
    freq_alloc, failed_channel = allocate_fdm_frequencies(desired_channels)
    actual_channels = len(freq_alloc)
    if failed_channel is not None:
        logging.warning(f"Allocated only {actual_channels} channels (requested {desired_channels}).")
    sample_per_bit = int(SAMPLE_RATE * BIT_DURATION)
    channel_bits = ["" for _ in range(actual_channels)]
    for idx, bit in enumerate(bitstream):
        channel_bits[idx % actual_channels] += bit
    
    # Calculate total output size and check memory requirements
    max_len = max(len(bits) for bits in channel_bits)
    total_samples = max_len * sample_per_bit
    estimated_memory_gb = (total_samples * 4) / (1024**3)  # 4 bytes per float32
    
    if estimated_memory_gb > 2:  # Warn if more than 2GB
        logging.warning(f"Processing requires approximately {estimated_memory_gb:.2f} GB of memory")
        if estimated_memory_gb > 8:  # Critical if more than 8GB
            logging.error(f"Memory requirement of {estimated_memory_gb:.2f} GB exceeds reasonable limits")
            choice = input("Continue anyway? This may crash your system! (Y/N): ").strip().lower()
            if choice != 'y':
                logging.info("Operation cancelled by user due to memory concerns.")
                wait_and_exit(1)
    
    # Process in chunks to reduce memory usage
    chunk_size = min(1000, max_len)  # Process 1000 frames at a time or less if fewer frames
    combined_signal = np.array([], dtype=np.float32)
    
    for chunk_start in range(0, max_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, max_len)
        logging.info(f"Processing FDM chunk {chunk_start+1}-{chunk_end} of {max_len}")
        
        chunk_frames = []
        for frame_idx in range(chunk_start, chunk_end):
            frame_signal = np.zeros(sample_per_bit, dtype=np.float32)
            for ch in range(actual_channels):
                if frame_idx < len(channel_bits[ch]):
                    bit = channel_bits[ch][frame_idx]
                    chan_freq0, chan_freq1 = freq_alloc[ch]
                    tone = generate_tone(chan_freq1 if bit == '1' else chan_freq0, BIT_DURATION)
                    frame_signal += tone.astype(np.float32)
            chunk_frames.append(frame_signal)
        
        chunk_signal = np.concatenate(chunk_frames)
        # Normalize each chunk
        if np.max(np.abs(chunk_signal)) > 0:
            chunk_signal = chunk_signal / np.max(np.abs(chunk_signal))
        
        combined_signal = np.append(combined_signal, chunk_signal)
        
        # Free memory
        del chunk_frames
        del chunk_signal
        import gc
        gc.collect()
    
    # Final normalization
    if np.max(np.abs(combined_signal)) > 0:
        combined_signal = combined_signal / np.max(np.abs(combined_signal))
    
    return combined_signal

def demodulate_audio_fdm(audio_arr, desired_channels):
    freq_alloc, failed_channel = allocate_fdm_frequencies(desired_channels)
    actual_channels = len(freq_alloc)
    if failed_channel is not None:
        logging.warning(f"Demodulating using {actual_channels} channels (requested {desired_channels}).")
    sample_per_bit = int(SAMPLE_RATE * BIT_DURATION)
    total_frames = len(audio_arr) // sample_per_bit
    
    # Pre-generate all tone templates
    tone_templates = []
    for ch in range(actual_channels):
        chan_freq0, chan_freq1 = freq_alloc[ch]
        tone0 = generate_tone(chan_freq0, BIT_DURATION).astype(np.float64)
        tone1 = generate_tone(chan_freq1, BIT_DURATION).astype(np.float64)
        tone_templates.append((tone0, tone1))
    
    channels_bits = ["" for _ in range(actual_channels)]
    for i in range(total_frames):
        frame = audio_arr[i*sample_per_bit:(i+1)*sample_per_bit].astype(np.float64)
        for ch in range(actual_channels):
            tone0, tone1 = tone_templates[ch]
            bit = '1' if np.dot(frame, tone1) > np.dot(frame, tone0) else '0'
            channels_bits[ch] += bit
    
    reassembled = ""
    max_len = max(len(bits) for bits in channels_bits)
    for i in range(max_len):
        for ch in range(actual_channels):
            if i < len(channels_bits[ch]):
                reassembled += channels_bits[ch][i]
    return reassembled

def demodulate_audio(audio_arr):
    sample_per_bit = int(SAMPLE_RATE * BIT_DURATION)
    bitstream = ''
    total_samples = len(audio_arr)
    for i in range(0, total_samples, sample_per_bit):
        block = audio_arr[i:i+sample_per_bit]
        if len(block) < sample_per_bit:
            break
        tone0 = generate_tone(FREQ0, BIT_DURATION).astype(np.float64)
        tone1 = generate_tone(FREQ1, BIT_DURATION).astype(np.float64)
        block = block.astype(np.float64)
        corr0 = np.dot(block, tone0)
        corr1 = np.dot(block, tone1)
        bitstream += '1' if corr1 > corr0 else '0'
    return bitstream

def create_audio_data(text=None, image=None, image_mode=None, passphrase=None, fdm=False, channels=1):
    payload = build_payload(text, image, image_mode)
    full_packet = prepare_for_transmission(payload, passphrase)
    bitstream = bits_from_bytes(full_packet)
    if fdm:
        audio = modulate_bits_fdm(bitstream, desired_channels=channels)
    else:
        audio = modulate_bits(bitstream)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16

def encode_data(text=None, image=None, image_mode=None, passphrase=None, fdm=False, channels=0):
    multi_file = False
    if fdm:
        if channels == 0:
            for n in range(1, MAX_AUTO_CHANNELS + 1):
                audio_int16 = create_audio_data(text, image, image_mode, passphrase, fdm=True, channels=n)
                if len(audio_int16.tobytes()) <= MAX_WAV_BYTES:
                    logging.info(f"Auto FDM selected {n} channel(s) to fit in one file.")
                    channels = n
                    break
            else:
                channels = MAX_AUTO_CHANNELS
                multi_file = True
                logging.warning("Maximum auto channels reached but payload still exceeds file size limit.")
        else:
            audio_int16 = create_audio_data(text, image, image_mode, passphrase, fdm=True, channels=channels)
            if len(audio_int16.tobytes()) > MAX_WAV_BYTES:
                multi_file = True
                logging.warning("Manual FDM channels still result in payload overflow.")
        audio_int16 = create_audio_data(text, image, image_mode, passphrase, fdm=True, channels=channels)
    else:
        audio_int16 = create_audio_data(text, image, image_mode, passphrase, fdm=False, channels=1)
    
    raw_bytes = audio_int16.tobytes()
    if len(raw_bytes) <= MAX_WAV_BYTES and not multi_file:
        audio_int16 = embed_part_info(audio_int16, part_index=1, part_total=1)
        return [audio_int16]
    else:
        logging.info("Splitting payload into multiple WAV files...")
        samples_per_file = (MAX_WAV_BYTES // 2) - 1024
        total_samples = len(audio_int16)
        num_files = math.ceil(total_samples / samples_per_file)
        audio_files = []
        for i in range(num_files):
            start_idx = i * samples_per_file
            end_idx = min((i + 1) * samples_per_file, total_samples)
            part = audio_int16[start_idx:end_idx]
            part = embed_part_info(part, part_index=i+1, part_total=num_files)
            if len(part.tobytes()) > MAX_WAV_BYTES:
                logging.error("Part size still exceeds WAV file limit after splitting. Aborting.")
                wait_and_exit(1)
            audio_files.append(part)
            logging.info(f"Prepared part {i+1}/{num_files} with {len(part)} samples (including header).")
        return audio_files

def decode_data(audio_arr, passphrase=None, fdm=False, channels=1):
    if fdm:
        bitstream = demodulate_audio_fdm(audio_arr, desired_channels=channels)
    else:
        bitstream = demodulate_audio(audio_arr)
    preamble_bits = bits_from_bytes(PREAMBLE)
    start_idx = bitstream.find(preamble_bits)
    if start_idx >= 0:
        start_idx += len(preamble_bits)
    else:
        logging.error("Preamble not found during decoding.")
        return None
    header_bits = bitstream[start_idx:start_idx+32]
    if len(header_bits) == 32:
        data_length = int(header_bits, 2)
        data_start = start_idx + 32
        data_bits = bitstream[data_start:data_start+data_length*8]
        if len(data_bits) < data_length*8:
            logging.error("Incomplete data per header during decoding.")
            data_bytes = None
        else:
            data_bytes = bytes(int(data_bits[i:i+8], 2) for i in range(0, len(data_bits), 8))
    else:
        logging.error("Incomplete header during decoding.")
        data_bytes = None
    if data_bytes is None or data_length <= 0:
        logging.warning("Header appears corrupted; attempting backup end-marker detection.")
        remaining_bits = bitstream[start_idx+32:]
        remaining_bytes = bytes(int(remaining_bits[i:i+8], 2) for i in range(0, len(remaining_bits), 8))
        marker = bytes([255]) + b'\x00\x00\x00\x00'
        pos = remaining_bytes.find(marker)
        if pos >= 0:
            data_bytes = remaining_bytes[:pos]
        else:
            logging.error("Backup end marker detection failed during decoding.")
            return None
    if passphrase and data_bytes is not None:
        if len(data_bytes) < 4:
            logging.error("Data too short for HMAC check during decoding.")
            return None
        sig_len = struct.unpack('>I', data_bytes[-4:])[0]
        if len(data_bytes) < 4 + sig_len:
            logging.error("Data too short for HMAC signature during decoding.")
            return None
        sig = data_bytes[-(4+sig_len):-4]
        main_data = data_bytes[:-(4+sig_len)]
        if not hmac_verify(main_data, sig, passphrase):
            logging.error("HMAC verification failed. Data may have been tampered with.")
            choice = input("Continue decoding anyway? This is NOT RECOMMENDED for security-sensitive data (Y/N): ").strip().lower()
            if choice != 'y':
                logging.info("Decoding cancelled by user.")
                return None
            else:
                logging.info("Continuing decoding despite HMAC failure - security compromised!")
        data_bytes = main_data
    if not data_bytes or len(data_bytes) < 4:
        logging.error("No valid data found after header/HMAC checks.")
        return None
    payload_length = struct.unpack('>I', data_bytes[:4])[0]
    if 4 + payload_length > len(data_bytes):
        logging.warning("Declared payload length exceeds available data; using available data instead.")
        payload = data_bytes[4:]
    else:
        payload = data_bytes[4:4+payload_length]
    return payload

def parse_payload(payload):
    i = 0
    segments = {}
    TYPE_END = 255
    while i < len(payload):
        if i + 5 > len(payload):
            logging.warning("Incomplete segment header during parsing; stopping.")
            break
        seg_type = payload[i]
        i += 1
        seg_len = struct.unpack('>I', payload[i:i+4])[0]
        i += 4
        if i + seg_len > len(payload):
            logging.warning("Segment length exceeds available data; truncating segment.")
            seg_data = payload[i:]
            i = len(payload)
        else:
            seg_data = payload[i:i+seg_len]
            i += seg_len
        if len(seg_data) < 2:
            logging.warning("Segment too short to contain a CRC; stopping parsing.")
            break
        declared_crc = struct.unpack('>H', seg_data[-2:])[0]
        core_data = seg_data[:-2]
        computed_crc = crc16_ccitt(core_data)
        if computed_crc != declared_crc:
            logging.warning(f"CRC mismatch: declared=0x{declared_crc:04X} vs computed=0x{computed_crc:04X}")
        if seg_type == 1:
            txt = core_data.decode('utf-8', errors='hashreplace')
            segments.setdefault('text', []).append(txt)
        elif seg_type == 2:
            if len(core_data) < 4:
                logging.error("Image segment too short during parsing.")
                continue
            w, h = struct.unpack('>HH', core_data[:4])
            img_data = core_data[4:]
            try:
                im = Image.frombytes('RGB', (w, h), img_data)
            except Exception:
                try:
                    im = Image.frombytes('L', (w, h), img_data)
                except Exception as e:
                    logging.error(f"Failed to reconstruct image: {e}")
                    continue
            segments.setdefault('images', []).append(im)
        elif seg_type == TYPE_END:
            logging.info("End marker reached during payload parsing.")
            break
        else:
            try:
                txt = core_data.decode('utf-8', errors='hashreplace')
                segments.setdefault('text', []).append(txt)
            except Exception as e:
                logging.error(f"Failed to decode unknown segment type as text: {e}")
    return segments

# New Function to embed part info
def embed_part_info(audio_int16, part_index, part_total):
    header_samples = np.array([PART_INFO_MARKER, part_total, part_index, 0], dtype=np.int16)
    return np.concatenate([header_samples, audio_int16])

# Function to read part info from file
def read_wave_mono_with_part_info(fname):
    arr = read_wave_mono(fname)
    if len(arr) < 4:
        logging.error("Audio file too short to contain part header.")
        wait_and_exit(1)
    if arr[0] == PART_INFO_MARKER:
        part_total = int(arr[1])
        part_index = int(arr[2])
        audio_data = arr[4:]
        logging.info(f"File '{fname}' has embedded part info: part {part_index} of {part_total}.")
        return part_index, part_total, audio_data
    else:
        logging.info(f"File '{fname}' does not have embedded part info; treating as a single file.")
        return 1, 1, arr

# Main Routine
def main():
    parser = argparse.ArgumentParser(description="EDTI Encode Decode Text and Images")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--encode', action='store_true', help="Encode text/image into audio")
    group.add_argument('--decode', action='store_true', help="Decode text/image from audio")
    parser.add_argument('--text-data', type=str, help="Text to encode")
    parser.add_argument('--image-file', type=str, help="Path to an image to encode")
    parser.add_argument('--image-mode', choices=['rgb', 'grayscale'], help="Force image mode; omit for auto")
    parser.add_argument('--passphrase', type=str, help="Passphrase for HMAC signing/verification")
    parser.add_argument('--output', type=str, help="Base output WAV file name for encoding")
    parser.add_argument('--input', nargs='+', type=str, help="Input WAV file(s) for decoding")
    parser.add_argument('--fdm', action='store_true', help="Enable FDM mode")
    parser.add_argument('--fdm-channels', type=int, default=0,
                        help="Desired number of FDM channels; 0 for automatic (default: 0)")
    args = parser.parse_args()

    if args.encode:
        base_out = args.output or 'Output.wav'
        if not args.fdm:
            audio_files = [embed_part_info(create_audio_data(text=args.text_data,
                                                              image=args.image_file,
                                                              image_mode=args.image_mode,
                                                              passphrase=args.passphrase,
                                                              fdm=False, channels=1), part_index=1, part_total=1)]
        else:
            audio_files = encode_data(text=args.text_data, image=args.image_file,
                                      image_mode=args.image_mode, passphrase=args.passphrase,
                                      fdm=True, channels=args.fdm_channels)
        if len(audio_files) == 1:
            out_wav = ensure_wav_extension(base_out)
            save_wave_mono(out_wav, audio_files[0])
        else:
            base, ext = os.path.splitext(base_out)
            for idx, audio in enumerate(audio_files, start=1):
                part_name = ensure_wav_extension(f"{base}_part{idx}{ext}")
                save_wave_mono(part_name, audio)
            logging.info(f"Encoding completed in {len(audio_files)} parts.")
    else:
        if not args.input:
            logging.error("No input file(s) provided for decoding.")
            wait_and_exit(1)
        parts = []
        for fname in args.input:
            part_index, part_total, audio_data = read_wave_mono_with_part_info(fname)
            parts.append((part_index, audio_data))
        parts.sort(key=lambda x: x[0])
        combined_audio = np.concatenate([data for (idx, data) in parts])
        channels = args.fdm_channels if args.fdm else 1
        payload = decode_data(combined_audio, passphrase=args.passphrase, fdm=args.fdm, channels=channels)
        if not payload:
            logging.error("No valid payload decoded; decoding failed.")
            return
        segments = parse_payload(payload)
        for i, txt in enumerate(segments.get('text', []), 1):
            logging.info(f"Decoded TEXT #{i}: {txt}")
        for i, im in enumerate(segments.get('images', []), 1):
            outf = f"decoded_image_{i}.png"
            try:
                im.save(outf)
                logging.info(f"Decoded IMAGE saved as {outf}")
            except Exception as e:
                logging.error(f"Error saving image #{i}: {e}")

if __name__ == '__main__':
    main()
