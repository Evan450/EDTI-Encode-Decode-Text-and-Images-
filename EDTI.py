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
Version: 5.7b
Description:
  - Fixed HMAC bug
"""

import numpy as np
import argparse, sys, wave, struct, logging, hmac, hashlib, codecs
from PIL import Image

# --- Logging configuration ---
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

########################################
#              CONFIGURATION
########################################
CONFIG = {
    'SAMPLE_RATE': 44100,
    'BIT_DURATION': 0.04,          # seconds per bit
    'FREQ0': 800,                  # tone for bit '0'
    'FREQ1': 3500,                 # tone for bit '1'
    'PREAMBLE': b'EDTI',           # 4-byte marker for synchronization
    'HEADER_SIZE': 32,             # 32 bits = 4 bytes header indicating data length
    'MIN_HMAC_SIG_LEN': 32,        # expected HMAC signature length (SHA-256)
    'TEXT_ERROR_THRESHOLD': 0.3,   # if >30% of text chars are '#', output "#"
    'MAX_IMAGE_DIMENSION': 452,    # max dimension for images
    'MAX_WAV_BYTES': (1 << 32) - 1 # 4GB minus 1
}

SAMPLE_RATE  = CONFIG['SAMPLE_RATE']
BIT_DURATION = CONFIG['BIT_DURATION']
FREQ0        = CONFIG['FREQ0']
FREQ1        = CONFIG['FREQ1']
PREAMBLE     = CONFIG['PREAMBLE']
HEADER_SIZE  = CONFIG['HEADER_SIZE']
MAX_WAV_BYTES= CONFIG['MAX_WAV_BYTES']

########################################
# Custom UTF-8 Error Handler: replace undecodable bytes with '#'
########################################
def hash_replacement(error):
    return ('#', error.end)
codecs.register_error('hashreplace', hash_replacement)

########################################
# Utility Functions
########################################

def wait_and_exit(code=1):
    input("Press Enter to exit...")
    sys.exit(code)

def ensure_wav_extension(fname):
    if not fname.lower().endswith('.wav'):
        fname += '.wav'
    return fname

def generate_tone(freq, duration, amplitude=1.0):
    N = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, N, endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
    window = np.hanning(N)
    return sine_wave * window

def save_wave_mono(fname, data):
    """
    Saves the data to a standard WAV. We enforce the 4GB limit.
    If data exceeds 4GB, we log an error and exit.
    """
    raw_bytes = data.tobytes()
    if len(raw_bytes) > MAX_WAV_BYTES:
        logging.error("WAV data exceeds 4GB limit! Aborting.")
        wait_and_exit(1)

    with wave.open(fname, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)
    logging.info(f"Saved mono WAV => '{fname}'.")

def read_wave_mono(fname):
    with wave.open(fname, 'rb') as wf:
        if wf.getnchannels() != 1:
            logging.error("Expected mono WAV.")
            wait_and_exit(1)
        frames = wf.readframes(wf.getnframes())
        arr = np.frombuffer(frames, dtype=np.int16)
    logging.info(f"Read WAV => '{fname}', frames={wf.getnframes()}")
    return arr

def hmac_sign(data, passphrase):
    return hmac.new(passphrase.encode('utf-8'), data, hashlib.sha256).digest()

def hmac_verify(data, signature, passphrase):
    return hmac.compare_digest(hmac_sign(data, passphrase), signature)

########################################
# CRC16 Implementation (Segment-level error detection)
########################################

def crc16_ccitt(data, poly=0x1021, init=0xFFFF):
    """
    Compute a 16-bit CRC (CRC-CCITT variant).
    """
    crc = init
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if (crc & 0x8000) != 0:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF

########################################
# Payload Construction (Text & Image)
########################################

def build_payload(text=None, image=None, image_mode=None):
    """
    Build a payload from text and/or image, with segment-level CRC16 appended.
    Each segment:
      1 byte: segment type (1=text, 2=image, 255=end)
      4 bytes: segment length (data + 2 CRC bytes)
      data
      2 bytes: CRC16
    """
    payload = bytearray()
    TYPE_TEXT  = 1
    TYPE_IMAGE = 2
    TYPE_END   = 255

    # Text Segment
    if text:
        seg_data = text.encode('utf-8', errors='hashreplace')
        seg_crc = crc16_ccitt(seg_data)
        seg_len = len(seg_data) + 2  # +2 for the CRC
        payload.append(TYPE_TEXT)
        payload += struct.pack('>I', seg_len)
        payload += seg_data
        payload += struct.pack('>H', seg_crc)

    # Image Segment
    if image:
        try:
            im = Image.open(image)
            # Resize if needed
            if im.width > CONFIG['MAX_IMAGE_DIMENSION'] or im.height > CONFIG['MAX_IMAGE_DIMENSION']:
                im.thumbnail((CONFIG['MAX_IMAGE_DIMENSION'], CONFIG['MAX_IMAGE_DIMENSION']))

            # Determine color mode
            if image_mode == 'grayscale':
                im = im.convert('L')
            elif image_mode == 'rgb':
                im = im.convert('RGB')
            else:
                # auto
                if im.mode in ['RGB', 'RGBA', 'P']:
                    im = im.convert('RGB')
                else:
                    im = im.convert('L')

            w, h = im.size
            img_data = np.array(im).tobytes()
            meta = struct.pack('>HH', w, h)
            seg_data = meta + img_data
            seg_crc = crc16_ccitt(seg_data)
            seg_len = len(seg_data) + 2  # +2 for CRC
            payload.append(TYPE_IMAGE)
            payload += struct.pack('>I', seg_len)
            payload += seg_data
            payload += struct.pack('>H', seg_crc)
        except Exception as e:
            logging.error(f"Error processing image: {e}")

    # End Marker
    seg_crc = 0
    seg_len = 2  # minimal 2 bytes for CRC
    payload.append(TYPE_END)
    payload += struct.pack('>I', seg_len)
    payload += struct.pack('>H', seg_crc)

    return bytes(payload)

def prepare_for_transmission(payload, passphrase=None):
    """
    1. Prepend a 4-byte payload length.
    2. If passphrase is provided, append 4-byte signature length + HMAC signature.
    3. Prepend a preamble and a 4-byte header (data length).
    """
    data = struct.pack('>I', len(payload)) + payload
    if passphrase:
        sig = hmac_sign(data, passphrase)
        data += struct.pack('>I', len(sig)) + sig
    header = PREAMBLE + struct.pack('>I', len(data))
    return header + data

########################################
# Modulation/Demodulation Functions
########################################

def bits_from_bytes(data_bytes):
    return ''.join(f'{b:08b}' for b in data_bytes)

def modulate_bits(bitstream):
    tone0 = generate_tone(FREQ0, BIT_DURATION).astype(np.float32)
    tone1 = generate_tone(FREQ1, BIT_DURATION).astype(np.float32)
    samples = []
    for bit in bitstream:
        samples.append(tone1.copy() if bit=='1' else tone0.copy())
    return np.concatenate(samples)

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

def find_preamble(bitstream, preamble_bits):
    idx = bitstream.find(preamble_bits)
    if idx >= 0:
        return idx + len(preamble_bits)
    return -1

########################################
# Encoding and Decoding Routines
########################################

def create_audio_data(text=None, image=None, image_mode=None, passphrase=None):
    """
    Helper that builds the payload, modulates it to audio, and returns the np.int16 array.
    """
    payload = build_payload(text, image, image_mode)
    full_packet = prepare_for_transmission(payload, passphrase)
    bitstream = bits_from_bytes(full_packet)
    audio = modulate_bits(bitstream)
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16

def encode_data(text=None, image=None, image_mode=None, passphrase=None):
    """
    Attempt encoding in the requested/auto color mode. If it exceeds 4GB, 
    automatically try grayscale (if not already grayscale).
    """
    # 1) Try requested/auto color mode
    audio_int16 = create_audio_data(text, image, image_mode, passphrase)
    raw_bytes = audio_int16.tobytes()
    if len(raw_bytes) <= MAX_WAV_BYTES:
        return audio_int16

    # If we exceed 4GB and the user didn't explicitly request grayscale, try forced grayscale
    if image and image_mode != 'grayscale':
        logging.warning("Audio exceeds 4GB limit; attempting forced grayscale.")
        audio_int16_gs = create_audio_data(text, image, 'grayscale', passphrase)
        if len(audio_int16_gs.tobytes()) <= MAX_WAV_BYTES:
            return audio_int16_gs
        else:
            logging.error("Forced grayscale audio exceeds 4GB limit. Aborting.")
            wait_and_exit(1)
    else:
        # Already grayscale or no image => no further fallback
        logging.error("Audio exceeds 4GB limit. No further fallback. Aborting.")
        wait_and_exit(1)

def decode_data(audio_arr, passphrase=None):
    bitstream = demodulate_audio(audio_arr)
    preamble_bits = bits_from_bytes(PREAMBLE)
    start_idx = find_preamble(bitstream, preamble_bits)
    if start_idx < 0:
        logging.error("Preamble not found.")
        return None

    # Read header: next 32 bits => data length
    header_bits = bitstream[start_idx:start_idx+32]
    if len(header_bits) == 32:
        data_length = int(header_bits, 2)
        data_start = start_idx + 32
        data_bits = bitstream[data_start:data_start+data_length*8]
        if len(data_bits) < data_length*8:
            logging.error("Incomplete data bits per header.")
            data_bytes = None
        else:
            data_bytes = bytes(int(data_bits[i:i+8], 2) for i in range(0, len(data_bits), 8))
    else:
        logging.error("Header incomplete.")
        data_bytes = None

    # Backup: if header fails or looks invalid, try end-marker detection
    if data_bytes is None or not data_length or data_length > len(bitstream)//8:
        logging.warning("Header appears corrupted; attempting backup via end marker.")
        remaining_bits = bitstream[start_idx+32:]
        remaining_bytes = bytes(int(remaining_bits[i:i+8], 2) for i in range(0, len(remaining_bits), 8))
        marker = bytes([255]) + b'\x00\x00\x00\x00'
        pos = remaining_bytes.find(marker)
        if pos >= 0:
            data_bytes = remaining_bytes[:pos]
        else:
            logging.error("Backup end marker detection failed.")
            return None

    # Verify HMAC if passphrase was used
    if passphrase and data_bytes is not None:
        if len(data_bytes) < 4:
            logging.error("Data too short for HMAC check.")
            return None
        sig_len = struct.unpack('>I', data_bytes[-4:])[0]
        if len(data_bytes) < 4 + sig_len:
            logging.error("Data too short for HMAC signature.")
            return None
        sig = data_bytes[-(4+sig_len):-4]
        main_data = data_bytes[:-(4+sig_len)]
        if not hmac_verify(main_data, sig, passphrase):
            logging.error("HMAC verification failed. Message may have been tampered with.")
            choice = input("Continue decoding anyway? (Y/N): ").strip().lower()
            if choice != 'y':
                logging.info("Decoding cancelled by user.")
                return None
            else:
                logging.info("Continuing despite HMAC failure.")
        data_bytes = main_data  # Always remove HMAC signature

    if not data_bytes or len(data_bytes) < 4:
        logging.error("No valid data after header/HMAC checks.")
        return None

    payload_length = struct.unpack('>I', data_bytes[:4])[0]
    # If declared length > available, use what we have
    if 4 + payload_length > len(data_bytes):
        logging.warning("Declared payload length exceeds available data; using available data instead.")
        payload = data_bytes[4:]
    else:
        payload = data_bytes[4:4+payload_length]

    return payload

def parse_payload(payload):
    """
    Parse the payload into segments. 
    For text => 'hashreplace'.
    For images => best-effort reconstruction.
    """
    i = 0
    segments = {}
    TYPE_END = 255

    while i < len(payload):
        if i + 5 > len(payload):
            logging.warning("Not enough bytes left for a complete segment header; stopping.")
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
            logging.warning("Segment data too short to contain CRC.")
            break

        declared_crc = struct.unpack('>H', seg_data[-2:])[0]
        core_data = seg_data[:-2]
        computed_crc = crc16_ccitt(core_data)
        if computed_crc != declared_crc:
            logging.warning(f"Segment CRC mismatch: declared=0x{declared_crc:04X}, computed=0x{computed_crc:04X}")

        if seg_type == 1:
            # text
            txt = core_data.decode('utf-8', errors='hashreplace')
            segments.setdefault('text', []).append(txt)
        elif seg_type == 2:
            # image
            if len(core_data) < 4:
                logging.error("Image segment too short.")
                continue
            w, h = struct.unpack('>HH', core_data[:4])
            img_data = core_data[4:]
            try:
                # Attempt RGB
                from PIL import Image
                im = Image.frombytes('RGB', (w, h), img_data)
            except:
                # fallback grayscale
                im = Image.frombytes('L', (w, h), img_data)
            segments.setdefault('images', []).append(im)
        elif seg_type == TYPE_END:
            logging.info("End marker encountered; done parsing.")
            break
        else:
            logging.warning(f"Unknown segment type {seg_type}, treating as corrupted text.")
            txt = core_data.decode('utf-8', errors='hashreplace')
            segments.setdefault('text', []).append(txt)

    return segments

########################################
# Main
########################################

def main():
    parser = argparse.ArgumentParser(description="EDTI Encode Decode Text and Images")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--encode', action='store_true', help="Encode text/image into audio")
    group.add_argument('--decode', action='store_true', help="Decode text/image from audio")
    parser.add_argument('--text-data', type=str, help="Text to encode")
    parser.add_argument('--image-file', type=str, help="Path to an image to encode")
    parser.add_argument('--image-mode', choices=['rgb','grayscale'], help="Force image mode; omit for auto")
    parser.add_argument('--passphrase', type=str, help="Passphrase for HMAC signing/verification")
    parser.add_argument('--input', type=str, help="Input WAV file for decoding")
    parser.add_argument('--output', type=str, help="Output WAV file for encoding")
    args = parser.parse_args()

    if args.encode:
        out_wav = ensure_wav_extension(args.output or 'Output.wav')
        audio_int16 = encode_data(
            text=args.text_data,
            image=args.image_file,
            image_mode=args.image_mode,
            passphrase=args.passphrase
        )
        save_wave_mono(out_wav, audio_int16)
        logging.info("Encoding completed successfully.")

    else:
        in_wav = args.input or (args.output or 'Output.wav')
        audio_arr = read_wave_mono(in_wav)
        payload = decode_data(audio_arr, passphrase=args.passphrase)
        if not payload:
            logging.error("No valid payload decoded; decode failed.")
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
