Encode Decode Text and Images Manual (v6.1 Beta)
=================================================

1. Introduction

EDTI (Encode Decode Text and Images) is a tool that encodes text and images into audio using frequency-shift keying (FSK) modulation and decodes them back to their original form. In version 6.1b, several significant improvements have been introduced:
  - **FDM (Frequency Division Multiplexing) Support:** New FDM mode allows encoding over multiple channels simultaneously. Users can opt for automatic channel allocation (up to 16 channels) or manually specify the number of channels using command-line options.
  - **Enhanced Memory Management:** The script now estimates memory usage before processing. If the expected memory exceeds reasonable limits, a warning is issued—and, in extreme cases, the user may be prompted for confirmation before proceeding.
  - **Multi-file Output:** To address potential file size issues (WAV file size limit of 4GB), the encoded payload may be split into multiple WAV files. Each part is marked with embedded part information to ensure proper reassembly during decoding.
  - **Miscellaneous Bug Fixes and Improvements:** Additional fixes include better error handling (especially for HMAC verification) and refined payload reconstruction.

2. How It Works

**Payload Construction:**  
- Text and image data are broken into segments.  
- Each segment consists of:
  - A type marker (1 for text, 2 for image, 255 for end).
  - A 4-byte length field (data length plus 2 bytes for CRC16).
  - The actual data.
  - A 2-byte CRC16 for error detection.

**Modulation & Demodulation:**  
- The complete payload is preprocessed by prepending a preamble ("EDTI") and a header that specifies the payload length.  
- By default, bits are modulated into audio tones using dedicated frequencies for ‘0’ and ‘1’.  
- In FDM mode, the bitstream is distributed across multiple channels—each channel is allocated a pair of frequencies dynamically. Channel allocation is performed automatically (unless manually specified), and the script warns if the user‑requested channels cannot be fully allocated.
- Demodulation reverses this process, recovering the bitstream for header processing and payload extraction.

**HMAC Signing & Verification:**  
- When a passphrase is provided, HMAC (SHA‑256) signing is applied to the data to guarantee integrity.  
- During decoding, if the HMAC check fails, the user is prompted on whether to continue; however, continuing is not recommended for security‑sensitive data.

3. New Features & Enhancements

- **FDM Mode:**
  - *Automatic/Manual Channel Selection:* When enabled with the `--fdm` flag, the script supports multi‑channel audio encoding. The number of channels can be auto‑detected (by trying 1 to 16 channels until the payload fits) or manually set using the `--fdm-channels` option.
  - *Frequency Allocation:* The script allocates unique frequency pairs for each channel while ensuring a minimum gap between frequencies.
  
- **Memory Management & Chunking:**
  - Before modulation in FDM mode, an estimate of the memory requirement is calculated. If this requirement exceeds a specified threshold (with warnings for >2GB and critical prompts for >8GB), the user is notified.
  - To minimize memory overhead, audio processing is performed in chunks.

- **Multi-file Output:**
  - When the encoded WAV file would exceed the 4GB limit, the payload is split across several files.  
  - Each output file has an embedded header with part information (e.g., part number and total parts) to assist in correct reassembly during decoding.

4. Usage Details

**Encoding Process:**  
- Build the payload from the provided text and/or image.  
- Apply a header (preamble + payload length) and append an HMAC signature if a passphrase is provided.
- Convert the payload into a bitstream and modulate it into audio tones.  
- If FDM mode is enabled, the bitstream is split across the desired channels; otherwise, the default single‑channel mode is used.
- The output WAV file is either produced as a single file or split into multiple parts if the size limit is exceeded.

**Decoding Process:**  
- Read the WAV file(s) and, if necessary, reassemble multiple parts using the embedded part info.  
- Demodulate the audio to extract the bitstream.
- Locate the preamble and parse the header to ascertain the payload length.
- Verify the HMAC signature if a passphrase is provided and then parse each segment (text/image).
- Reconstruct text with any error replacements and restore images using image processing libraries.

5. Troubleshooting

- **WAV File Size:**  
  Ensure that if your payload is large (especially when using images or FDM), you review the multi‑file output. The script automatically splits files if the 4GB limit is approached.
  
- **Memory Usage:**  
  When using FDM mode, pay attention to memory warnings. If prompted, consider reducing the payload size or adjusting the number of FDM channels.
  
- **HMAC Verification:**  
  Use the correct passphrase for both encoding and decoding to avoid integrity check failures.
  
- **CRC Errors:**  
  CRC mismatches may be logged if any segment data is partially corrupted. In such cases, partial or placeholder data may be recovered.

6. Final Notes

This manual applies to EDTI version 6.1 Beta. Future updates may introduce additional features or further modifications.
For contributions, bug reports, or feature requests, please submit issues or pull requests on the GitHub repository.

Support & Contributions:  
Contributions, bug reports, and feature requests are welcome. Please open an issue report or submit a pull request on GitHub.
