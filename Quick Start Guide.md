EDTI Quick-Start Guide (v6.1 Beta)
===================================

1. Installation and Setup

Clone the Repository:
  git clone https://github.com/Evan450/EDTI-Encode-Decode-Text-and-Images-.git
  cd EDTI-Encode-Decode-Text-and-Images-.git

Dependencies:
  - Python 3
  - numpy, Pillow (for image processing), and the standard libraries (wave, struct, hmac, etc.)
  - Optional: argparse (for the command-line interface)

Install via:
  pip install numpy pillow

Configuration:
  - Default settings (such as sample rate, bit duration, preamble, etc.) are defined in the CONFIG dictionary inside the script.
  - Users may adjust these settings by editing the CONFIG values or via command-line options (e.g., enabling FDM mode).

2. Running the Script

**Encoding Data:**

- **Default (Single‑Channel) Mode:**
  To encode text and/or an image into a WAV file:
    python EDTI.py --encode --text "Your message here" --image [Image Path] --output output.wav --passphrase "your_passphrase"

- **FDM Mode (Multi‑Channel Encoding):**

To activate FDM mode—which uses multiple frequency channels for encoding—the `--fdm` flag must be provided. Optionally, specify the number of channels with `--fdm-channels` (set to 0 for automatic selection):
   python EDTI.py --encode --text "Your message here" --image [Image Path] --output output.wav --passphrase "your_passphrase" --fdm --fdm-channels 4
   *Note:* FDM mode also introduces memory management checks and may split the output into multiple WAV files if the size limit is exceeded.

**Decoding Data:**

- **Single-File Decoding:**

To decode a WAV file back into its original payload:
   python EDTI.py --decode --input output.wav --passphrase "your_passphrase"
 
- **Multi‑File Decoding & FDM Mode:**
If your encoding resulted in multiple parts (or if FDM mode was used), provide all the relevant WAV files as input:
   python EDTI.py --decode --input output_part1.wav output_part2.wav ... --passphrase "your_passphrase" --fdm --fdm-channels 4

3. New Features in v6.1 Beta

- **FDM (Frequency Division Multiplexing) Support:**  
Allows the bitstream to be modulated across multiple channels for more robust and scalable encoding. If enabled, the script will either automatically determine the optimal number of channels (up to 16) or use the number specified with `--fdm-channels`.

- **Memory Warning and Chunking:**  
When operating in FDM mode, the tool estimates the memory required for processing. It will issue a warning if the estimated memory exceeds 2GB and prompt for confirmation if it is critically high (beyond 8GB). Audio processing is conducted in chunks to reduce peak memory usage.

- **Multi-file Output:**  
In cases where the encoded audio exceeds the 4GB WAV file size limit, the payload will be split into several files. Each file includes a header with part information to allow seamless reassembly during decoding.

4. Final Notes
- Versioning: This Quick-Start Guide corresponds to EDTI version 6.1 Beta. Future updates may bring additional features or changes.

- Support & Contributions:
Contributions, bug reports, and feature requests are welcome. Please use the GitHub issue tracker or submit pull requests.
