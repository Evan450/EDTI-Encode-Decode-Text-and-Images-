EDTI Quick-Start Guide (v5.7b)

1. Installation and Setup

Clone the Repository:
git clone https://github.com/Evan450/EDTI-Encode-Decode-Text-and-Images-.git
cd EDTI-Encode-Decode-Text-and-Images-.git

Dependencies:
Python 3

numpy, Pillow (for image processing), and standard libraries (wave, struct, hmac, etc.)

Optional: argparse (for command-line interface)

Install via:
pip install numpy pillow

Configuration:
The default settings (e.g., sample rate, bit duration, preamble) are defined in the script. These can be adjusted by editing the CONFIG dictionary.

2. Running the Script

Encoding Data:
To encode text and/or an image into a WAV file, run:

python EDTI.py --encode --text "[You'r message (without brakets)]" --image [Image Path] --output output.wav --passphrase "[passphrase]"

Decoding Data:
To decode a WAV file back into its constituent payload, run:

python EDTI.py --decode --input output.wav --passphrase "[passphrase]"

Fixes in v5.7b:
Corrected an HMAC bug to ensure reliable signature verification.

Improved error handling during HMAC verification and payload reconstruction.

Final Notes:
Versioning: This manual applies to EDTI version 5.7b. Future updates may introduce additional features or modifications.

Support & Contributions: Contributions, bug reports, and feature requests are welcome. Please open an issue report or submit a pull request on GitHub.
