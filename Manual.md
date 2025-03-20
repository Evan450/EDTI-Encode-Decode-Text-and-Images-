Encode Decode Text and Images Manual (v5.7b)

Note: This one specifically was an absoulte PAIN to deal with so... want to play a game? 
Just guess how many FULL reworks this took, and I'll leave a note at the end that tells you!
Also, Sorry that this isn't really a manual (as it doesn't tell you how to use it and stuff) I'm trying something new, again sorry for the inconvenince.

1. Introduction

EDTI (Encode Decode Text and Images) encodes text and images into audio using FSK modulation and decodes them back.
It implements segment-level error detection with CRC16 and uses HMAC for message integrity.
Version 5.7b fixes an HMAC bug for improved security.

2. How It Works

Payload Construction:
Constructs segments for text and images.

Each segment includes a type marker, a 4-byte length field, the data, and a 2-byte CRC16.

Modulation & Demodulation:
The full payload is prefixed with a preamble ("EDTI") and a header indicating data length.

Bits are modulated into audio tones (using distinct frequencies for 0 and 1).

The demodulation process reconstructs the bitstream to recover the payload.

HMAC Signing & Verification:
When a passphrase is provided, an HMAC signature is appended to the payload to ensure data integrity.

Error Handling:
Custom UTF-8 error handling replaces undecodable bytes with ‘#’.

In decoding, if the HMAC check fails, the user is prompted whether to continue.

3. Usage Details

Encoding Process:
Build the payload from provided text and/or image.

Prepend a header (preamble + data length) and append an HMAC signature if a passphrase is given.

Convert the payload to a bitstream and modulate it to create a WAV file.
Decoding Process:

Demodulate the audio to extract the bitstream.

Locate the preamble and read the header to determine payload length.

Verify and strip the HMAC signature (if present), then parse segments.

Reconstruct text (with hash replacements for errors) and images using PIL.

4. Troubleshooting

WAV File Size:
The script enforces a 4GB limit on WAV files. If exceeded, it attempts fallback options (e.g., forced grayscale for images).

HMAC Verification:
Ensure that the correct passphrase is provided during encoding and decoding.

CRC Errors:
If a segment’s CRC does not match, a warning is logged. The output may include partial or placeholder data

5. Final Notes:

Versioning:
This manual applies to EDTI version 5.7b. Future updates may introduce additional features or modifications.

Note: This is the end of the game! Here's the answer: FIVE. It took five total reworks to get this thing to work... I really wish, that it took less. :'(

Support & Contributions:
Contributions, bug reports, and feature requests are welcome. Please open an issue report or submit a pull request on GitHub.
