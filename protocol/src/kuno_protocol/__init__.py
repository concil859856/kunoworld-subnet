"""Shared wire protocol for KunoWorld.

Everything that two parties must agree on byte-for-byte lives here: the HPKE job
envelope, the chunked blob format, job/request schemas, model profiles and the
H3/LTX switch, attestation evidence, and signed receipts.
"""

__version__ = "0.1.0"
