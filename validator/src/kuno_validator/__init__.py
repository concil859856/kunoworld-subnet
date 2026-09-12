"""KunoWorld validator.

Each round it:
  1. challenges every enclave with a fresh nonce and verifies the attestation itself;
  2. submits canary jobs through the normal encrypted path and checks the results;
  3. scores miners from the public receipt ledger (verified video compute units,
     split across model families by the owner-signed switch, gated on attestation
     and reliability);
  4. sets weights on the subnet.
"""

__version__ = "0.1.0"
