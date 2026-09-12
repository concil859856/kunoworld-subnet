"""KunoWorld miner worker.

This process is the whole enclave application: it runs inside the Intel TDX
confidential VM with the GPUs in confidential-computing mode. The miner operator
only boots the VM; they never see keys, prompts, media or outputs.

Logging rule: never log request content, input media, or exception messages
that could contain them.
"""

__version__ = "0.1.0"
