#!/usr/bin/env python3
"""
Multi-Bank PDF Statement Processor
Modular architecture for supporting multiple bank formats
"""

from .base_processor import BaseProcessor
from .mbb_processor import MBBProcessor
from .pbb_processor import PBBProcessor
from .hlb_processor import HLBProcessor
from .rhb_processor import RHBProcessor
from .ocbc_processor import OCBCProcessor
from .uob_processor import UOBProcessor

# Bank processor registry
BANK_PROCESSORS = {
    'MBB': MBBProcessor,
    'PBB': PBBProcessor,
    'HLB': HLBProcessor,
    'RHB': RHBProcessor,
    'OCBC': OCBCProcessor,
    'UOB': UOBProcessor
}

def get_processor(bank_code: str) -> BaseProcessor:
    """Get the appropriate processor for the given bank code"""
    if bank_code not in BANK_PROCESSORS:
        raise ValueError(f"Unsupported bank code: {bank_code}. Supported banks: {list(BANK_PROCESSORS.keys())}")
    
    return BANK_PROCESSORS[bank_code]()

def get_supported_banks() -> list:
    """Get list of supported bank codes"""
    return list(BANK_PROCESSORS.keys()) 