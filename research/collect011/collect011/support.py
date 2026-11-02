"""Bounded, value-independent support lane for explicit structured records.

This reference uses CPU BLAKE2b and a fixed bit vector; no GPU speed is claimed.
No false negatives for keys actually added since reset, barring corruption.
False positives, saturation and lack of deletion are explicit limitations.
It is not an entity extractor and is not enabled on unstructured LM tokens.
"""
import hashlib

class SupportLane:
    def __init__(self,bits=4096,hashes=4):
        if type(bits) is not int or bits<8 or bits%8 or type(hashes) is not int or not 1<=hashes<=8:
            raise ValueError('byte-aligned positive capacity and 1..8 hashes required')
        self.bits,self.hashes=bits,hashes
        self.buffer=bytearray(bits//8)

    def indices(self,key):
        if not isinstance(key,bytes) or not key:raise ValueError('canonical nonempty key bytes required')
        d=hashlib.blake2b(key,digest_size=32,person=b'COLLECT011-role').digest()
        return [int.from_bytes(d[i*4:i*4+4],'little')%self.bits for i in range(self.hashes)]

    def add(self,key):
        for i in self.indices(key):self.buffer[i//8]|=1<<(i%8)

    def contains(self,key):
        return all(self.buffer[i//8]&(1<<(i%8)) for i in self.indices(key))

    def reset(self):self.buffer[:]=bytes(len(self.buffer))
    def state_bytes(self):return len(self.buffer)
