# utils/common.py
def text_to_bitstring(text: str) -> str:
    return ''.join(f"{byte:08b}" for byte in text.encode('utf-8'))

def bitstring_to_text(bitstring: str) -> str:
    bytes_list = [bitstring[i:i+8] for i in range(0, len(bitstring), 8)]
    byte_values = [int(b, 2) for b in bytes_list if len(b) == 8]
    return bytes(byte_values).decode('utf-8', errors='ignore')
