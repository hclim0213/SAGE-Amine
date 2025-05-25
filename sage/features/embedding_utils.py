"""
Copyright (c) 2024 Hocheol Lim.
"""
PAD = " "
BEGIN = "j"
END = "\n"

char_to_int = {
            PAD: 0,
            BEGIN: 1,
            END: 2,
            "#": 20,
            "%": 22,
            "(": 25,
            ")": 24,
            "+": 26,
            "-": 27,
            ".": 30,
            "0": 32,
            "1": 31,
            "2": 34,
            "3": 33,
            "4": 36,
            "5": 35,
            "6": 38,
            "7": 37,
            "8": 40,
            "9": 39,
            "=": 41,
            "A": 7,
            "B": 11,
            "C": 19,
            "F": 4,
            "H": 6,
            "I": 5,
            "N": 10,
            "O": 9,
            "P": 12,
            "S": 13,
            "X": 15,
            "Y": 14,
            "Z": 3,
            "[": 16,
            "]": 18,
            "b": 21,
            "c": 8,
            "n": 17,
            "o": 29,
            "p": 23,
            "s": 28,
            "@": 42,
            "R": 43,
            "/": 44,
            "\\": 45,
            "E": 46,
        }

encode_dict = {"Br": "Y", "Cl": "X", "Si": "A", "Se": "Z", "@@": "R", "se": "E"}
decode_dict = {v: k for k, v in encode_dict.items()}

int_to_char = {value:key for key, value in char_to_int.items()}

def smiles_encode(smiles: str) -> str:
    temp_smiles = smiles
    for symbol, token in encode_dict.items():
        temp_smiles = temp_smiles.replace(symbol, token)
    return temp_smiles

def smiles_decode(smiles: str) -> str:
    temp_smiles = smiles
    for symbol, token in decode_dict.items():
        temp_smiles = temp_smiles.replace(symbol, token)
    return temp_smiles

def get_char_to_int():
    """
    Get the lookup table (for easy import)
    """
    return char_to_int

def get_int_to_char():
    """
    Get the lookup table (for easy import)
    """
    return int_to_char
    
def smiles_char_to_int(s, start_end=True):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    
    s = smiles_encode(s)
    
    if start_end:
        return [1] + [char_to_int[a] for a in s] + [2]
    else:
        return [char_to_int[a] for a in s]

def smiles_int_to_char(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    
    s = "".join([int_to_char[i] for i in s if i not in [char_to_int[PAD], char_to_int[BEGIN], char_to_int[END]]])
    return smiles_decode(s)
