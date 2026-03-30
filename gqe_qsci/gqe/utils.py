from collections import Counter

import cudaq

def convert_pauli_to_cudaq_spin(pauli: dict | str) -> cudaq.SpinOperatorTerm:
    if isinstance(pauli, str):
        pauli = {i: v for i, v in enumerate(pauli)}
    term = None
    for k, v in pauli.items():
        if v == 'X':
            piece = cudaq.spin.x(k)
        elif v == 'Y':
            piece = cudaq.spin.y(k)
        elif v == 'Z':
            piece = cudaq.spin.z(k)
        else:
            continue
        term = piece if term is None else (term * piece)
    return term

def get_pauli_evolution_gate_count(pauli: str) -> int:
    counter = Counter(pauli)
    nX, nY, nZ = counter.get('X', 0), counter.get('Y', 0), counter.get('Z', 0)
    w = nX + nY + nZ

    out = Counter()
    out["cx"] = 0 if w <= 1 else 2 * (w - 1)
    out["h"] = 2 * (nX + nY)
    out["s"] = nY
    out["sdg"] = nY
    out["rz"] = 1 if w >= 1 else 0
    out["total"] = sum(out[g]for g in ("cx", "h", "s", "sdg", "rz"))
    return out