from math import log, log2
from fractions import Fraction

NON_GATES = {"measure", "barrier", "reset", "delay"}


# autocorrelation peak strength (Apeak) is defined in the first row of Table 2 of the Yang/Markidis paper

def calculateApeak(count, Q):
    numBits = int(log2(Q))
    shots = sum(count.values())
    py = [count.get(format(y, f'0{numBits}b'), 0) / shots for y in range(Q)]
    uy = 1 / Q
    qy = [p - uy for p in py]

    A = []
    for l in range(Q):
        Al = sum(qy[y] * qy[(y + l) % Q] for y in range(Q))
        A.append(Al)

    if A[0] == 0:
        return 0

    return max(A[1:]) / A[0]


# normalized entropy (Hnorm) is defined in the second row of Table 2 of the Yang/Markidis paper

def calculateHnorm (count, Q):

    shots = sum (count.values())
    py = [c / shots for c in count.values()]
    top = -sum (p * log (p) for p in py if p > 0)
    Hnorm = top / log (Q)
    return Hnorm


# r is defined in section 2 of the Yang/Markidis paper

def calculateR (y, Q, N, a):

    if y == 0:
        return None

    r0 = Fraction (y, Q).limit_denominator (N).denominator

    if r0 <= 1:
        return None

    if pow (a, r0, N) == 1:
        return r0

    return None


# m is defined in section 2 of the Yang/Markidis paper

def calculateM (count, Q, N, a):

    shots = sum (count.values())
    m = {}

    for bitstring, c in count.items():

        y = int (bitstring, 2)
        p = c / shots
        r = calculateR (y, Q, N, a)

        if r is None:
            continue

        m[r] = m.get (r, 0) + p
        
    return m


# dominant verified mass fraction (M1frac) is defined in the third row of Table 2 of the Yang/Markidis paper

def calculateM1frac (m):

    Mver = sum (m.values())
    
    if Mver == 0:
        return 0

    sortedM = sorted (m.values(), reverse = True)
    M1 = sortedM[0]

    return M1 / Mver


# verified margin fraction (DeltaVerFrac) is defined in the fourth row of Table 2 of the Yang/Markidis paper

def calculateDeltaVerFrac (m):

    Mver = sum (m.values())
    
    if Mver == 0:
        return 0

    sortedM = sorted (m.values(), reverse = True)
    M1 = sortedM[0]
    M2 = sortedM[1] if len (sortedM) > 1 else 0

    return (M1 - M2) / Mver


# Function to avoid having to calculate m twice

def calculatePostProcessingAware (count, Q, N, a):

    m = calculateM (count, Q, N, a)

    return calculateM1frac (m), calculateDeltaVerFrac (m)


# true order as defined in Section 2 of the Yang/Markidis paper

def calculateTrueOrder (a, N):

    r = 1

    while pow (a, r, N) != 1:
        r += 1

    return r


# the definition of recoverable is defined in section 2 of the Yang/Markidis paper

def calculateRecoverable (count, Q, N, a, trueOrder):

    m = calculateM (count, Q, N, a)

    if not m:
        return False

    predictedOrder = max (m, key = m.get)

    return predictedOrder == trueOrder


# filter function used for circuit depth measurement

def isGate (inst):
    return inst.operation.name not in NON_GATES


# filter function used for circuit depth measurement

def isTwoQubitGate (inst):
    return isGate (inst) and len (inst.qubits) == 2


# provides a full breakdown of the circuit by gate name

def calculateOpCounts (circuit):
    return dict (circuit.count_ops())


# calculates number of gates within the circuit

def calculateGateCounts (circuit):

    n1q = 0
    n2q = 0
    nMulti = 0

    for inst in circuit.data:
        if not isGate (inst):
            continue
        k = len (inst.qubits)
        if k == 1:
            n1q += 1
        elif k == 2:
            n2q += 1
        else:
            nMulti += 1

    return n1q, n2q, nMulti


# calculates the depth of the circuits based off of two criteria

def calculateDepths (circuit):

    depth = circuit.depth (filter_function = isGate)
    depth2q = circuit.depth (filter_function = isTwoQubitGate)
    return depth, depth2q


# calculates how many distinct physical qubits the compiled circuit touches

def calculatePhysicalQubitsUsed (circuit):

    return len ({circuit.find_bit (q).index for inst in circuit.data for q in inst.qubits})


# run all the analysis in one function

def calculateEverything (count, Q, N, a, trueOrder, circuit):

    M1frac, DeltaVerFrac = calculatePostProcessingAware (count, Q, N, a)
    depth, depth2q = calculateDepths (circuit)

    results = {
        "Apeak":  calculateApeak (count, Q),
        "Hnorm":  calculateHnorm (count, Q),
        "M1frac":  M1frac,
        "DeltaVerFrac":  DeltaVerFrac,
        "opCounts":  calculateOpCounts (circuit),
        "gateCounts":  calculateGateCounts (circuit),
        "depth":  depth,
        "depth2q":  depth2q,
        "physicalQubits":  calculatePhysicalQubitsUsed (circuit),
        "recoverable":  calculateRecoverable (count, Q, N, a, trueOrder)
    }

    return results