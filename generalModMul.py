import numpy as np
from qiskit import QuantumCircuit
from math import pi, gcd


# converts a computational basis into a Fourier basis without
# the final traditional swap.

def qftNoSwap (qc, qubits):
	
	n = len (qubits)
	for i in reversed (range (n)):
		qc.h (qubits[i])
		for j in range (i):
			qc.cp (pi / 2 ** (i - j), qubits[j], qubits[i])


# converts a Fourier basis into a computational basis without
# the final traditional swap.

def iqftNoSwap (qc, qubits):
	
	n = len (qubits)
	for i in range (n):
		for j in range (i):
			qc.cp (-pi / 2 ** (i - j), qubits[j], qubits[i])
		qc.h (qubits[i])


# controlled function to add a constant phase to phi.  by default,
# there are no controls and the constant phase is always added.
# however, if there are controls provided, then each bit of the
# control has to be 1 in order for the constant phase to be added.

def phiAddConst (qc, qubits, a, controls = None):
	
	n = len (qubits)
	a = a % (2 ** n)
	for i in range (n):
		angle = 2 * pi * a / (2 ** (i + 1))
		if controls:
			qc.mcp (angle, controls, qubits[i])
		else:
			qc.p (angle, qubits[i])


# function to calculate (b + a) mod N.

def phiAddModNCircuit (n1, a, N, numControls):
	
	total = n1 + 1 + numControls
	qc = QuantumCircuit (total, name = f"ADD_{a}_modN")

	b = list (range (n1))
	anc = n1

	controls = list (range (n1 + 1, n1 + 1 + numControls)) if numControls else None
	top = b[-1]

	# adds a, then subtracts N.  step 1 is controlled, step 2 is not

	phiAddConst (qc, b, a, controls)
	phiAddConst (qc, b, -N)

	# convert to computational basis, copy the top bit into the ancilla,
	# then convert back to Fourier basis.  if the ancilla is 1,
	# then b + a < N.
	
	iqftNoSwap (qc, b)
	qc.cx (top, anc)
	qftNoSwap (qc, b)

	# only happens if we are currently negative, which means ancilla is 1
	# brings back to positive if currently negative.
	
	phiAddConst (qc, b, N, [anc])

	# we are done with the ancilla, now we need to change it back to |0>
	# if it was changed.  we do not want to measure the ancilla, we only
	# needed it to detect whether or not the previous result was negative
	
	phiAddConst (qc, b, -a, controls)

	# convert to computational basis, use a quantum version of xor to
	# revert the ancilla back to |0>, convert back to Fourier basis
	
	iqftNoSwap (qc, b)
	qc.x (top)
	qc.cx (top, anc)
	qc.x (top)
	qftNoSwap (qc, b)

	# add a back.  a was taken away to do the ancilla clean up
	
	phiAddConst (qc, b, a, controls)
	
	return qc

# function to calculate b += a * x mod N.  uses long multiplication

def cmultModNCircuit (n, N, a, outerCtrl = True):

	n1 = n + 1
	x = list (range (n))
	b = list (range (n, n + n1))

	anc = n + n1
	ctrl = [n + n1 + 1] if outerCtrl else []

	total = n + n1 + 1 + len (ctrl)
	qc = QuantumCircuit (total, name = f"CMULT_{a}")

	# convert to Fourier basis
	
	qftNoSwap (qc, b)

	# the long multiplication loop, goes through each bit of x
	
	for i in range (n):

		# converts each bit of i into a modded phase rotation angles
		# if that bit is 1

		aa = (a * (2 ** i)) % N

		# number of control qubits for the adder

		numCtrl = len (ctrl) + 1

		# build the adder block.  uses to_instruction() due to
		# the overall structure being a nested circuit

		sub = phiAddModNCircuit (n1, aa, N, numCtrl)
		qubitMap = b + [anc] + ctrl + [x[i]]
		qc.append (sub.to_instruction(), qubitMap)

	# convert to computational basis
	
	iqftNoSwap (qc, b)

	return qc


# replacement for the rotationGate used in the specific case version
# of Shor's Algorithm.  calculates a * x mod N when ctrl is 1.

def controlledMultModNGate (n, N, a):

	# ensures that a is coprime to N

	if gcd (a, N) != 1:
		raise ValueError (f"a = {a} is not coprime to N = {N}; gcd = {gcd (a, N)}")

	n1 = n + 1
	x = list (range (n))
	b = list (range (n, n + n1))

	anc = n + n1
	ctrl = n + n1 + 1

	# the total number of qubits that the gate acts on
	
	total = ctrl + 1
	qc = QuantumCircuit (total, name = f"CU_{a}mod{N}")

	# calculates b += a * x mod N

	cmult = cmultModNCircuit (n, N, a, outerCtrl = True)
	qc.append (cmult.to_instruction(), x + b + [anc, ctrl])

	# controlled swaps x and b.  x now holds the information in b

	for i in range (n):
		qc.cswap (ctrl, x[i], b[i])

	# clear out b using the inverse of a.  required to clear out any
	# possible entangled garbage superpositions

	aInv = pow (a, -1, N)
	cmultInv = cmultModNCircuit (n, N, aInv, outerCtrl = True).inverse()
	qc.append (cmultInv.to_instruction(), x + b + [anc, ctrl])

	# qc.to_instruction() is used due to this being a nested circuit

	gate = qc.to_instruction()
	gate.name = f"CUa{a}modN{N}"

	return gate, total