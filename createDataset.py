import argparse
import json
import sys
import time
from fractions import Fraction
from math import gcd

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.synthesis import synth_qft_full
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_ibm_runtime import fake_provider

from generalModMul import controlledMultModNGate
from qanalysis import calculateTrueOrder, calculateEverything

# code from https://quantum.cloud.ibm.com/docs/en/tutorials/shors-algorithm

def a2kmodN (a, k , N):

	for _ in range (k):
		a = int (np.mod (a**2, N))
	return a


# finds the best seed that would run quickly

def bestTranspile (circuit, backend, seeds = range (16)):

	best = None
	for seed in seeds:

		c = transpile (circuit, backend, optimization_level = 1, seed_transpiler = seed)
		used = {c.find_bit (q).index for inst in c.data for q in inst.qubits}
		key = (len (used), c.num_nonlocal_gates())

		if best is None or key < best[0]:
			best = (key, c, seed)

	(nQubits, _), compiled, seed = best
	return compiled, seed


# code modified from https://quantum.cloud.ibm.com/docs/en/tutorials/shors-algorithm
# code to make the circuit

def makeCircuit (N, a, t, approxDegree = 0):

	numTarget = N.bit_length()

	# reduce a so that a2kmodN returns the correct value

	a = a % N

	bList = [a2kmodN (a, k, N) for k in range (t)]

	# prepare all the quantum and classical registers

	control = QuantumRegister (t, name = "C")
	target = QuantumRegister (numTarget, name = "T")
	work = QuantumRegister (numTarget + 1, name = "W")
	anc = QuantumRegister (1, name = "A")
	output = ClassicalRegister (t, name = "out")

	# create the circuit

	circuit = QuantumCircuit (control, target, work, anc, output)

	# add the controlledMultModNGate

	circuit.x (target[0])
	for qubit, b in zip (control, bList):

		circuit.h (qubit)
		gate, _ = controlledMultModNGate (numTarget, N, b)

		# gate qubit order is [x (n), b (n+1), ancilla (1), control (1)]

		circuit.append (gate, list (target) + list (work) + list (anc) + [qubit])

	circuit.compose (
		synth_qft_full (t, approximation_degree = approxDegree, inverse = True),
		qubits = control,
		inplace = True,
	)

	circuit.measure (control, output)
	return circuit


# code modified from https://quantum.cloud.ibm.com/docs/en/tutorials/shors-algorithm
# code to run the experiment and return the counts, compiled circuit, and times

def runExperiment (backend, circuit, shots = 1024):

	# transpile the circuit

	start = time.perf_counter()
	compiled, seed = bestTranspile (circuit, backend)
	transpileTime = time.perf_counter() - start
	print ("Transpile:", transpileTime)

	#submit the circuit

	start = time.perf_counter()
	job = backend.run (compiled, shots = shots)
	submitTime = time.perf_counter() - start
	print ("Submitted:", submitTime)

	# get results from the circuit

	start = time.perf_counter()
	counts = job.result().get_counts()
	resultTime = time.perf_counter() - start
	print ("Result:", resultTime)

	return counts, compiled, {
		"transpileTime": transpileTime,
		"submitTime": submitTime,
		"resultTime": resultTime,
		"transpileSeed":  seed
	}

# code to get the fake backend from the string name

def getFakeBackend (name):

	if not name.startswith ("Fake"):
		short = name.replace ("ibm_", "")
		name = "Fake" + short[:1].upper() + short[1:]
	cls = getattr (fake_provider, name, None)
	if cls is None:
		known = sorted (n for n in dir(fake_provider) if n.startswith ("Fake") and n[4:5].isupper())
		raise ValueError (f"unknown fake backend '{name}'. Some options: {known[:15]}")
	return cls()


# creates the noise model using generic data

def makeNoiseModel (scale):

	e1 = 2.4e-4
	e2 = 7.8e-3
	ro = 2.0e-2

	error1 = min (1, scale * e1 * 2)
	error2 = min (1, scale * e2 * 4 / 3)
	readout = min (0.5, scale * ro)

	nm = NoiseModel (basis_gates = ["sx", "x", "rz", "cx"])
	nm.add_all_qubit_quantum_error (depolarizing_error (error1, 1), ["sx", "x"])
	nm.add_all_qubit_quantum_error (depolarizing_error (error2, 2), ["cx"])
	nm.add_all_qubit_readout_error (ReadoutError ([[1 - readout, readout], [readout, 1 - readout]]))
	return nm


# code to run the experiment based off of the parameters provided and analyze
# the distribution.  returns a JSON.

def runAll (N, a, t, approx, shots, fakeName, targetNames, noiseScale):

	dataset = []

	# if the gcd of a and N is not equal to 1, then the answer is trivial

	if gcd (a, N) != 1:
		print ("gcd (a, N) != 1")
		return dataset

	# if the true order is less than or equal to 1, then the answer is trivial

	trueOrder = calculateTrueOrder (a, N)
	if trueOrder <= 1:
		print ("trueOrder <= 1")
		return dataset

	# create the circuit

	Q = 2 ** t
	circuit = makeCircuit (N, a, t, approx)
	print (f"N = {N} a = {a} t = {t} approx = {approx} shots = {shots} "
		  f"trueOrder = {trueOrder} logicalQubits = {circuit.num_qubits}")

	# get the fake backend and then run on the targets

	fakeBackend = getFakeBackend (fakeName)
	targets = []
	if "simulator" in targetNames:
		targets.append (("Simulator", AerSimulator (noise_model = makeNoiseModel(noiseScale))))
	if "fake" in targetNames:
		targets.append (("FakeBackend", fakeBackend))

	for label, backend in targets:

		print (N, a, t, approx, label, backend.name)

		try:
			counts, compiled, timings = runExperiment (backend, circuit, shots)
		except Exception as e:
			print (f"{label} failed: {e}")
			continue

		dataset.append ({
			"N":  N,
			"a":  a,
			"t":  t,
			"Q":  Q,
			"approxDegree":  approx,
			"shots":  shots,
			"target":  label,
			"backend":  backend.name,
			"counts":  counts,
			"trueOrder":  trueOrder,
			"noiseScale":  noiseScale,
			**timings,
			**calculateEverything (counts, Q, N, a, trueOrder, compiled),
		})

	return dataset


# the default json types

def _jsonDefault (o):

	if isinstance (o, np.integer):
		return int (o)
	if isinstance (o, np.floating):
		return float (o)
	if isinstance (o, np.ndarray):
		return o.tolist()
	if isinstance (o, (set, frozenset)):
		return sorted (o)
	raise TypeError (f"not JSON serializable: {type (o)}")


def main():

	p = argparse.ArgumentParser (description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument ("--N", type=int, required = True, help = "number to factor (odd, composite)")
	p.add_argument ("--a", type=int, required = True, help = "base, coprime to N")
	p.add_argument ("--t", type=int, required = True, help = "number of counting qubits")
	p.add_argument ("--approx", type = int, default = 0, help = "QFT approximation degree (default 0)")
	p.add_argument ("--shots", type = int, default = 1024)
	p.add_argument ("--targets", nargs = "+", choices = ["simulator", "fake"], default = ["simulator"])
	p.add_argument ("--fake-backend", default = "FakeSherbrooke")
	p.add_argument ("--noiseScale", default = 1.0)
	args = p.parse_args()

	out = args.out or f"shor_N{args.N}_a{args.a}_t{args.t}_approx{args.approx}.json"

	dataset = runAll (args.N, args.a, args.t, args.approx, args.shots, args.fake_backend, args.targets, args.noiseScale)

	with open ("generalDataset.jsonl", "a") as f:
		for row in dataset:
			f.write (json.dumps (row, default = _jsonDefault) + "\n")
	return 0 if dataset else 1


if __name__ == "__main__":
	sys.exit (main())