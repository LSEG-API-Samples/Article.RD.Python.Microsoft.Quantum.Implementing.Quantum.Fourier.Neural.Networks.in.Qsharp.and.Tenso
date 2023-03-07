# %%
import qsharp

from QuantumSpace import QuantumRegister, QuantumFourierTransform, SuperpositionQubit

print('Quantum Register:')
print(''.join(map(str, QuantumRegister.simulate(nq=2, thetas=[0.5, 0.5], phases=[0.5, 0.5]))))
# %%
print('Superposition qubit:')
print(SuperpositionQubit.simulate(theta=0.5, phase=0.5))
# %%
print('Quantum DFT:')
print(''.join(map(str, QuantumFourierTransform.simulate(n=3, thetas=[0.5, 0.5, 0.5], phases=[0.5, 0.5, 0.5]))))
# %%
