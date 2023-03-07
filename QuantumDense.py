# %%
import datetime
import numpy as np
import tensorflow as tf
import qsharp

from QuantumSpace import QuantumFourierTransform
from keras.layers import Layer
from dataclasses import dataclass

# %%
@dataclass(frozen=True)
class QuantumCircuitModuleExceptionData:
    data: str


class QuantumCircuitModuleException(Exception):
    def __init__(self, exception_details):
        self.details = exception_details

    def to_string(self):
        return self.details.data


class QuantumCircuitModule:
    def __init__(self, qubits=3):
        self.qubit_num = qubits
        self.probabilities = tf.constant([[0.5] * self.qubit_num])
        self.phase_probabilities = tf.constant([1] * self.qubit_num)
        self.thetas = []
        self.phis = []
    
    def p_to_angle(self, p):
        try:
            angle = 2 * np.arccos(np.sqrt(p))
        except Exception as e:
            raise QuantumCircuitModuleException(
                        QuantumCircuitModuleExceptionData(str({f"""'timestamp': '{datetime.datetime.now().
                                         strftime("%m/%d/%Y, %H:%M:%S")}',
                                         'function': 'p_to_angle',
                                         'message': '{e.message}'"""})))
        return angle

    def superposition_qubits(self, probabilities: tf.Tensor, phases: tf.Tensor):
        try:
            reshaped_probabilities = tf.reshape(probabilities, [self.qubit_num])
            reshaped_phases = tf.reshape(phases, [self.qubit_num])
            static_probabilities = tf.get_static_value(reshaped_probabilities[:])
            static_phases = tf.get_static_value(reshaped_phases[:])


            self.thetas = []
            self.phis = []
            for ix, p in enumerate(static_probabilities):
                p = np.abs(p)
                theta = self.p_to_angle(p)
                phi = self.p_to_angle(static_phases[ix])
                self.thetas.append(theta)
                self.phis.append(phi)
        except Exception as e:
            raise QuantumCircuitModuleException(
                        QuantumCircuitModuleExceptionData(str({f"""'timestamp': '{datetime.datetime.now().
                                         strftime("%m/%d/%Y, %H:%M:%S")}',
                                         'function': 'superposition_qubits',
                                         'message': '{e.message}'"""})))

    def quantum_execute(self, probabilities, phases):
        try:
            self.superposition_qubits(probabilities, phases)

            circuit_result = ''.join(map(str, QuantumFourierTransform.simulate(n=self.qubit_num, thetas=self.thetas, phases=self.phis)))
            qubits_results = [float(x)  for x in list(circuit_result)]
            qubit_tensor_results = tf.convert_to_tensor(qubits_results, dtype=tf.float32)
        except Exception as e:
            raise QuantumCircuitModuleException(
                        QuantumCircuitModuleExceptionData(str({f"""'timestamp': '{datetime.datetime.now().
                                         strftime("%m/%d/%Y, %H:%M:%S")}',
                                         'function': 'quantum_execute',
                                         'message': '{e.message}'"""})))
        return qubit_tensor_results

class QFTLayer(Layer):
    def __init__(self, qubits=3, execute_on_AzureQ=False):
        super(QFTLayer, self).__init__()

        self.qubits = qubits        
        self.tensor_history = []
        self.execute_on_AzureQ = execute_on_AzureQ
        
        self.circuit = QuantumCircuitModule(self.qubits)

    def build(self, input_shape):
        kernel_p_initialisation = tf.random_normal_initializer()
        self.kernel_p = tf.Variable(name="kernel_p",
                                    initial_value=kernel_p_initialisation(shape=(input_shape[-1],
                                                                          self.qubits),
                                                                          dtype='float32'),
                                    trainable=True)

        kernel_phi_initialisation = tf.zeros_initializer()

        self.kernel_phi = tf.Variable(name="kernel_phi",
                                      initial_value=kernel_phi_initialisation(shape=(self.qubits,),
                                                                              dtype='float32'),
                                      trainable=True)

    def call(self, inputs):
        try:
            output = tf.matmul(inputs, self.kernel_p)

            quantum_register_output = self.circuit.quantum_execute(tf.reshape(output, [1, self.qubits]), self.kernel_phi)
            quantum_register_output = tf.reshape(tf.convert_to_tensor(quantum_register_output), (1, 1, self.qubits))
            output += (quantum_register_output - output)

        except QuantumCircuitModuleException as qex:
            raise qex
        return output


class VQNNModel(tf.keras.Model):
    def __init__(self):
        super(VQNNModel, self).__init__(name='VQNN')

        self.driver_layer = tf.keras.layers.Dense(3, activation='relu')
        self.quantum_layer = QFTLayer(2)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input_tensor):
        try:
            x = self.driver_layer(input_tensor, training=True)
            x = self.quantum_layer(x, training=True)
            x = tf.nn.relu(x)
            x = self.output_layer(x, training=True)
        except QuantumCircuitModuleException as qex:
            print(qex)
        return x
