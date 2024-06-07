"""
This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see <https://www.gnu.org/licenses/>. 
"""

import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import cirq
import sympy
from cirq.contrib.svg import SVGCircuit
from IPython.display import display

"""
Generalized Quantum Rule System
"""
class OneQubitVQC(tf.keras.Model):
    
    """ 
        nInputs : Number of circuit inputs
        parameters: List of pairs (inputs, rotations) where:
           inputs: string containing values XYZ
           parameters: List of pairz (gate, angle) where:
              gate: String of length 1 with values XYZ
              angle: Rotation angle
        output_observables: String of observables in X/Y/Z to be used
    """
    def __init__(self, nInputs : int, structure : list, output_observables : str):
        super().__init__()
        self.__nInputs= nInputs
        self.__structure= structure
        self.__output_observables = output_observables
        
        # Create the circuit with One qubit
        self.__qubits = cirq.GridQubit.rect(1, 1)
        self.__circuit, self.__inSym= self.create_circuit()
        
        
        symbols = [str(symb) for symb in self.__inSym]
        self.__indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

            

        self.__empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        
        self.__observables= []
        for obs in self.__output_observables:
            if obs == 'Z':
                self.__observables.append(cirq.Z(self.__qubits[0]))
            elif obs == 'X':
                self.__observables.append(cirq.X(self.__qubits[0]))
            elif obs == 'Y':
                self.__observables.append(cirq.Y(self.__qubits[0]))
            else:
                raise Exception('OneQubitVQC.__init__: Unknown observable {}'.format(obs))
        
        self.__computation_layer = tfq.layers.ControlledPQC(self.__circuit, self.__observables) 


    def print_circuit(self):
        print('\n\nCircuit:\n')
        print(self.__circuit)
        print()
        
        
    def plot_circuit(self):
        display(SVGCircuit(self.__circuit))
        


    # Creates the circuit model
    def create_circuit(self):
        
        nInputs = self.__nInputs
        qubits= self.__qubits
        structure= self.__structure
        
        # Inputs
        inSym= sympy.symbols(f'x_(0:{nInputs})')
        inSym= np.asarray(inSym).reshape(-1)

        # Input embedding Circuit 
        circuit = cirq.Circuit()


        current_input= 0
        for inputs, rotations in structure:
            
            # input data
            for in_type in inputs:
                if in_type == 'X':
                    circuit.append( cirq.rx(inSym[current_input])(qubits[0]) )
                elif in_type == 'Y':
                    circuit.append( cirq.ry(inSym[current_input])(qubits[0]) )
                elif in_type == 'Z':
                    circuit.append( cirq.rz(inSym[current_input])(qubits[0]) )
                else:
                    raise Exception('OneQubitVQC.create_circuit: Unknown input rotation {}'.format(in_type))
                current_input+= 1

            # rotations data
            for rot_type, angle in rotations:
                if rot_type == 'X':
                    circuit.append( cirq.rx(angle)(qubits[0]) )
                elif rot_type == 'Y':
                    circuit.append( cirq.ry(angle)(qubits[0]) )
                elif rot_type == 'Z':
                    circuit.append( cirq.rz(angle)(qubits[0]) )
                else:
                    raise Exception ('OneQubitVQC.create_circuit: Unknown rotation type {}'.format(rot_type))
        
        return circuit, list(inSym.flat)


    # Get expectation of observables for input data        
    def call(self, inputs):
        
        inputs= tf.convert_to_tensor(inputs, dtype=tf.float32)
        
        # get the number of input patterns, and prepare the batch for parallel execution
        batch_dim = inputs.shape[0] # tf.gather(tf.shape(inputs), 0)
        tiled_circuits = tf.repeat(self.__empty_circuit, repeats=batch_dim)
        
        out= self.__computation_layer([tiled_circuits, inputs])
        return out        


