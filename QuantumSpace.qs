namespace QuantumSpace {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Diagnostics;

    operation SuperpositionQubit(theta : Double, phase : Double) : Result {
        use q = Qubit();  
        Rx(theta, q);
        Ry(phase, q);      
        return M(q);           
    }

    operation QuantumRegister(nq :Int, thetas : Double[], phases : Double[]): Result[] {
        use qubits = Qubit[nq];
        ApplyToEach(H, qubits);
        return ForEach(MResetZ, qubits);
    }

    operation QuantumFourierTransform(n: Int, thetas : Double[], phases : Double[]) :Result[] {
        mutable output = [Zero, size=n];
        mutable stage_degree_fraction = 2.0;

        use quantum_register = Qubit[n];

        for i in 0 .. Length(quantum_register) - 1 {
            Rx(thetas[i], quantum_register[i]);
            Ry(phases[i], quantum_register[i]); 
            for k in i+1 .. Length(quantum_register) - 1 {
                for fr in 0 .. (k-i-2){
                    set stage_degree_fraction = stage_degree_fraction * 2.0;
                }
                Controlled R1([quantum_register[k]], (PI()/stage_degree_fraction, quantum_register[i]));
                set stage_degree_fraction = 2.0;
            }
        }        

        for i in 0 .. n/2 - 1 {
            SWAP(quantum_register[i], quantum_register[n-1]);
        }

        for i in IndexRange(quantum_register){
            set output w/= i <- M(quantum_register[i]);
        }
        
        ResetAll(quantum_register);

        return output;
    }
}
