for i in $(seq 0.5 0.01 1.0)
  do
    echo "Tau: $i"
    sed -i "s/tau_l = .*/tau_l = $i/g" advection_proof_spec.py
    python advection_proof_spec.py
    python rhs_op_eig.py
  done
