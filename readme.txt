Run Serial:
cd stage0_serial

./train_gpt2

after training (weights are provided),
./inference

Run MPI:
cd stage1_mpi

mpirun -n 8 -f ../hostfile ./train_gpt2
mpirun -n 16 -f ../hostfile ./train_gpt2

after training (weights are provided),
./inference

Run CUDA:
cd stage2_cuda

./train_gpt2

after training (weights are provided),
./inference

Run MPI+CUDA:
cd stage3_mpi-cuda

mpirun -n 8 -f ../hostfile ./train_gpt2
mpirun -n 16 -f ../hostfile ./train_gpt2

after training (weights are provided),
./inference
