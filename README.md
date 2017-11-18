# Cuda-Matrix-Multiplication
Relevant Code Files : <br/>
cuda_mmult_kernel.cu - <br/>
<br/>
Contains three different implementations of Cuda Matrix Multiplication Kernel: <br/>
1.) Naive implementation using global memory <br/>
2.) Tiled Shared memory implemetation with coalesced accesses <br/>
3.) Tiled Shared memory implementation with prefetching <br/>
<br/>
cuda_mmult.cu - <br/>
Compares the different implementations <br/>
<br/>

To run: <br/>
```
1.) make 
2.) ./out <size of matrix> <number of repeats>
```
