# MST-Parallel

This project implements the algorithms of Boruvka and Kruskal for creating a minimum spanning tree (MST) of a weighted, undirected graph in C with parallelization via MPI.

Boruvka's algorithm:

1. Track components via union-find data structure with union by rank and path compression
2. Parallelized search for minimum outgoing edge

Kruskal's algorithm:

1. Sorting edges via parallelized merge sort
2. Track components via union-find data structure with union by rank and path compression

## Conclusion  

Use of Boruvka's algorithm is recommended to find the MST of a graph.

## Commands to Run

To compile:

mpicc filename

To run:

mpirun ./a.out graph1.csv -np 4
