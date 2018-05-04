// C standard header files
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

const int UNSET_ELEMENT = -1;

typedef struct Set {
	int elements;
	int* canonicalElements;
	int* rank;
} Set;

typedef struct WeightedGraph {
	int edges;
	int vertices;
	int* edgeList;
} WeightedGraph;

//initialize and allocate memory for the members of the graph
void newWeightedGraph(WeightedGraph* graph, const int vertices, const int edges) {
	graph->edges = edges;
	graph->vertices = vertices;
	graph->edgeList = (int*) calloc(edges * 3, sizeof(int));
}

// read a previously generated maze file and store it in the graph
void readGraphFile(WeightedGraph* graph, const char inputFileName[]) {
	// open the file
	FILE* inputFile;
	const char* inputMode = "r";
	inputFile = fopen(inputFileName, inputMode);
	if (inputFile == NULL) {
		fprintf(stderr, "Couldn't open input file, exiting!\n");
		exit(EXIT_FAILURE);
	}

	int fscanfResult;

	// first line contains number of vertices and edges
	int vertices = 0;
	int edges = 0;
	fscanfResult = fscanf(inputFile, "%d %d", &vertices, &edges);
	newWeightedGraph(graph, vertices, edges);

	int from;
	int to;
	int weight;
	for (int i = 0; i < edges; i++) {
		fscanfResult = fscanf(inputFile, "%d %d %d", &from, &to, &weight);
		graph->edgeList[i * 3] = from;
		graph->edgeList[i * 3 + 1] = to;
		graph->edgeList[i * 3 + 2] = weight;

		if (fscanfResult == EOF) {
			fprintf(stderr,"Something went wrong during reading of graph file, exiting!\n");
			fclose(inputFile);
			exit(EXIT_FAILURE);
		}
	}

	fclose(inputFile);
}

// print all edges of the graph in "from to weight" format
void printWeightedGraph(const WeightedGraph* graph) {
	printf("------------------------------------------------\n");
	for (int i = 0; i < graph->edges; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%d\t", graph->edgeList[i * 3 + j]);
		}
		printf("\n");
	}
	printf("------------------------------------------------\n");
}

void newSet(Set* set, const int elements) {
	set->elements = elements;
	set->canonicalElements = (int*) malloc(elements * sizeof(int)); // maintain parent 
	memset(set->canonicalElements, UNSET_ELEMENT, elements * sizeof(int));
	set->rank = (int*) calloc(elements, sizeof(int)); //  maintain rank
}

//return the canonical element of a vertex with path compression
int findSet(const Set* set, const int vertex) {
	if (set->canonicalElements[vertex] == UNSET_ELEMENT) {
		return vertex;
	} 
	else {
		set->canonicalElements[vertex] = findSet(set,set->canonicalElements[vertex]);
		return set->canonicalElements[vertex];
	}
}

// merge the set of parent1 and parent2 with union by rank
void unionSet(Set* set, const int parent1, const int parent2) {
	int root1 = findSet(set, parent1);
	int root2 = findSet(set, parent2);

	if (root1 == root2) {
		return;
	} 
	// Attach smaller rank tree under root of high
	else if (set->rank[root1] < set->rank[root2]) {
		set->canonicalElements[root1] = root2;
	} 
	else if (set->rank[root1] > set->rank[root2]) {
		set->canonicalElements[root2] = root1;
	} 
	// If ranks are same, then make one as root and 
    // increment its rank by one
	else {
		set->canonicalElements[root1] = root2;
		set->rank[root2] = set->rank[root1] + 1;
	}
}

// copy an edge
void copyEdge(int* to, int* from) {
	memcpy(to, from, 3 * sizeof(int));
}

// scatter the edge list of a graph
void scatterEdgeList(int* edgeList, int* edgeListPart, const int elements,int* elementsPart) {
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Scatter(edgeList, *elementsPart * 3, MPI_INT, edgeListPart,	*elementsPart * 3, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == size - 1 && elements % *elementsPart != 0) {
		// number of elements and processes isn't divisible without remainder
		*elementsPart = elements % *elementsPart;
	}

	if (elements / 2 + 1 < size && elements != size) {
		if (rank == 0) {
			fprintf(stderr, "Unsupported size/process combination, exiting!\n");
		}
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}
}

// cleanup set data
void deleteSet(Set* set) {
	free(set->canonicalElements);
	free(set->rank);
}

//cleanup graph data
void deleteWeightedGraph(WeightedGraph* graph) {
	free(graph->edgeList);
}

//find a MST of the graph using Boruvka's algorithm
void mstBoruvka(const WeightedGraph* graph, WeightedGraph* mst) {
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	bool parallel = size != 1;

	// send number of edges and vertices
	int edges;
	int vertices;
	if (rank == 0) {
		edges = graph->edges;
		vertices = graph->vertices;
		MPI_Bcast(&edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast(&edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}

	// scatter the edges to search in them
	int edgesPart = (edges + size - 1) / size;
	int* edgeListPart = (int*) malloc(edgesPart * 3 * sizeof(int));
	if (parallel) {
		scatterEdgeList(graph->edgeList, edgeListPart, edges, &edgesPart);
	} 
	else {
		edgeListPart = graph->edgeList;
	}

	// create needed data structures
	Set* set = &(Set ) { .elements = 0, .canonicalElements = NULL, .rank =NULL };
	newSet(set, vertices);

	int edgesMST = 0;
	int* closestEdge = (int*) malloc(vertices * 3 * sizeof(int));
	int* closestEdgeRecieved;
	if (parallel) {
		closestEdgeRecieved = (int*) malloc(vertices * 3 * sizeof(int));
	}

	for (int i = 1; i < vertices && edgesMST < vertices - 1; i *= 2) {
		// reset all closestEdge
		for (int j = 0; j < vertices; j++) {
			closestEdge[j * 3 + 2] = INT_MAX;
		}

		// find closestEdge
		for (int j = 0; j < edgesPart; j++) {
			int* currentEdge = &edgeListPart[j * 3];
			int canonicalElements[2] = { findSet(set, currentEdge[0]), findSet(set, currentEdge[1]) };

			// eventually update closestEdge
			if (canonicalElements[0] != canonicalElements[1]) {
				for (int k = 0; k < 2; k++) {
					bool closestEdgeNotSet = closestEdge[canonicalElements[k]* 3 + 2] == INT_MAX;
					bool weightSmaller = currentEdge[2] < closestEdge[canonicalElements[k] * 3	+ 2];
					if (closestEdgeNotSet || weightSmaller) {
						copyEdge(&closestEdge[canonicalElements[k] * 3],currentEdge);
					}
				}
			}
		}

		if (parallel) {
			int from;
			int to;
			for (int step = 1; step < size; step *= 2) {
				if (rank % (2 * step) == 0) {
					from = rank + step;
					if (from < size) {
						MPI_Recv(closestEdgeRecieved, vertices * 3,	MPI_INT, from, 0, MPI_COMM_WORLD, &status);

						// combine all closestEdge parts
						for (int i = 0; i < vertices; i++) {
							int currentVertex = i * 3;
							if (closestEdgeRecieved[currentVertex + 2]< closestEdge[currentVertex + 2]) {
								copyEdge(&closestEdge[currentVertex],&closestEdgeRecieved[currentVertex]);
							}
						}
					}
				} 
				else if (rank % step == 0) {
					to = rank - step;
					MPI_Send(closestEdge, vertices * 3, MPI_INT, to,0,MPI_COMM_WORLD);
				}
			}
			// publish all closestEdge parts
			MPI_Bcast(closestEdge, vertices * 3, MPI_INT, 0,MPI_COMM_WORLD);
		}

		// add new edges to MST
		for (int j = 0; j < vertices; j++) {
			if (closestEdge[j * 3 + 2] != INT_MAX) {
				int from = closestEdge[j * 3];
				int to = closestEdge[j * 3 + 1];

				// prevent adding the same edge twice
				if (findSet(set, from) != findSet(set, to)) {
					if (rank == 0) {
						copyEdge(&mst->edgeList[edgesMST * 3],&closestEdge[j * 3]);
					}
					edgesMST++;
					unionSet(set, from, to);
				}
			}
		}
	}

	// clean up
	deleteSet(set);
	free(closestEdge);
	if (parallel) {
		free(closestEdgeRecieved);
		free(edgeListPart);
	}
}

// main program
int main(int argc, char* argv[]) {
	// MPI variables and initialization
	int rank;
	int size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	// graph Variables
	WeightedGraph* graph = &(WeightedGraph ) { .edges = 0, .vertices = 0,.edgeList = NULL };
	WeightedGraph* mst = &(WeightedGraph ) { .edges = 0, .vertices = 0,	.edgeList = NULL };

	if (rank == 0) {

		// read the maze file and store it in the graph
		readGraphFile(graph, argv[1]);

		// print the edges of the read graph
		printf("Original Graph:\n");
		printWeightedGraph(graph);

		newWeightedGraph(mst, graph->vertices, graph->vertices - 1);
	}

	double start = MPI_Wtime();
	// use Boruvka's algorithm
	mstBoruvka(graph, mst);

	if (rank == 0) {
		

		// print the edges of the MST
		printf("Minimum Spanning Tree (Boruvka):\n");
		printWeightedGraph(mst);

		unsigned long weightMST = 0;
		for (int i = 0; i < mst->edges; i++) {
			weightMST += mst->edgeList[i * 3 + 2];
		}

		printf("MST weight: %lu\n", weightMST);
		printf("Time elapsed: %f s\n", MPI_Wtime() - start);
		// cleanup
		deleteWeightedGraph(graph);
		deleteWeightedGraph(mst);
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}