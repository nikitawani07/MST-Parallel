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
	set->canonicalElements = (int*) malloc(elements * sizeof(int));
	memset(set->canonicalElements, UNSET_ELEMENT, elements * sizeof(int));
	set->rank = (int*) calloc(elements, sizeof(int));
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
	else if (set->rank[root1] < set->rank[root2]) {
		set->canonicalElements[root1] = root2;
	} 
	else if (set->rank[root1] > set->rank[root2]) {
		set->canonicalElements[root2] = root1;
	} 
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

// merge sorted lists, start and end are inclusive
void merge(int* edgeList, const int start, const int end, const int pivot) {
	int length = end - start + 1;
	int* working = (int*) malloc(length * 3 * sizeof(int));

	// copy first part
	memcpy(working, &edgeList[start * 3],(pivot - start + 1) * 3 * sizeof(int));

	// copy second part reverse to simpify merge
	int workingEnd = end + pivot - start + 1;
	for (int i = pivot + 1; i <= end; i++) {
		copyEdge(&working[(workingEnd - i) * 3],&edgeList[i * 3]);
	}

	int left = 0;
	int right = end - start;
	for (int k = start; k <= end; k++) {
		if (working[right * 3 + 2]< working[left * 3 + 2]) {
			copyEdge(&edgeList[k * 3],&working[right * 3]);
			right--;
		} else {
			copyEdge(&edgeList[k * 3],&working[left * 3]);
			left++;
		}
	}

	// clean up
	free(working);
}

//sort the edge list using merge sort, start and end are inclusive
void mergeSort(int* edgeList, const int start, const int end) {
	if (start != end) {
		// recursively divide the list in two parts and sort them
		int pivot = (start + end) / 2;
		mergeSort(edgeList, start, pivot);
		mergeSort(edgeList, pivot + 1, end);

		merge(edgeList, start, end, pivot);
	}
}

// sort the edges of the graph in parallel with mergesort in parallel
void sort(WeightedGraph* graph) {
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	bool parallel = size != 1;

	// send number of elements
	int elements;
	if (rank == 0) {
		elements = graph->edges;
		MPI_Bcast(&elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} 
	else {
		MPI_Bcast(&elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}

	// scatter the edges to sort
	int elementsPart = (elements + size - 1) / size;
	int* edgeListPart = (int*) malloc(elementsPart * 3 * sizeof(int));
	if (parallel) {
		scatterEdgeList(graph->edgeList, edgeListPart, elements, &elementsPart);
	} else {
		edgeListPart = graph->edgeList;
	}

	// sort the part
	mergeSort(edgeListPart, 0, elementsPart - 1);

	if (parallel) {
		// merge all parts
		int from;
		int to;
		int elementsRecieved;
		for (int step = 1; step < size; step *= 2) {
			if (rank % (2 * step) == 0) {
				from = rank + step;
				if (from < size) {
					MPI_Recv(&elementsRecieved, 1, MPI_INT, from, 0,MPI_COMM_WORLD, &status);
					edgeListPart = realloc(edgeListPart,(elementsPart + elementsRecieved) * 3* sizeof(int));
					MPI_Recv(&edgeListPart[elementsPart * 3],elementsRecieved * 3,MPI_INT, from, 0, MPI_COMM_WORLD, &status);
					merge(edgeListPart, 0, elementsPart + elementsRecieved - 1,	elementsPart - 1);
					elementsPart += elementsRecieved;
				}
			} 
			else if (rank % step == 0) {
				to = rank - step;
				MPI_Send(&elementsPart, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
				MPI_Send(edgeListPart, elementsPart * 3, MPI_INT, to,0,	MPI_COMM_WORLD);
			}
		}

		// edgeListPart is the new edgeList of the graph, cleanup other memory
		if (rank == 0) {
			free(graph->edgeList);
			graph->edgeList = edgeListPart;
		} else {
			free(edgeListPart);
		}
	} else {
		graph->edgeList = edgeListPart;
	}
}

// find a MST of the graph using Kruskal's algorithm
void mstKruskal(WeightedGraph* graph, WeightedGraph* mst) {
	// create needed data structures
	Set* set = &(Set ) { .elements = 0, .canonicalElements = NULL, .rank =NULL };
	newSet(set, graph->vertices);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// sort the edges of the graph
	sort(graph);

	if (rank == 0) {
		// add edges to the MST
		int currentEdge = 0;
		for (int edgesMST = 0;edgesMST < graph->vertices - 1 || currentEdge < graph->edges;) {
			// check for loops if edge would be inserted
			int canonicalElementFrom = findSet(set,	graph->edgeList[currentEdge * 3]);
			int canonicalElementTo = findSet(set,graph->edgeList[currentEdge * 3 + 1]);
			if (canonicalElementFrom != canonicalElementTo) {
				// add edge to MST
				copyEdge(&mst->edgeList[edgesMST * 3],&graph->edgeList[currentEdge * 3]);
				unionSet(set, canonicalElementFrom, canonicalElementTo);
				edgesMST++;
			}
			currentEdge++;
		}
	}

	// clean up
	deleteSet(set);
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
	// use Kruskal's algorithm
	mstKruskal(graph, mst);
	

	if (rank == 0) {

		// print the edges of the MST
		printf("Minimum Spanning Tree (Kruskal):\n");
		printWeightedGraph(mst);

		unsigned long weightMST = 0;
		for (int i = 0; i < mst->edges; i++) {
			weightMST += mst->edgeList[i * 3 + 2];
		}
		printf("MST weight: %lu\n", weightMST);

		// cleanup
		deleteWeightedGraph(graph);
		deleteWeightedGraph(mst);

		printf("Time elapsed: %f s\n", MPI_Wtime() - start);
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}