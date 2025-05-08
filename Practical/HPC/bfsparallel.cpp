#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

int main() {
    int num_vertices, num_edges, source;
    cout << "Enter number of nodes, edges and start node for BFS: ";
    cin >> num_vertices >> num_edges >> source;

    vector<vector<int>> adj(num_vertices + 1); // Adjacency list for the graph

    cout << "Enter " << num_edges << " edges (e.g., 1 2):" << endl;
    for (int i = 0; i < num_edges; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);  // Because the graph is undirected
    }

    vector<bool> visited(num_vertices + 1, false); // Visited array
    queue<int> q;   // Queue for BFS traversal

    // Start from the source node
    q.push(source);
    visited[source] = true;

    cout << "BFS Traversal: ";

    while (!q.empty()) {
        int level_size;

        // Critical section to safely get the current size of the queue
        #pragma omp critical
        {
            level_size = q.size();
        }

        // Process all nodes at the current BFS level in parallel
        #pragma omp parallel for
        for (int i = 0; i < level_size; i++) {
            int curr_vertex;

            // Critical block to safely access and modify the queue
            #pragma omp critical
            {
                if (!q.empty()) {
                    curr_vertex = q.front();
                    q.pop();
                } else {
                    curr_vertex = -1;  // If queue became empty
                }
            }

            if (curr_vertex == -1) continue;

            // Print current node (in parallel, so output might be out of order)
            #pragma omp critical
            cout << curr_vertex << " ";

            // Explore all neighbors of the current node
            for (int j = 0; j < adj[curr_vertex].size(); j++) {
                int neighbor = adj[curr_vertex][j];

                // Critical section to avoid duplicate visits and ensure queue safety
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }

    cout << endl;
    return 0;
}





















//Enter number of nodes, edges and start node for BFS: 6 7 1
// Enter 7 edges (e.g., 1 2):
// 1 2
// 1 3
// 2 4
// 2 5
// 3 5
// 4 6
// 5 6
// BFS Traversal: 1 2 3 4 5 6 



// What is #pragma omp?
// #pragma omp is a directive specifically for OpenMP (Open Multi-Processing).

// OpenMP is a library for parallel programming in C/C++ and Fortran.

// It allows your program to run parts of code on multiple CPU cores at the same time (multithreading).

// You need to compile with -fopenmp to enable OpenMP.


/*Directive	What it does
#pragma omp parallel	Starts a parallel region where multiple threads can execute
#pragma omp for	Used inside a parallel region to divide loop iterations
#pragma omp sections	Runs different blocks of code in parallel
#pragma omp single	Ensures only one thread executes that block
#pragma omp barrier	Makes all threads wait until they’ve all reached this point
#pragma omp atomic	Similar to critical, but for single simple operations like x++
#pragma omp task	Defines a task that can be run asynchronously by any thread*/