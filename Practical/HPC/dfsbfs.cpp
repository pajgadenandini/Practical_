#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

const int MAXN = 100005;
vector<int> adj[MAXN];
bool visited_dfs[MAXN];
bool visited_bfs[MAXN];

// Parallel DFS using OpenMP tasks safely
void parallelDFS(int node) {
    bool alreadyVisited = false;

    // Safely check and mark visited
    #pragma omp critical
    {
        if (visited_dfs[node]) {
            alreadyVisited = true;
        } else {
            visited_dfs[node] = true;
            cout << node << " ";
        }
    }

    if (alreadyVisited) return;

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int neighbor : adj[node]) {
                #pragma omp task firstprivate(neighbor)
                {
                    parallelDFS(neighbor);
                }
            }
        }
    }
}


// Parallel BFS using OpenMP
void parallelBFS(int source) {
    queue<int> q;
    q.push(source);
    visited_bfs[source] = true;

    cout << "BFS Traversal: ";

    while (!q.empty()) {
        int level_size;

        #pragma omp critical
        {
            level_size = q.size();
        }

        vector<int> curr_level;
        for (int i = 0; i < level_size; i++) {
            #pragma omp critical
            {
                if (!q.empty()) {
                    curr_level.push_back(q.front());
                    q.pop();
                }
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < curr_level.size(); i++) {
            int curr = curr_level[i];
            #pragma omp critical
            cout << curr << " ";

            for (int neighbor : adj[curr]) {
                #pragma omp critical
                {
                    if (!visited_bfs[neighbor]) {
                        visited_bfs[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }

    cout << endl;
}

int main() {
    int n, m;
    cout << "Enter number of nodes and edges: ";
    cin >> n >> m;

    cout << "Enter edges (u v):" << endl;
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int start;
    cout << "Enter start node: ";
    cin >> start;

    int choice;
    cout << "Choose traversal:\n1. Parallel BFS\n2. Parallel DFS\nEnter choice: ";
    cin >> choice;

    if (choice == 1) {
        fill(visited_bfs, visited_bfs + MAXN, false);
        parallelBFS(start);
    } else if (choice == 2) {
        fill(visited_dfs, visited_dfs + MAXN, false);
        cout << "DFS Traversal: ";
        parallelDFS(start);
        cout << endl;
    } else {
        cout << "Invalid choice.\n";
    }

    return 0;
}







































































/*g++ -o dfsbfs dfsbfs.cpp -fopenmp

C:\Users\Nandini\OneDrive\Desktop\Practical\HPC>dfsbfs.exe
Enter number of nodes and edges: 5 4
Enter edges (u v):
1 2
1 3
2 3
E^C
C:\Users\Nandini\OneDrive\Desktop\Practical\HPC>dfsbfs.exe
Enter number of nodes and edges: 5 4
Enter edges (u v):
1 2
1 3
2 4
2 5
Enter start node: 1
Choose traversal:
1. Parallel BFS
2. Parallel DFS
Enter choice: 1
BFS Traversal: 1 3 2 5 4

C:\Users\Nandini\OneDrive\Desktop\Practical\HPC>dfsbfs.exe
Enter number of nodes and edges: 5 4
Enter edges (u v):
1 2
1 3
2 4
2 5
Enter start node: 1
Choose traversal:
1. Parallel BFS
2. Parallel DFS
Enter choice: 2
DFS Traversal: 1 2 3 4 5


🔍 What's happening here?
#pragma omp parallel starts a parallel region — threads are ready to run.

#pragma omp single nowait: only one thread launches tasks (don’t wait for others).

#pragma omp task: for each neighbor, create a task to visit that node.

firstprivate(neighbor): each task gets its own copy of neighbor so there's no conflict between threads.

🤔 Why use parallel DFS?
Regular DFS uses one CPU thread — slow for big graphs.

Parallel DFS:

Assigns neighbors to different threads.

Can speed up traversal significantly for large or deep graphs.

🔁 Summary
Code Part	Purpose
visited_dfs[node] = true	Track visited nodes to avoid cycles
#pragma omp critical	Avoid race conditions on shared memory (visited_dfs)
#pragma omp parallel	Launch multi-threading environment
#pragma omp task	Allow different threads to process different branches (neighbors)
firstprivate(neighbor)	Isolate task data to prevent conflict
alreadyVisited flag	Avoid return inside OpenMP block (legal structure)


queue<int> q;
q.push(source);
visited_bfs[source] = true;

cout << "BFS Traversal: ";
What this does:
Start from the source node.

Mark it as visited.

Begin printing the traversal.

We process all nodes at the current level (same distance from source).
level_size stores how many nodes are currently in the queue (the current BFS level).
#pragma omp critical makes sure only one thread reads the queue size at a time.
Spawn multiple threads — each thread processes one node in the current level.
Safely pop a node from the queue — only one thread pops at a time.
If the queue is empty unexpectedly, assign -1 as a placeholder.

queue<int> q	Stores nodes to visit (level-wise)
level_size = q.size()	Processes one BFS level at a time
#pragma omp parallel for	Processes multiple nodes from the same level in parallel
#pragma omp critical	Prevents race conditions on shared structures (queue, visited array)
q.push() and q.pop()	Add/remove nodes to/from queue safely
visited_bfs[]	Track which nodes have been visited

Feature	DFS	BFS
Data Structure	Recursive/Stack	Queue
Order	Go deep before wide	Go wide before deep
Parallelism	Use OpenMP tasks	Use parallel loop for each level
Use Case	Tree traversal, backtracking	Shortest path in unweighted graph

applications bfs -> google maps, binary trees, linkedinwifi
dfs -> backttracking problems, tree traversal*/