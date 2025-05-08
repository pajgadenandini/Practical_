#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

const int MAXN = 100005;
vector<int> adj[MAXN];
bool visited[MAXN];

void dfs(int node) {
    visited[node] = true;  

    // Parallel region begins
    #pragma omp parallel  

    {
        #pragma omp single nowait  //single master thread to loop over neighbors.
        {
            for (int i = 0; i < adj[node].size(); i++) {
                int next_node = adj[node][i];

                #pragma omp task firstprivate(next_node) //For each neighbor, we create a task (parallel thread):
                {
                    if (!visited[next_node]) {
                        #pragma omp critical  //ensures only one thread modifies visited[] at a time
                        {
                            if (!visited[next_node]) {
                                dfs(next_node);
                            }
                        }
                    }
                }
            }
        }
    }
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

    int start_node;
    cout << "Enter start node for DFS: ";
    cin >> start_node;

    dfs(start_node);

    cout << "Visited Nodes in DFS: ";
    for (int i = 1; i <= n; i++) {
        if (visited[i]) cout << i << " ";
    }
    cout << endl;

    return 0;
}



























// Enter number of nodes and edges: 5 4
// Enter edges (u v):
// 1 2
// 1 3
// 3 4
// 3 5
// Enter start node for DFS: 1
// Visited Nodes in DFS: 1 2 3 4 5 
