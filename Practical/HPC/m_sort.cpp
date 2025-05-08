#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Merge function to combine two sorted halves
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);
    // Copy data to temp arrays L[] and R[]
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    // Merge the temp arrays back into arr[l..r]
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    // Copy the remaining elements of L[], if any
    while (i < n1) arr[k++] = L[i++];
    // Copy the remaining elements of R[], if any
    while (j < n2) arr[k++] = R[j++];
}

// Recursive merge sort with OpenMP parallel tasks
void merge_sort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        
        // Recursively divide and conquer
        #pragma omp task shared(arr)
        merge_sort(arr, l, m);

        #pragma omp task shared(arr)
        merge_sort(arr, m + 1, r);
        
        // Wait for all tasks to finish before merging
        #pragma omp taskwait
        merge(arr, l, m, r);
    }
}

// Wrapper to handle parallel section
void parallel_merge_sort(vector<int>& arr) {
    #pragma omp parallel
    {
        #pragma omp single
        merge_sort(arr, 0, arr.size() - 1);
    }
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> arr(n), arr_copy(n);

    cout << "Enter the elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
        arr_copy[i] = arr[i];  // Copy for parallel sort
    }

    // Sequential merge sort timing
    double start = omp_get_wtime();  //start timer
    merge_sort(arr, 0, n - 1);       //sequential sort
    double end = omp_get_wtime();    //end timer

    cout << "Sorted array (sequential): ";
    for (int val : arr) cout << val << " ";
    cout << "\nSequential merge sort time: " << end - start << " seconds\n";

    // Parallel merge sort timing
    start = omp_get_wtime();
    parallel_merge_sort(arr_copy);
    end = omp_get_wtime();

    cout << "Sorted array (parallel): ";
    for (int val : arr_copy) cout << val << " ";
    cout << "\nParallel merge sort time: " << end - start << " seconds\n";

    return 0;
}
