#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Function for sequential bubble sort
void bubble_sort_sequential(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

// Function for parallel bubble sort using odd-even transposition
void bubble_sort_parallel(vector<int>& arr) {
    bool isSorted = false;
    while (!isSorted) {
        isSorted = true;

        // Odd indexed elements comparison in parallel
        #pragma omp parallel for
        for (int i = 0; i < arr.size() - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }

        // Even indexed elements comparison in parallel
        #pragma omp parallel for
        for (int i = 1; i < arr.size() - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
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

    // Sequential Bubble Sort
    double start = omp_get_wtime();
    bubble_sort_sequential(arr);
    double end = omp_get_wtime();
    cout << "Sorted array (sequential bubble sort): ";
    for (int val : arr) cout << val << " ";
    cout << "\nSequential bubble sort time: " << end - start << " seconds\n";

    // Parallel Bubble Sort
    start = omp_get_wtime();
    bubble_sort_parallel(arr_copy);
    end = omp_get_wtime();
    cout << "Sorted array (parallel bubble sort): ";
    for (int val : arr_copy) cout << val << " ";
    cout << "\nParallel bubble sort time: " << end - start << " seconds\n";

    return 0;
}
