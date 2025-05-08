#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Sequential Bubble Sort
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

// Parallel Bubble Sort using Odd-Even Transposition
void bubble_sort_parallel(vector<int>& arr) {
    bool isSorted = false;
    while (!isSorted) {
        isSorted = true;

        #pragma omp parallel for
        for (int i = 0; i < arr.size() - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }

        #pragma omp parallel for
        for (int i = 1; i < arr.size() - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
    }
}

// Merge function for Merge Sort
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

// Sequential Merge Sort
void merge_sort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        merge_sort(arr, l, m);
        merge_sort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort
void parallel_merge_sort(vector<int>& arr) {
    #pragma omp parallel
    {
        #pragma omp single
        merge_sort_parallel(arr, 0, arr.size() - 1);
    }
}

void merge_sort_parallel(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        #pragma omp task shared(arr)
        merge_sort_parallel(arr, l, m);

        #pragma omp task shared(arr)
        merge_sort_parallel(arr, m + 1, r);

        #pragma omp taskwait
        merge(arr, l, m, r);
    }
}

// Main Function
int main() {
    int n, choice;
    vector<int> arr, arr_copy;

    cout << "Enter the number of elements: ";
    cin >> n;
    arr.resize(n);
    arr_copy.resize(n);

    cout << "Enter the elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
        arr_copy[i] = arr[i];
    }

    double start, end;
    bool running = true;

    while (running) {
        cout << "\nChoose sorting algorithm:\n";
        cout << "1. Sequential Bubble Sort\n";
        cout << "2. Parallel Bubble Sort\n";
        cout << "3. Sequential Merge Sort\n";
        cout << "4. Parallel Merge Sort\n";
        cout << "5. Exit\n";
        cout << "Enter your choice (1-5): ";
        cin >> choice;

        switch (choice) {
            case 1:
                start = omp_get_wtime();
                bubble_sort_sequential(arr);
                end = omp_get_wtime();
                cout << "Sorted array (sequential bubble sort): ";
                for (int val : arr) cout << val << " ";
                cout << "\nSequential Bubble Sort Time: " << end - start << " seconds\n";
                break;

            case 2:
                start = omp_get_wtime();
                bubble_sort_parallel(arr_copy);
                end = omp_get_wtime();
                cout << "Sorted array (parallel bubble sort): ";
                for (int val : arr_copy) cout << val << " ";
                cout << "\nParallel Bubble Sort Time: " << end - start << " seconds\n";
                break;

            case 3:
                start = omp_get_wtime();
                merge_sort(arr, 0, n - 1);
                end = omp_get_wtime();
                cout << "Sorted array (sequential merge sort): ";
                for (int val : arr) cout << val << " ";
                cout << "\nSequential Merge Sort Time: " << end - start << " seconds\n";
                break;

            case 4:
                start = omp_get_wtime();
                parallel_merge_sort(arr_copy);
                end = omp_get_wtime();
                cout << "Sorted array (parallel merge sort): ";
                for (int val : arr_copy) cout << val << " ";
                cout << "\nParallel Merge Sort Time: " << end - start << " seconds\n";
                break;

            case 5:
                running = false;
                cout << "Exiting program.\n";
                break;

            default:
                cout << "Invalid choice! Please try again.\n";
        }
    }

    return 0;
}
