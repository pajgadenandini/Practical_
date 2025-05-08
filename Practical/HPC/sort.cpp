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
void merge_sort_parallel(vector<int>& arr, int l, int r);

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


































































































// sudo apt update
// sudo apt install g++
// cd /path/to/your/code
// g++ -o sort sort.cpp -fopenmp
// ./sort

// application bubble -> educational settings,debugguing O[n2]
// merge -> linked list,large datsets O[nlogn]







// Enter the number of elements: 20
// Enter the elements:
// 42 87 19 56 33 78 65 92 11 49 8 71 53 28 94 36 50 61 24 77

// Choose sorting algorithm:
// 1. Sequential Bubble Sort
// 2. Parallel Bubble Sort
// 3. Sequential Merge Sort
// 4. Parallel Merge Sort
// 5. Exit
// Enter your choice (1-5): 1
// Sorted array (sequential bubble sort): 8 11 19 24 28 33 36 42 49 50 53 56 61 65 71 77 78 87 92 94
// Sequential Bubble Sort Time: 0 seconds

// Choose sorting algorithm:
// 1. Sequential Bubble Sort
// 2. Parallel Bubble Sort
// 3. Sequential Merge Sort
// 4. Parallel Merge Sort
// 5. Exit
// Enter your choice (1-5): 2
// Sorted array (parallel bubble sort): 8 11 19 24 28 33 36 42 49 50 53 56 61 65 71 77 78 87 92 94
// Parallel Bubble Sort Time: 0.00200009 seconds

// Choose sorting algorithm:
// 1. Sequential Bubble Sort
// 2. Parallel Bubble Sort
// 3. Sequential Merge Sort
// 4. Parallel Merge Sort
// 5. Exit
// Enter your choice (1-5): 3
// Sorted array (sequential merge sort): 8 11 19 24 28 33 36 42 49 50 53 56 61 65 71 77 78 87 92 94
// Sequential Merge Sort Time: 0 seconds

// Choose sorting algorithm:
// 1. Sequential Bubble Sort
// 2. Parallel Bubble Sort
// 3. Sequential Merge Sort
// 4. Parallel Merge Sort
// 5. Exit
// Enter your choice (1-5): 4
// Sorted array (parallel merge sort): 8 11 19 24 28 33 36 42 49 50 53 56 61 65 71 77 78 87 92 94
// Parallel Merge Sort Time: 0.00100017 seconds

// Choose sorting algorithm:
// 1. Sequential Bubble Sort
// 2. Parallel Bubble Sort
// 3. Sequential Merge Sort
// 4. Parallel Merge Sort
// 5. Exit
// Enter your choice (1-5):



/*Functions in the Code:
bubble_sort_sequential:

This function implements the standard sequential bubble sort algorithm. It loops through the array and compares adjacent elements, swapping them if they are in the wrong order. This process repeats until the array is sorted.

bubble_sort_parallel:

This is a parallelized version of the bubble sort algorithm. It uses odd-even transposition to make the process faster.

It runs two loops in parallel:

The first loop compares and swaps elements at odd indices.

The second loop compares and swaps elements at even indices.

This improves performance by reducing the time spent on sorting.

merge and merge_sort:

These functions implement merge sort, a divide-and-conquer algorithm.

merge function: Merges two sorted subarrays into one sorted array.

merge_sort function: Recursively divides the array into two halves and calls the merge function to combine the sorted halves.

parallel_merge_sort and merge_sort_parallel:

These functions are used to implement the parallel version of merge sort.

They divide the array into smaller subarrays recursively (using tasks in OpenMP) and merge them in parallel.

#pragma omp task tells OpenMP to execute the merge sort on separate tasks in parallel.

#pragma omp parallel for runs the loop in parallel across multiple threads.

#pragma omp task defines a parallel task.

#pragma omp taskwait ensures that all tasks are completed before continuing.*/