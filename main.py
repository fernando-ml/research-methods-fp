import numpy as np
import time
import matplotlib.pyplot as plt

def quicksort_numpy(arr: np.ndarray) -> None:
    if len(arr) <= 1:
        return

    pivot = arr[0]
    left = arr[arr < pivot]
    equal = arr[arr == pivot]
    right = arr[arr > pivot]

    quicksort_numpy(left)
    quicksort_numpy(right)

    arr[:] = np.concatenate((left, equal, right))

def mergesort_numpy(arr: np.ndarray) -> np.ndarray:
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = mergesort_numpy(arr[:mid])
    right = mergesort_numpy(arr[mid:])

    return merge_numpy(left, right)

def merge_numpy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(len(left) + len(right), dtype=left.dtype)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result[k] = left[i]
            i += 1
        else:
            result[k] = right[j]
            j += 1
        k += 1

    result[k:] = left[i:] if i < len(left) else right[j:]
    return result

def shellsort_numpy(arr: np.ndarray) -> None:
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

def bubblesort_numpy(arr: np.ndarray) -> None:
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break

def generate_data_numpy(size: int, data_type: str) -> np.ndarray:
    if data_type == "random":
        return np.random.randint(0, 10000, size)
    elif data_type == "nearly_sorted":
        arr = np.arange(size)
        swaps = size // 20
        for _ in range(swaps):
            i, j = np.random.randint(0, size, 2)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    else:  # reverse_sorted
        return np.arange(size - 1, -1, -1)

def run_numpy_experiments():
    sizes = [100, 200, 500, 1000, 1300, 1500,2000]
    dataset_types = ["random", "nearly_sorted", "reverse_sorted"]
    algorithms = {
        "Quicksort": quicksort_numpy,
        "Mergesort": mergesort_numpy,
        "Shellsort": shellsort_numpy,
        "Bubblesort": bubblesort_numpy
    }

    results = {}
    for size in sizes:
        for dtype in dataset_types:
            for algo_name, algo_func in algorithms.items():
                data = generate_data_numpy(size, dtype)
                data_copy = data.copy()

                start_time = time.perf_counter_ns()
                if algo_name == "Mergesort":
                    algo_func(data_copy)
                else:
                    algo_func(data_copy)
                end_time = time.perf_counter_ns()

                time_taken = end_time - start_time

                key = f"{size}_{dtype}"
                if key not in results:
                    results[key] = {}
                results[key][algo_name] = time_taken

    return results

def plot_results(results):
    dataset_types = ["random", "nearly_sorted", "reverse_sorted"]
    fig, axes = plt.subplots(len(dataset_types), 1, figsize=(6, 3 * len(dataset_types)), sharex=True) # Single column, multiple rows

    markers = {
        "Quicksort": "*", 
        "Mergesort": "o",
        "Shellsort": "s",
        "Bubblesort": "D"
    }

    colors = {
        "Quicksort": "blue",  
        "Mergesort": "green",
        "Shellsort": "orange",
        "Bubblesort": "red"
    }

    for i, dtype in enumerate(dataset_types):
        ax = axes[i]
        for algo_name in ["Quicksort", "Mergesort", "Shellsort", "Bubblesort"]:
            x_values = []
            y_values = []
            for size in [100, 500, 1000, 1500, 2000]: 
                key = f"{size}_{dtype}"
                if key in results and algo_name in results[key]:
                    x_values.append(size)
                    y_values.append(results[key][algo_name])

            ax.plot(x_values, y_values, label=f"{algo_name}", marker=markers[algo_name], color=colors[algo_name], markersize=8, linewidth=2)

        ax.set_xlabel("Input Size", fontsize=10)  
        ax.set_ylabel("Execution Time (nanoseconds)", fontsize=10)  
        ax.set_title(f"Data Type: {dtype}", fontsize=12)  
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.set_yscale('log') 
    plt.tight_layout()
    plt.savefig("sorting_algorithm_comparison.pdf")
    plt.show()

results = run_numpy_experiments()
plot_results(results)