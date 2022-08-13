from asyncio import as_completed
import numpy as np
import concurrent.futures

from typing import List

from hausdorff_latest import directed_hausdorff_distances_separate


def compute_elements_per_worker(total_elements: int, n_workers: int) -> int:
    return int(np.ceil(total_elements / n_workers))

def compute_slices(X: np.ndarray, n_workers: int) -> List[slice]:
    elements_per_worker = compute_elements_per_worker(X.shape[0], n_workers)
    slices = [
        slice(i*elements_per_worker, (i+1)*elements_per_worker)
        for i in range(n_workers)
    ]
    return slices

def compute(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    distances = directed_hausdorff_distances_separate(X, Y)
    return distances


def parallel_compute_hausdorff(X: np.ndarray, Y: np.ndarray, n_workers: int):
    distances = np.zeros(X.shape[0])
    slices = compute_slices(X, n_workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:

        future_to_slice = {
            executor.submit(compute, X=X[X_slice], Y=Y) : X_slice
            for X_slice in slices
        }

        for future in concurrent.futures.as_completed(future_to_slice):
            X_slice = future_to_slice[future]
            try:
                subset_distances = future.result()
            except Exception as exc:
                print(f'computation for slice {X_slice} failed with {exc}')
            else:
                distances[X_slice] = subset_distances
    return distances



if __name__ == '__main__':
    card_X = 1000
    card_Y = 1000
    
    X = np.random.randint(0, 100, size=(card_X, 3))
    Y = np.random.randint(0, 100, size=(card_Y, 3))

    N_WORKERS = 8
    result = parallel_compute_hausdorff(X, X, N_WORKERS)
    print(result.min())
    print(result.max())


