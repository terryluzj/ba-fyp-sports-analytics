import numpy as np
import re


def parse_rank_info(rank):
    # Parse rank information
    try:
        return re.search(r'\d+', rank).group(0)
    except AttributeError:
        return np.nan


def get_dtw_distance(a_arr, b_arr, diff_func=lambda a, b: abs(a-b), time_window=0):
    # Calculate DTW distance
    if time_window <= 0:
        time_window = max(len(a_arr), len(b_arr))
    cost_matrix = np.empty(len(a_arr), len(b_arr))
    dist_matrix = np.empty(len(a_arr), len(b_arr))
    dist_matrix[0][0] = diff_func(a_arr[0], b_arr[0])
    cost_matrix[0][0] = cost_matrix[0][0]

    for i_idx in range(1, len(a_arr)):
        # Iterate through the first array
        cost_matrix[i_idx][0] = diff_func(a_arr[i_idx], b_arr[0])
        dist_matrix[i_idx][0] = dist_matrix[i_idx - 1, 0] + cost_matrix[i_idx, 0]

    for j_idx in range(1, len(b_arr)):
        # Iterate through the second array
        cost_matrix[0][j_idx] = diff_func(a_arr[0], b_arr[j_idx])
        dist_matrix[0][j_idx] = dist_matrix[0, j_idx] + cost_matrix[0, j_idx]

    for i_idx in range(1, len(a_arr)):
        # Build rest of the matrix in bottom-up approach
        window_start = max(1, i_idx - time_window)
        window_end = min(len(b_arr), i_idx + time_window)
        for j_idx in range(window_start, window_end):
            cost_matrix[i_idx][j_idx] = diff_func(a_arr[i_idx], b_arr[j_idx])
            dist_matrix[i_idx][j_idx] = min(dist_matrix[i_idx-1][j_idx],
                                            dist_matrix[i_idx][j_idx-1],
                                            dist_matrix[i_idx-1][j_idx-1]) + cost_matrix[i_idx][j_idx]

    return dist_matrix[len(a_arr)-1][len(b_arr)-1]
