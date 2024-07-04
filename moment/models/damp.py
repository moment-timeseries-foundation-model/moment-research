import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from moment.utils import MASS_V2, contains_constant_regions, nextpow2


def DAMP_2_0(
    time_series: np.ndarray,
    subsequence_length: int,
    stride: int,
    location_to_start_processing: int,
    lookahead: int = 0,
    enable_output: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """Computes DAMP of a time series.
    Website: https://sites.google.com/view/discord-aware-matrix-profile/home
    Algorithm: https://drive.google.com/file/d/1FwiLHrgoOUOTHeIXHAFgy2flQ1alRSoN/view

    Args:
        time_series (np.ndarray): Univariate time series
        subsequence_length (int): Window size
        stride (int): Window stride
        location_to_start_processing (int): Start/End index of test/train set
        lookahead (int, optional): How far to look ahead for pruning. Defaults to 0.
        enable_output (bool, optional): Print results and save plot. Defaults to True.

    Raises:
        Exception: See code.
        Description: https://docs.google.com/presentation/d/1_-LGilUJpYRbRZpitw05EgkiOZX52kRd/edit#slide=id.p11

    Returns:
        Tuple[np.ndarray, float, int]: Matrix profile, discord score and its corresponding position in the profile
    """
    assert (subsequence_length > 10) and (
        subsequence_length <= 1000
    ), "`subsequence_length` must be > 10 or <= 1000."

    # Lookahead indicates how long the algorithm has a delay
    if lookahead is None:
        lookahead = int(2 ** nextpow2(16 * subsequence_length))
    elif (lookahead != 0) and (lookahead != 2 ** nextpow2(lookahead)):
        lookahead = int(2 ** nextpow2(lookahead))

    # Handle invalid inputs
    # 1. Constant Regions
    if contains_constant_regions(
        time_series=time_series, subsequence_length=subsequence_length
    ):
        raise Exception(
            "ERROR: This dataset contains constant and/or near constant regions.\nWe define the time series with an overall variance less than 0.2 or with a constant region within its sliding window as the time series containing constant and/or near constant regions.\nSuch regions can cause both false positives and false negatives depending on how you define anomalies.\nAnd more importantly, it can also result in imaginary numbers in the calculated Left Matrix Profile, from which we cannot get the correct score value and position of the top discord.\n** The program has been terminated. **"
        )

    # 2. Location to Start Processing
    if (location_to_start_processing / subsequence_length) < 4:
        print(
            "WARNING: location_to_start_processing/subsequence_length is less than four.\nWe recommend that you allow DAMP to see at least four cycles, otherwise you may get false positives early on.\nIf you have training data from the same domain, you can prepend the training data, like this Data = [trainingdata, testdata], and call DAMP(data, S, length(trainingdata))"
        )
        if location_to_start_processing < subsequence_length:
            print(
                f"location_to_start_processing cannot be less than the subsequence length.\nlocation_to_start_processing has been set to {subsequence_length}"
            )
            location_to_start_processing = subsequence_length
        print("------------------------------------------\n\n")
    else:
        if location_to_start_processing > (len(time_series) - subsequence_length + 1):
            print(
                "WARNING: location_to_start_processing cannot be greater than length(time_series)-S+1"
            )
            location_to_start_processing = len(time_series) - subsequence_length + 1
            print(
                f"location_to_start_processing has been set to {location_to_start_processing}"
            )
            print("------------------------------------------\n\n")

    # 3. Subsequence length
    # Subsequence length recommendation based on a peak-finding algorithm TBD.

    # Initialization
    # This is a special Matrix Profile, it only looks left (backwards in time)
    left_mp = np.zeros(time_series.shape)

    # The best discord score so far
    best_so_far = -np.inf

    # A Boolean vector where 1 means execute the current iteration and 0 means skip the current iteration
    bool_vec = np.ones(len(time_series))

    # Handle the prefix to get a relatively high best so far discord score
    for i in range(
        location_to_start_processing - 1,
        location_to_start_processing + 16 * subsequence_length,
        stride,
    ):
        # Skip the current iteration if the corresponding boolean value is 0, otherwise execute the current iteration
        if not bool_vec[i]:
            left_mp[i] = left_mp[i - 1] - 1e-05
            continue

        # Use the brute force for the left Matrix Profile value
        if i + subsequence_length - 1 > len(time_series):
            break

        query = time_series[i : i + subsequence_length]
        left_mp[i] = np.amin(MASS_V2(time_series[:i], query))

        # Update the best so far discord score
        best_so_far = np.amax(left_mp)

        # If lookahead is 0, then it is a pure online algorithm with no pruning
        if lookahead != 0:
            # Perform forward MASS for pruning
            # The index at the beginning of the forward mass should be avoided in the exclusion zone
            start_of_mass = min(i + subsequence_length - 1, len(time_series))
            end_of_mass = min(start_of_mass + lookahead - 1, len(time_series))

            # The length of lookahead should be longer than that of the query
            if (end_of_mass - start_of_mass + 1) > subsequence_length:
                distance_profile = MASS_V2(
                    time_series[start_of_mass : end_of_mass + 1], query
                )

                # Find the subsequence indices less than the best so far discord score
                dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]

                # Converting indexes on distance profile to indexes on time series
                ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass

                # update the Boolean vector
                bool_vec[ts_index_less_than_BSF] = 0

    # Remaining test data except for the prefix
    for i in range(
        location_to_start_processing + 16 * subsequence_length,
        len(time_series) - subsequence_length + 1,
        stride,
    ):
        # Skip the current iteration if the corresponding boolean value is 0, otherwise execute the current iteration
        if not bool_vec[i]:
            # We subtract a very small number here to avoid the pruned subsequence having the same discord score as the real discord
            left_mp[i] = left_mp[i - 1] - 1e-05
            continue

        # Initialization for classic DAMP
        # Approximate leftMP value for the current subsequence
        approximate_distance = np.inf

        # X indicates how long a time series to look backwards
        X = int(2 ** nextpow2(8 * subsequence_length))

        # flag indicates if it is the first iteration of DAMP
        flag = 1

        # expansion_num indicates how many times the search has been expanded backward
        expansion_num = 0

        if i + subsequence_length - 1 > len(time_series):
            break

        query = time_series[i : i + subsequence_length]

        # Classic DAMP
        while approximate_distance >= best_so_far:
            # Case 1: Execute the algorithm on the time series segment farthest from the current subsequence
            # Arrived at the beginning of the time series
            if i - X + 1 + (expansion_num * subsequence_length) < 1:
                approximate_distance = np.amin(MASS_V2(x=time_series[: i + 1], y=query))
                left_mp[i] = approximate_distance

                # Update the best discord so far
                if approximate_distance > best_so_far:
                    # The current subsequence is the best discord so far
                    best_so_far = approximate_distance

                break

            if flag == 1:
                # Case 2: Execute the algorithm on the time series segment closest to the current subsequence
                flag = 0
                approximate_distance = np.amin(
                    MASS_V2(time_series[i - X + 1 : i + 1], query)
                )
            else:
                # Case 3: All other cases
                X_start = i - X + 1 + (expansion_num * subsequence_length)
                X_end = i - X // 2 + (expansion_num * subsequence_length)
                approximate_distance = np.amin(
                    MASS_V2(x=time_series[X_start : X_end + 1], y=query)
                )

            if approximate_distance < best_so_far:
                # If a value less than the current best discord score exists on the distance profile, stop searching
                left_mp[i] = approximate_distance
                break

            # Otherwise expand the search
            X *= 2
            expansion_num += 1

        # If lookahead is 0, then it is a pure online algorithm with no pruning
        if lookahead != 0:
            # Perform forward MASS for pruning
            # The index at the beginning of the forward mass should be avoided in the exclusion zone
            start_of_mass = min(i + subsequence_length, len(time_series))
            end_of_mass = min(start_of_mass + lookahead - 1, len(time_series))

            # The length of lookahead should be longer than that of the query
            if (end_of_mass - start_of_mass) > subsequence_length:
                distance_profile = MASS_V2(
                    x=time_series[start_of_mass : end_of_mass + 1], y=query
                )

                # Find the subsequence indices less than the best so far discord score
                dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]

                # Converting indexes on distance profile to indexes on time series
                ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass

                # Update the Boolean vector
                bool_vec[ts_index_less_than_BSF] = 0

    # Get pruning rate
    PV = bool_vec[
        location_to_start_processing - 1 : len(time_series) - subsequence_length + 1
    ]
    PR = (len(PV) - sum(PV)) / len(PV)

    # Get top discord
    discord_score, position = np.amax(left_mp), np.argmax(left_mp)
    print("\nResults:")
    print(f"Pruning Rate: {PR}")
    print(f"Predicted discord score/position: {discord_score} / {position}")

    # Outputs
    if enable_output:
        save_name = f"damp_{subsequence_length}_{stride}_{location_to_start_processing}_{lookahead}"

        # Save DAMP
        abs_left_mp = abs(left_mp)
        os.makedirs("./outputs", exist_ok=True)
        damp_save_path = f"./outputs/{save_name}.npy"
        np.save(damp_save_path, left_mp)

        # Create plot
        plt.figure(figsize=(30, 20))
        plt.plot(
            (time_series - np.min(time_series))
            / (np.max(time_series) - np.min(time_series))
            + 1.1,
            c="black",
        )
        plt.plot(abs_left_mp / np.max(abs_left_mp), c="blue", label="DAMP")

        os.makedirs("./figures", exist_ok=True)
        plot_save_path = f"./figures/{save_name}.png"

        plt.title(save_name, fontdict={"fontsize": 40})
        plt.xlabel("Index", fontdict={"fontsize": 40})
        plt.ylabel("Value", fontdict={"fontsize": 40})
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        # Save plot
        plt.savefig(plot_save_path)
        print(f"Saved figure in {plot_save_path}")

    return left_mp, discord_score, position


def contains_constant_regions(time_series: np.ndarray, subsequence_length: int) -> bool:
    bool_vec = False
    constant_indices = np.nonzero(np.concatenate(([1], np.diff(time_series), [1])))[0]
    constant_length = np.amax(np.diff(constant_indices))
    if (constant_length >= subsequence_length) or (np.var(time_series) < 0.2):
        bool_vec = True

    return bool_vec


def MASS_V2(x=None, y=None):
    # x is the data, y is the query
    m = len(y)
    n = len(x)

    # Compute y stats -- O(n)
    meany = np.mean(y)
    sigmay = np.std(y)

    # Compute x stats
    x_less_than_m = x[: m - 1]
    divider = np.arange(1, m, dtype=float)
    cumsum_ = x_less_than_m.cumsum()
    square_sum_less_than_m = (x_less_than_m**2).cumsum()
    mean_less_than_m = cumsum_ / divider
    std_less_than_m = np.sqrt(
        (square_sum_less_than_m - (cumsum_**2) / divider) / divider
    )

    windows = np.lib.stride_tricks.sliding_window_view(x, m)
    mean_greater_than_m = windows.mean(axis=1)
    std_greater_than_m = windows.std(axis=1)

    meanx = np.concatenate([mean_less_than_m, mean_greater_than_m])
    sigmax = np.concatenate([std_less_than_m, std_greater_than_m])

    y = y[::-1]
    y = np.concatenate((y, [0] * (n - m)))

    # The main trick of getting dot products in O(n log n) time
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    Z = np.multiply(X, Y)
    z = np.fft.ifft(Z).real

    dist = 2 * (
        m - (z[m - 1 : n] - m * meanx[m - 1 : n] * meany) / (sigmax[m - 1 : n] * sigmay)
    )
    dist = np.sqrt(dist)
    return dist


def nextpow2(x: int) -> float:
    """Computes the exponent of next higher power of 2.
    MATLAB reference: https://www.mathworks.com/help/matlab/ref/nextpow2.html

    Args:
        x (int): Integer

    Returns:
        float: Exponent of next higher power of 2
    """
    return np.ceil(np.log2(np.abs(x)))


def xcorr(
    x: np.ndarray, y: np.ndarray, n_lag: int = 3000, scale: str = "coeff"
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes autocorrelation with lag.
    Modified from https://stackoverflow.com/questions/43652911/python-normalizing-1d-cross-correlation.

    Args:
        x (np.ndarray): Univariate time series
        y (np.ndarray): Univariate time series
        n_lag (int): Lag
        scale (str, optional): Scaling method. Defaults to "coeff".

    Returns:
        Tuple[np.ndarray, np.ndarray]: Autocorrelation and lags
    """
    # Pad shorter array
    if x.size > y.size:
        pad_amount = x.size - y.size
        y = np.append(y, np.repeat(0, pad_amount))
    elif y.size > x.size:
        pad_amount = y.size - x.size
        x = np.append(x, np.repeat(0, pad_amount))

    lags = np.arange(-n_lag, n_lag + 1)
    corr = np.correlate(x, y, mode="same")
    i_center = len(corr) // 2
    corr = corr[i_center - n_lag : i_center + n_lag + 1]

    if scale == "biased":
        corr = corr / x.size
    elif scale == "unbiased":
        corr /= x.size - abs(lags)
    elif scale == "coeff":
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    return corr, lags


def findpeaks():
    raise NotImplementedError


if __name__ == "__main__":
    # Set parameters
    parser = argparse.ArgumentParser(description="Set parameters")
    parser.add_argument("--subsequence_length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--location_to_start_processing", type=int, default=400)
    parser.add_argument("--lookahead", type=int, default=None)
    parser.add_argument("--enable_output", action="store_true")
    args = parser.parse_args()

    # Load data
    time_series = np.loadtxt("data/samples/BourkeStreetMall.txt")

    # Run DAMP
    DAMP_2_0(
        time_series=time_series,
        subsequence_length=args.subsequence_length,
        stride=args.stride,
        location_to_start_processing=args.location_to_start_processing,
        lookahead=args.lookahead,
        enable_output=args.enable_output,
    )
