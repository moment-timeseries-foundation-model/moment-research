import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


def interpolate_timeseries(
    timeseries: npt.NDArray, interp_length: int = 512
) -> npt.NDArray:
    x = np.linspace(0, 1, timeseries.shape[-1])
    f = interp1d(x, timeseries)
    x_new = np.linspace(0, 1, interp_length)
    timeseries = f(x_new)
    return timeseries


def upsample_timeseries(
    timeseries: npt.NDArray,
    seq_len: int,
    sampling_type: str = "pad",
    direction: str = "backward",
    **kwargs,
) -> npt.NDArray:
    timeseries_len = len(timeseries)
    input_mask = np.ones(seq_len)

    if timeseries_len >= seq_len:
        return timeseries, input_mask

    if sampling_type == "interpolate":
        timeseries = interpolate_timeseries(timeseries, seq_len)
    elif sampling_type == "pad" and direction == "forward":
        timeseries = np.pad(timeseries, (0, seq_len - timeseries_len), **kwargs)
        input_mask[: seq_len - timeseries_len] = 0
    elif sampling_type == "pad" and direction == "backward":
        timeseries = np.pad(timeseries, (seq_len - timeseries_len, 0), **kwargs)
        input_mask[: seq_len - timeseries_len] = 0
    else:
        error_msg = "Direction must be one of 'forward' or 'backward'"
        raise ValueError(error_msg)

    assert len(timeseries) == seq_len, "Padding failed"
    return timeseries, input_mask


def downsample_timeseries(
    timeseries: npt.NDArray, seq_len: int, sampling_type: str = "interpolate"
):
    input_mask = np.ones(seq_len)
    if sampling_type == "last":
        timeseries = timeseries[:seq_len]
    elif sampling_type == "first":
        timeseries = timeseries[seq_len:]
    elif sampling_type == "random":
        idx = np.random.randint(0, timeseries.shape[0] - seq_len)
        timeseries = timeseries[idx : idx + seq_len]
    elif sampling_type == "interpolate":
        timeseries = interpolate_timeseries(timeseries, seq_len)
    elif sampling_type == "subsample":
        factor = len(timeseries) // seq_len
        timeseries = timeseries[::factor]
        timeseries, input_mask = upsample_timeseries(
            timeseries, seq_len, sampling_type="pad", direction="forward"
        )
    else:
        error_msg = "Mode must be one of 'last', 'random',\
                'first', 'interpolate' or 'subsample'"
        raise ValueError(error_msg)
    return timeseries, input_mask
