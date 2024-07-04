from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, is_dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
from jax import grad, vmap
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from moment.data.serialize import SerializerSettings, deserialize_str, serialize_arr
from moment.utils.utils import grid_iter

STEP_MULTIPLIER = 1.1

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def llama2_model_string(model_size, chat):
    chat = "chat-" if chat else ""
    return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"


def get_tokenizer(model):
    name_parts = model.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1
    assert model_size in ["7b", "13b", "70b"]

    tokenizer = LlamaTokenizer.from_pretrained(
        llama2_model_string(model_size, chat),
        use_fast=True,
        token="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",
    )
    # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", trust_remote_code=True)

    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token
    len(tokenizer)

    return tokenizer


def get_model_and_tokenizer(model):
    name_parts = model.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1

    assert model_size in ["7b", "13b", "70b"]

    tokenizer = get_tokenizer(model)

    model = LlamaForCausalLM.from_pretrained(
        llama2_model_string(model_size, chat),
        device_map="auto",
        torch_dtype=torch.float16,
        token="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",
    )
    model.eval()

    return model, tokenizer


def llama_tokenize_fn(str, model):
    tokenizer = get_tokenizer(model)
    return tokenizer(str)


def llama_nll_fn(
    model,
    input_arr,
    target_arr,
    settings: SerializerSettings,
    transform,
    count_seps=True,
    temp=1,
):
    """Returns the NLL/dimension (log base e) of the target array (continuous) according to the LM
        conditioned on the input array. Applies relevant log determinant for transforms and
        converts from discrete NLL of the LLM to continuous by assuming uniform within the bins.
    inputs:
        input_arr: (n,) context array
        target_arr: (n,) ground truth array
    Returns: NLL/D
    """
    model, tokenizer = get_model_and_tokenizer(model)

    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    full_series = input_str + target_str

    batch = tokenizer([full_series], return_tensors="pt", add_special_tokens=True)
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    good_tokens_str = list("0123456789" + settings.time_sep)
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
    out["logits"][:, :, bad_tokens] = -100

    input_ids = batch["input_ids"][0][1:]
    logprobs = torch.nn.functional.log_softmax(out["logits"], dim=-1)[0][:-1]
    logprobs = logprobs[torch.arange(len(input_ids)), input_ids].cpu().numpy()

    tokens = tokenizer.batch_decode(
        input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    input_len = len(
        tokenizer(
            [input_str],
            return_tensors="pt",
        )["input_ids"][0]
    )
    input_len = input_len - 2  # remove the BOS token

    logprobs = logprobs[input_len:]
    tokens = tokens[input_len:]
    BPD = -logprobs.sum() / len(target_arr)

    # print("BPD unadjusted:", -logprobs.sum()/len(target_arr), "BPD adjusted:", BPD)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec * np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll - avg_logdet_dydx


def llama_completion_fn(
    model,
    input_str,
    steps,
    settings,
    batch_size=1,
    num_samples=20,
    temp=0.9,
    top_p=0.9,
):
    avg_tokens_per_step = len(input_str) / steps
    max_tokens = int(avg_tokens_per_step * steps)

    model, tokenizer = get_model_and_tokenizer(model)

    gen_strs = []
    for _ in tqdm(range(num_samples // batch_size)):
        batch = tokenizer(
            [input_str],
            return_tensors="pt",
        )

        batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
        batch = {k: v.cuda() for k, v in batch.items()}

        good_tokens_str = list("0123456789" + settings.time_sep)
        good_tokens = [
            tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str
        ]
        good_tokens += [tokenizer.eos_token_id]
        bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            bad_words_ids=[[t] for t in bad_tokens],
            renormalize_logits=True,
        )

        gen_strs += tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    # gen_strs = [x.replace(input_str, '').strip() for x in gen_strs]

    return gen_strs


# Required: Text completion function for each model
# -----------------------------------------------
# Each model is mapped to a function that samples text completions.
# The completion function should follow this signature:
#
# Args:
#   - input_str (str): String representation of the input time series.
#   - steps (int): Number of steps to predict.
#   - settings (SerializerSettings): Serialization settings.
#   - num_samples (int): Number of completions to sample.
#   - temp (float): Temperature parameter for model's output randomness.
#
# Returns:
#   - list: Sampled completion strings from the model.
completion_fns = {
    "llama-7b": partial(llama_completion_fn, model="7b"),
    "llama-13b": partial(llama_completion_fn, model="13b"),
    "llama-70b": partial(llama_completion_fn, model="70b"),
    "llama-7b-chat": partial(llama_completion_fn, model="7b-chat"),
    "llama-13b-chat": partial(llama_completion_fn, model="13b-chat"),
    "llama-70b-chat": partial(llama_completion_fn, model="70b-chat"),
}

# Optional: NLL/D functions for each model
# -----------------------------------------------
# Each model is mapped to a function that computes the continuous Negative Log-Likelihood
# per Dimension (NLL/D). This is used for computing likelihoods only and not needed for sampling.
#
# The NLL function should follow this signature:
#
# Args:
#   - input_arr (np.ndarray): Input time series (history) after data transformation.
#   - target_arr (np.ndarray): Ground truth series (future) after data transformation.
#   - settings (SerializerSettings): Serialization settings.
#   - transform (callable): Data transformation function (e.g., scaling) for determining the Jacobian factor.
#   - count_seps (bool): If True, count time step separators in NLL computation, required if allowing variable number of digits.
#   - temp (float): Temperature parameter for sampling.
#
# Returns:
#   - float: Computed NLL per dimension for p(target_arr | input_arr).
nll_fns = {
    "llama-7b": partial(llama_nll_fn, model="7b"),
    "llama-13b": partial(llama_nll_fn, model="13b"),
    "llama-70b": partial(llama_nll_fn, model="70b"),
    "llama-7b-chat": partial(llama_nll_fn, model="7b-chat"),
    "llama-13b-chat": partial(llama_nll_fn, model="13b-chat"),
    "llama-70b-chat": partial(llama_nll_fn, model="70b-chat"),
}

# Optional: Tokenization function for each model, only needed if you want automatic input truncation.
# The tokenization function should follow this signature:
#
# Args:
#   - str (str): A string to tokenize.
# Returns:
#   - token_ids (list): A list of token ids.
tokenization_fns = {
    "llama-7b": partial(llama_tokenize_fn, model="7b"),
    "llama-13b": partial(llama_tokenize_fn, model="13b"),
    "llama-70b": partial(llama_tokenize_fn, model="70b"),
    "llama-7b-chat": partial(llama_tokenize_fn, model="7b-chat"),
    "llama-13b-chat": partial(llama_tokenize_fn, model="13b-chat"),
    "llama-70b-chat": partial(llama_tokenize_fn, model="70b-chat"),
}

# Optional: Context lengths for each model, only needed if you want automatic input truncation.
context_lengths = {
    "llama-7b": 4096,
    "llama-13b": 4096,
    "llama-70b": 4096,
    "llama-7b-chat": 4096,
    "llama-13b-chat": 4096,
    "llama-70b-chat": 4096,
}


@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """

    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x


def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha), 0.01)

        def transform(x):
            return x / q

        def inv_transform(x):
            return x * q
    else:
        min_ = np.min(history) - beta * (np.max(history) - np.min(history))
        q = np.quantile(history - min_, alpha)

        def transform(x):
            return (x - min_) / q

        def inv_transform(x):
            return x * q + min_

    return Scaler(transform=transform, inv_transform=inv_transform)


def truncate_input(input_arr, input_str, settings, model, steps):
    """
    Truncate inputs to the maximum context length for a given model.

    Args:
        input (array-like): input time series.
        input_str (str): serialized input time series.
        settings (SerializerSettings): Serialization settings.
        model (str): Name of the LLM model to use.
        steps (int): Number of steps to predict.
    Returns:
        tuple: Tuple containing:
            - input (array-like): Truncated input time series.
            - input_str (str): Truncated serialized input time series.
    """
    if model in tokenization_fns and model in context_lengths:
        tokenization_fn = tokenization_fns[model]
        context_length = context_lengths[model]
        input_str_chuncks = input_str.split(settings.time_sep)
        for i in range(len(input_str_chuncks) - 1):
            truncated_input_str = settings.time_sep.join(input_str_chuncks[i:])
            # add separator if not already present
            if not truncated_input_str.endswith(settings.time_sep):
                truncated_input_str += settings.time_sep
            input_tokens = tokenization_fn(truncated_input_str)
            num_input_tokens = len(input_tokens)
            avg_token_length = num_input_tokens / (len(input_str_chuncks) - i)
            num_output_tokens = avg_token_length * steps * STEP_MULTIPLIER
            if num_input_tokens + num_output_tokens <= context_length:
                truncated_input_arr = input_arr[i:]
                break
        if i > 0:
            print(
                f"Warning: Truncated input from {len(input_arr)} to {len(truncated_input_arr)}"
            )
        return truncated_input_arr, truncated_input_str
    else:
        return input_arr, input_str


def handle_prediction(pred, expected_length, strict=False):
    """
    Process the output from LLM after deserialization, which may be too long or too short, or None if deserialization failed on the first prediction step.

    Args:
        pred (array-like or None): The predicted values. None indicates deserialization failed.
        expected_length (int): Expected length of the prediction.
        strict (bool, optional): If True, returns None for invalid predictions. Defaults to False.

    Returns:
        array-like: Processed prediction.
    """
    if pred is None:
        return None
    else:
        if len(pred) < expected_length:
            if strict:
                print(
                    f"Warning: Prediction too short {len(pred)} < {expected_length}, returning None"
                )
                return None
            else:
                print(
                    f"Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value"
                )
                return np.concatenate(
                    [pred, np.full(expected_length - len(pred), pred[-1])]
                )
        else:
            return pred[:expected_length]


def generate_predictions(
    completion_fn,
    input_strs,
    steps,
    settings: SerializerSettings,
    scalers: None,
    num_samples=1,
    temp=0.7,
    parallel=True,
    strict_handling=False,
    max_concurrent=10,
    **kwargs,
):
    """
    Generate and process text completions from a language model for input time series.

    Args:
        completion_fn (callable): Function to obtain text completions from the LLM.
        input_strs (list of array-like): List of input time series.
        steps (int): Number of steps to predict.
        settings (SerializerSettings): Settings for serialization.
        scalers (list of Scaler, optional): List of Scaler objects. Defaults to None, meaning no scaling is applied.
        num_samples (int, optional): Number of samples to return. Defaults to 1.
        temp (float, optional): Temperature for sampling. Defaults to 0.7.
        parallel (bool, optional): If True, run completions in parallel. Defaults to True.
        strict_handling (bool, optional): If True, return None for predictions that don't have exactly the right format or expected length. Defaults to False.
        max_concurrent (int, optional): Maximum number of concurrent completions. Defaults to 50.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: Tuple containing:
            - preds (list of lists): Numerical predictions.
            - completions_list (list of lists): Raw text completions.
            - input_strs (list of str): Serialized input strings.
    """

    completions_list = []
    complete = lambda x: completion_fn(
        input_str=x,
        steps=steps * STEP_MULTIPLIER,
        settings=settings,
        num_samples=num_samples,
        temp=temp,
    )
    if parallel and len(input_strs) > 1:
        print("Running completions in parallel for each input")
        with ThreadPoolExecutor(min(max_concurrent, len(input_strs))) as p:
            completions_list = list(
                tqdm(p.map(complete, input_strs), total=len(input_strs))
            )
    else:
        completions_list = [complete(input_str) for input_str in tqdm(input_strs)]

    def completion_to_pred(completion, inv_transform):
        pred = handle_prediction(
            deserialize_str(completion, settings, ignore_last=False, steps=steps),
            expected_length=steps,
            strict=strict_handling,
        )
        if pred is not None:
            return inv_transform(pred)
        else:
            return None

    preds = [
        [
            completion_to_pred(completion, scaler.inv_transform)
            for completion in completions
        ]
        for completions, scaler in zip(completions_list, scalers)
    ]
    return preds, completions_list, input_strs


def get_llmtime_predictions_data(
    train,
    test,
    model,
    settings,
    num_samples=10,
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    parallel=True,
    **kwargs,
):
    """
    Obtain forecasts from an LLM based on training series (history) and evaluate likelihood on test series (true future).
    train and test can be either a single time series or a list of time series.

    Args:
        train (array-like or list of array-like): Training time series data (history).
        test (array-like or list of array-like): Test time series data (true future).
        model (str): Name of the LLM model to use. Must have a corresponding entry in completion_fns.
        settings (SerializerSettings or dict): Serialization settings.
        num_samples (int, optional): Number of samples to return. Defaults to 10.
        temp (float, optional): Temperature for sampling. Defaults to 0.7.
        alpha (float, optional): Scaling parameter. Defaults to 0.95.
        beta (float, optional): Shift parameter. Defaults to 0.3.
        basic (bool, optional): If True, use the basic version of data scaling. Defaults to False.
        parallel (bool, optional): If True, run predictions in parallel. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Dictionary containing predictions, samples, median, NLL/D averaged over each series, and other related information.
    """

    assert (
        model in completion_fns
    ), f"Invalid model {model}, must be one of {list(completion_fns.keys())}"
    completion_fn = completion_fns[model]
    nll_fn = nll_fns[model] if model in nll_fns else None

    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    if not isinstance(train, list):
        # Assume single train/test case
        train = [train]
        test = [test]

    for i in range(len(train)):
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index=pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(
                test[i],
                index=pd.RangeIndex(len(train[i]), len(test[i]) + len(train[i])),
            )

    test_len = len(test[0])
    assert all(
        len(t) == test_len for t in test
    ), f"All test series must have same length, got {[len(t) for t in test]}"

    # Create a unique scaler for each series
    scalers = [
        get_scaler(train[i].values, alpha=alpha, beta=beta, basic=basic)
        for i in range(len(train))
    ]

    # transform input_arrs
    input_arrs = [train[i].values for i in range(len(train))]
    transformed_input_arrs = np.array(
        [
            scaler.transform(input_array)
            for input_array, scaler in zip(input_arrs, scalers)
        ]
    )
    # serialize input_arrs
    input_strs = [
        serialize_arr(scaled_input_arr, settings)
        for scaled_input_arr in transformed_input_arrs
    ]
    # Truncate input_arrs to fit the maximum context length
    input_arrs, input_strs = zip(
        *[
            truncate_input(input_array, input_str, settings, model, test_len)
            for input_array, input_str in zip(input_arrs, input_strs)
        ]
    )

    steps = test_len
    samples = None
    medians = None
    completions_list = None
    if num_samples > 0:
        preds, completions_list, input_strs = generate_predictions(
            completion_fn,
            input_strs,
            steps,
            settings,
            scalers,
            num_samples=num_samples,
            temp=temp,
            parallel=parallel,
            **kwargs,
        )
        samples = [
            pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))
        ]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        medians = medians if len(medians) > 1 else medians[0]
    out_dict = {
        "samples": samples,
        "median": medians,
        "info": {
            "Method": model,
        },
        "completions_list": completions_list,
        "input_strs": input_strs,
    }
    # # Compute NLL/D on the true test series conditioned on the (truncated) input series
    # if nll_fn is not None:
    #     BPDs = [nll_fn(input_arr=input_arrs[i], target_arr=test[i].values, settings=settings, transform=scalers[i].transform, count_seps=True, temp=temp) for i in range(len(train))]
    #     out_dict['NLL/D'] = np.mean(BPDs)
    return out_dict


def make_validation_dataset(train, n_val, val_length):
    """Partition the training set into training and validation sets.

    Args:
        train (list): List of time series data for training.
        n_val (int): Number of validation samples.
        val_length (int): Length of each validation sample.

    Returns:
        tuple: Lists of training data without validation, validation data, and number of validation samples.
    """
    assert isinstance(train, list), "Train should be a list of series"

    train_minus_val_list, val_list = [], []
    if n_val is None:
        n_val = len(train)
    for train_series in train[:n_val]:
        train_len = max(len(train_series) - val_length, 1)
        train_minus_val, val = train_series[:train_len], train_series[train_len:]
        print(f"Train length: {len(train_minus_val)}, Val length: {len(val)}")
        train_minus_val_list.append(train_minus_val)
        val_list.append(val)

    return train_minus_val_list, val_list, n_val


def evaluate_hyper(hyper, train_minus_val, val, get_predictions_fn):
    """Evaluate a set of hyperparameters on the validation set.

    Args:
        hyper (dict): Dictionary of hyperparameters to evaluate.
        train_minus_val (list): List of training samples minus validation samples.
        val (list): List of validation samples.
        get_predictions_fn (callable): Function to get predictions.

    Returns:
        float: NLL/D value for the given hyperparameters, averaged over each series.
    """
    assert isinstance(train_minus_val, list) and isinstance(
        val, list
    ), "Train minus val and val should be lists of series"
    return get_predictions_fn(train_minus_val, val, **hyper, num_samples=0)["NLL/D"]


def get_autotuned_predictions_data(
    train,
    test,
    hypers,
    num_samples,
    get_predictions_fn,
    verbose=False,
    parallel=True,
    n_train=None,
    n_val=None,
):
    """
    Automatically tunes hyperparameters based on validation likelihood and retrieves predictions using the best hyperparameters. The validation set is constructed on the fly by splitting the training set.

    Args:
        train (list): List of time series training data.
        test (list): List of time series test data.
        hypers (Union[dict, list]): Either a dictionary specifying the grid search or an explicit list of hyperparameter settings.
        num_samples (int): Number of samples to retrieve.
        get_predictions_fn (callable): Function used to get predictions based on provided hyperparameters.
        verbose (bool, optional): If True, prints out detailed information during the tuning process. Defaults to False.
        parallel (bool, optional): If True, parallelizes the hyperparameter tuning process. Defaults to True.
        n_train (int, optional): Number of training samples to use. Defaults to None.
        n_val (int, optional): Number of validation samples to use. Defaults to None.

    Returns:
        dict: Dictionary containing predictions, best hyperparameters, and other related information.
    """
    if isinstance(hypers, dict):
        hypers = list(grid_iter(hypers))
    else:
        assert isinstance(hypers, list), "hypers must be a list or dict"
    if not isinstance(train, list):
        train = [train]
        test = [test]
    if n_val is None:
        n_val = len(train)
    if len(hypers) > 1:
        val_length = min(
            len(test[0]), int(np.mean([len(series) for series in train]) / 2)
        )
        train_minus_val, val, n_val = make_validation_dataset(
            train, n_val=n_val, val_length=val_length
        )  # use half of train as val for tiny train sets
        # remove validation series that has smaller length than required val_length
        train_minus_val, val = zip(
            *[
                (train_series, val_series)
                for train_series, val_series in zip(train_minus_val, val)
                if len(val_series) == val_length
            ]
        )
        train_minus_val = list(train_minus_val)
        val = list(val)
        if len(train_minus_val) <= int(0.9 * n_val):
            raise ValueError(
                f"Removed too many validation series. Only {len(train_minus_val)} out of {len(n_val)} series have length >= {val_length}. Try or decreasing val_length."
            )
        val_nlls = []

        def eval_hyper(hyper):
            try:
                return hyper, evaluate_hyper(
                    hyper, train_minus_val, val, get_predictions_fn
                )
            except ValueError:
                return hyper, float("inf")

        best_val_nll = float("inf")
        best_hyper = None
        if not parallel:
            for hyper in tqdm(hypers, desc="Hyperparameter search"):
                _, val_nll = eval_hyper(hyper)
                val_nlls.append(val_nll)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_hyper = hyper
                if verbose:
                    print(f"Hyper: {hyper} \n\t Val NLL: {val_nll:3f}")
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(eval_hyper, hyper) for hyper in hypers]
                for future in tqdm(
                    as_completed(futures),
                    total=len(hypers),
                    desc="Hyperparameter search",
                ):
                    hyper, val_nll = future.result()
                    val_nlls.append(val_nll)
                    if val_nll < best_val_nll:
                        best_val_nll = val_nll
                        best_hyper = hyper
                    if verbose:
                        print(f"Hyper: {hyper} \n\t Val NLL: {val_nll:3f}")
    else:
        best_hyper = hypers[0]
        best_val_nll = float("inf")
    print(f"Sampling with best hyper... {best_hyper} \n with NLL {best_val_nll:3f}")
    out = get_predictions_fn(
        train,
        test,
        **best_hyper,
        num_samples=num_samples,
        n_train=n_train,
        parallel=parallel,
    )
    out["best_hyper"] = convert_to_dict(best_hyper)
    return out


def convert_to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(elem) for elem in obj]
    elif is_dataclass(obj):
        return convert_to_dict(obj.__dict__)
    else:
        return obj
