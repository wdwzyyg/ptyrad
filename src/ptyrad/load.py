import os
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.io as sio

from ptyrad.utils import handle_hdf5_types, list_hdf5_keys, tensors_to_ndarrays, vprint

KeyType = Union[str, list[str], None]

###### These are data loading functions ######

def load_raw(file_path, shape, dtype=np.float32, offset=0, gap=1024):
    # shape = (N, height, width)
    # np.fromfile with custom dtype is faster than the np.read and np.frombuffer
    # This implementaiton is also roughly 2x faster (10sec vs 20sec) than load_hdf5 with a 128x128x128x128 (1GB) EMPAD dataset
    # Note that for custom processed empad2 raw there might be no gap between the images
    N, height, width = shape

    # Verify file size first
    expected_size = offset + N * (height * width * dtype().itemsize + gap)
    actual_size = os.path.getsize(file_path)

    if actual_size != expected_size:
        raise ValueError(f"Mismatch in expected ({expected_size} bytes = offset + N * (height * width * 4 + gap)) vs. actual ({actual_size} bytes) file size! Check your loading configurations!")
    
    # Define the custom dtype to include both data and gap
    custom_dtype = np.dtype([
        ('data', dtype, (height, width)),
        ('gap', np.uint8, gap)  # uint8 means 1 byte per gap element
    ])

    # Read the entire file using the custom dtype
    with open(file_path, 'rb') as f:
        f.seek(offset)
        raw_data = np.fromfile(f, dtype=custom_dtype, count=N)

    # Extract just the 'data' part (ignoring the gaps)
    data = raw_data['data']
    vprint("Success! Loaded .raw file path =", file_path)
    vprint("Imported .raw data shape =", data.shape)
    vprint("Imported .raw data type =", data.dtype)
    return data

def load_tif(file_path):
    from tifffile import imread

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    data = imread(file_path)
    vprint("Success! Loaded .tif file path =", file_path)
    vprint("Imported .tif data shape =", data.shape)
    return data

def load_npy(file_path):

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    data = np.load(file_path)
    vprint("Success! Loaded .npy file path =", file_path)
    vprint("Imported .npy data shape =", data.shape)
    return data

def load_measurements(
    path: str,
    key: Optional[str] = None,
    shape: Optional[Tuple[int, ...]] = None,
    offset: Optional[int] = None,
    gap: Optional[int] = None,
) -> np.ndarray:
    """
    Load diffraction measurements from a file. The file type is inferred from the extension.
    Currently supports .tif, .tiff, .npy, .mat, .h5, .hdf5, and .raw.

    Args:
        path (str): Path to the file.
        key (str): Key to specify the dataset (optional).
        shape (tuple): Shape of the data for .raw files (optional).
        offset (int): Offset for .raw files (optional).
        gap (int): Gap for .raw files (optional).

    Returns:
        numpy.ndarray: The loaded measurements.

    Raises:
        ValueError: If the file type is unsupported or no valid dataset is found.
    """

    file_path = path  # The function signature is simplified for users, although I think file_path is clearer

    # Check file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")

    # Infer file type from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in [".tif", ".tiff"]:
        return load_tif(file_path)

    elif ext == ".npy":
        return load_npy(file_path)

    elif ext in [".mat", ".h5", ".hdf5"]:
        return load_ND_with_key(file_path, key)

    elif ext == ".raw":
        if shape is None:
            raise ValueError(
                f"Please at least provide 'shape' of the expected data array to correctly load the .raw file {file_path}."
            )
        raw_args = {"shape": shape, "offset": offset, "gap": gap}
        raw_args = {
            k: v for k, v in raw_args.items() if v is not None
        }  # Remove argument with None
        return load_raw(file_path, **raw_args)

    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported types are .tif, .tiff, .mat, .h5, .hdf5, .npy, and .raw."
        )

def load_ND_with_key(
    file_path: str,
    key: Optional[str] = None,
    ndims: Optional[List[int]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load exactly one ND dataset from (possibly nested) files like .mat and .hdf5.

    Args:
        file_path (str): Path to the file.
        key (str, optional): Key to specify the dataset. If not provided, will search for all valid ND datasets.
        ndims (list): List of desired dimensions for filtering datasets.
        verbose (bool): Whether to print information about the datasets.

    Returns:
        numpy.ndarray: The loaded dataset.

    Raises:
        ValueError: If the file type is unsupported, or the key is invalid, or multiple/zero valid datasets are found.
    """

    if ndims is None:
        ndims = [3, 4]

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The specified file '{file_path}' does not exist. Please check your file path."
        )

    # Infer file type from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Select loader
    if ext == ".mat":
        load_func = load_fields_from_mat
    elif ext in [".h5", ".hdf5"]:
        load_func = load_hdf5
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported types are .mat, .h5, .hdf5."
        )

    # Load the data using the selected loader.
    if key in (None, ""):
        datasets_dict = load_func(file_path)  # None key would return a dict of the file
        valid_datasets = collect_ND_datasets(
            datasets_dict, ndims=ndims, verbose=verbose
        )  # This will search recursively and return all valid ND datasets
        if len(valid_datasets) == 1:
            return next(iter(valid_datasets.values()))
        elif len(valid_datasets) == 0:
            raise ValueError(
                f"No eligible datasets found in file with ndims = {ndims}. Please check the file and file path."
            )
        else:
            raise ValueError(
                f"Multiple eligible ND datasets found: {list(valid_datasets.keys())}. Please specify the dataset key explicitly."
            )

    elif isinstance(key, str):
        data_or_dict = load_func(
            file_path, key
        )  # String key would normally return ndarray, but incorrectly specified key may point to a group or anything else
        if isinstance(data_or_dict, np.ndarray):
            return data_or_dict
        else:
            raise ValueError(
                f"The returned value at key '{key}' is not an ndarray dataset, got type = {type(data_or_dict).__name__}. "
                "If you don't know the correct dataset key, try 'key=None' to search for eligible ND datasets from the entire file."
            )
    else:
        raise TypeError(f"`key` must be None or a string, but got key = '{key}'")

def collect_ND_datasets(
    data_dict: dict[str, Any],
    ndims: list[int] = None,
    delimiter: str = ".",
    verbose: bool = True,
    _parent_key: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """
    Collect ND numpy arrays from a (possibly nested) dictionary that match desired dimensionalities.

    Automatically traverses nested dictionaries and flattens keys with '//'.

    Args:
        data_dict (dict): Dictionary of datasets (flat or nested).
        ndims (list of int): Desired dimensionalities to match (e.g., [3, 4]).
        delimiter (str): String symbol used to seperate different levels of the full path to the dataset
        verbose (bool): Whether to print matched datasets.
        _parent_key (str, optional): **Internal use only.** Tracks nested keys during recursion. Do not set manually.

    Returns:
        dict[str, np.ndarray]: Matching datasets with flattened hierarchical keys.

    Raises:
        ValueError: If input is not a dict or no datasets match.
    """
    if not isinstance(data_dict, dict):
        raise ValueError("Input must be a dictionary containing datasets.")

    if ndims is None:
        ndims = [3, 4]

    results: dict[str, np.ndarray] = {}

    for key, val in data_dict.items():
        full_key = f"{_parent_key}{delimiter}{key}" if _parent_key else key

        if isinstance(val, np.ndarray):
            if val.ndim in ndims:
                results[full_key] = val

        elif isinstance(val, dict):
            results.update(
                collect_ND_datasets(
                    val, ndims=ndims, verbose=False, _parent_key=full_key
                )
            )

    if verbose and results:
        vprint(f"Found the following ND datasets with ndim in {ndims}:")
        for k, arr in results.items():
            vprint(f"  Key: '{k}', Shape: {arr.shape}, Dtype: {arr.dtype}")

    return results
    
###### These are reconstruction file loading functions ######
# Note that .mat and .hdf5 are also used for normal data

def load_fields_from_mat(file_path, target_field=None, squeeze_me=True, simplify_cells=True):
    """
    Load and extract specified fields from a MATLAB .mat file.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

    Parameters:
        file_path (str): The path to the MATLAB .mat file to be loaded and processed.
        target_field (str or list of str): The target field name(s) to extract from the .mat file.
            Specify a single field name as a string or multiple field names as a list of strings.
            Use "All" to load the entire .mat file.

    Returns:
        result_list (list or dict): A list containing the extracted field(s) as elements.
            If target_field is "All," the entire .mat file is returned as a dictionary.

    Raises:
        ValueError: If the nesting depth of target_field exceeds the maximum supported depth of 3.
        ValueError: If target_field is neither a string nor a list of strings.

    Examples:
        # Load the entire .mat file as a dictionary
        file_path = "your_file.mat"
        target_field = "All"
        result = load_fields_from_mat(file_path, target_field)

        # Extract a single field
        file_path = "your_file.mat"
        target_field = "object.sub_field"
        result = load_fields_from_mat(file_path, target_field)

        # Extract multiple fields
        file_path = "your_file.mat"
        target_field = ["object.sub_field", "another_object.field"]
        results = load_fields_from_mat(file_path, target_field)

        # Process the results
        for i, result in enumerate(results):
            if result is not None:
                vprint(f"Result {i + 1}: {result}")
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    result_list = []

    # Load entire .mat
    if target_field is None:
        try:
            mat_contents = sio.loadmat(
                file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells
            )
            vprint("Success! Loaded .mat File path =", file_path)
            return mat_contents
        except NotImplementedError:
            # If loading from MATLAB file complains, switch to HDF5
            vprint("Can't load .mat v7.3 with 'scipy.io.loadmat'. Switching to h5py.")
            mat_contents = {}
            with h5py.File(file_path, "r") as hdf_file:
                for key in hdf_file.keys():
                    mat_contents[key] = hdf_file[key][()]
            vprint("Success! Loaded .mat file path =", file_path)
            return mat_contents

    # Check target_field type
    if isinstance(target_field, str):
        target_fields = [target_field]
    elif isinstance(target_field, list):
        target_fields = target_field
    else:
        raise ValueError("target_field must be a string or a list of strings")

    # Load field by field in target_fields (list)
    for name in target_fields:
        try:
            mat_contents = sio.loadmat(
                file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells
            )
            fields = name.split(".")
            outputs = mat_contents

            if len(fields) > 3:
                raise ValueError("The maximum supported nesting depth is 3.")

            for field in fields:
                if field in outputs:
                    if isinstance(outputs, sio.matlab.mio5.mat_struct):
                        outputs = getattr(outputs, field)
                    else:
                        outputs = outputs[field]
                else:
                    vprint(f"Field '{field}' not found in file {file_path}")
                    result_list.append(None)
                    break
            else:
                result_list.append(outputs)
        except NotImplementedError:
            # If loading from MATLAB file complains, switch to HDF5
            vprint("Can't load .mat v7.3 with scipy. Switching to h5py.")
            if name == 'outputs.probe_positions': # Convert the scipy syntax to hdf5 syntax
                name = 'outputs/probe_positions'
            data = load_hdf5(file_path, name)
            result_list.append(data)
    vprint("Success! Loaded .mat file path =", file_path)
    return result_list[0] if len(result_list)==1 else result_list

def load_hdf5(
    file_path: str, key: KeyType = None, delimiter: str = "."
) -> Union[np.ndarray, dict[str, np.ndarray]]:
    """
    Load dataset(s) from an HDF5 file, recursively if groups are encountered.

    Parameters:
    - file_path (str): Path to the HDF5 file.
    - key (str | list[str] | None): Name(s) of the dataset(s) to load.
        - If None, '', or []: Load all datasets recursively, preserving the original nested structure.
        - If str: Load a single dataset or group. Supports hierarchical keys (e.g., 'group1.dataset1').
        - If list[str]: Load multiple datasets. The returned dictionary will have a flattened structure with the hierarchical key strings as keys.
    - delimiter (str): Delimiter for hierarchical keys (default: ".").

    Returns:
    - data (np.ndarray or dict): The loaded dataset(s).
        - If `key` is a string, returns a single `np.ndarray` or a nested dictionary if the key points to a group.
        - If `key` is a list of strings, returns a dictionary with the hierarchical key strings as keys and the corresponding datasets as values.
        - If `key` is None, returns a nested dictionary preserving the original structure of the HDF5 file.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - KeyError: If provided key(s) are not found in the file.
    - TypeError: If the key is not None, a string, or a list of strings.

    Notes:
    - **Hierarchical Keys**: 
        - The function supports hierarchical keys (e.g., 'group1.dataset1') to directly access nested datasets or groups.
        - When a list of hierarchical keys is provided, the returned dictionary will have a flattened structure with the hierarchical key strings as keys.
        - Example: If `key=["group1/dataset1", "group2/dataset2"]`, the returned dictionary will look like:
          ```python
          {
              "group1/dataset1": <numpy.ndarray>,
              "group2/dataset2": <numpy.ndarray>
          }
          ```

    - **Preserving Original Structure**:
        - If [key=None], the function recursively loads all datasets and groups, preserving the original nested structure of the HDF5 file.
        - Example: If the HDF5 file has the following structure:
          ```
          /group1
              /dataset1
              /dataset2
          /group2
              /dataset3
          ```
          The returned dictionary will look like:
          ```python
          {
              "group1": {
                  "dataset1": <numpy.ndarray>,
                  "dataset2": <numpy.ndarray>
              },
              "group2": {
                  "dataset3": <numpy.ndarray>
              }
          }
          ```

    - **Performance Considerations**:
        - Providing an exact key (e.g., [key="group1/dataset1"]) is significantly faster than recursively loading the entire file or traversing the hierarchy.
        - Recursive loading ([key=None]) or traversing with a list of keys can be slow for large HDF5 files with deeply nested structures or many datasets.

    Examples:
    1. **Load a Single Dataset**:
       ```python
       data = load_hdf5("example.h5", key="group1/dataset1")
       ```

    2. **Load Multiple Datasets with Hierarchical Keys**:
       ```python
       data = load_hdf5("example.h5", key=["group1/dataset1", "group2/dataset3"])
       # Output:
       # {
       #     "group1/dataset1": <numpy.ndarray>,
       #     "group2/dataset3": <numpy.ndarray>
       # }
       ```

    3. **Load All Datasets Preserving Original Structure**:
       ```python
       data = load_hdf5("example.h5", key=None)
       # Output:
       # {
       #     "group1": {
       #         "dataset1": <numpy.ndarray>,
       #         "dataset2": <numpy.ndarray>
       #     },
       #     "group2": {
       #         "dataset3": <numpy.ndarray>
       #     }
       # }
       ```

    """

    def _recursively_load(hobj, key=None, delimiter="."):
        """Recursively load h5py Group or Dataset into dict or array."""

        # Traverse hierarchically with a user-specified key
        if key is not None:
            parts = key.split(delimiter)
            for part in parts:
                if not isinstance(hobj, (h5py.Group, h5py.File)) or part not in hobj:
                    raise KeyError(
                        f"Key '{key}' not found. Failed at '{part}'. "
                        f"Available key(s) in this HDF5 file are {list_hdf5_keys(hf)}. "
                        "If you don't know the correct key, try 'key=None' to load the entire file as a dict."
                    )
                hobj = hobj[part]

        # Load the object without user-specified key
        if isinstance(hobj, h5py.Dataset):
            return handle_hdf5_types(hobj[()])
        elif isinstance(hobj, h5py.Group):
            return {k: _recursively_load(hobj[k]) for k in hobj}
        else:
            raise TypeError(f"Unsupported HDF5 object type: {type(hobj)}")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The specified file '{file_path}' does not exist. Please check your file path."
        )

    with h5py.File(file_path, "r") as hf:
        if key in (None, "", []):
            file_dict = {k: _recursively_load(hf[k]) for k in hf.keys()}
            vprint(f"Success! Loaded .hdf5 file as a dict from path = '{file_path}'")
            return file_dict

        elif isinstance(key, str):
            data = _recursively_load(hf, key=key, delimiter=delimiter)
            vprint(
                f"Success! Loaded .hdf5 file with key = '{key}' from path = '{file_path}'"
            )
            if isinstance(data, np.ndarray):
                vprint(f"Imported .hdf5 data shape = {data.shape}")
                vprint(f"Imported .hdf5 data type = {data.dtype}")
            return data

        elif isinstance(key, list):
            if not all(isinstance(k, str) for k in key):
                raise TypeError(
                    f"All elements in 'key' list must be strings, got {[type(k).__name__ for k in key]}"
                )
            missing = []
            for k in key:
                try:
                    _recursively_load(hf, key=k, delimiter=delimiter)
                except KeyError:
                    missing.append(k)
            if missing:
                raise KeyError(
                    f"Key(s) = {missing} not found. Available key(s) in this HDF5 file are {list_hdf5_keys(hf)}. "
                    "If you don't know the correct key, try 'key=None' to load the entire file as a dict."
                )
            datasets_dict = {
                k: _recursively_load(hf, key=k, delimiter=delimiter) for k in key
            }
            vprint(
                f"Success! Loaded .hdf5 file as a dict with keys = {key} from path = '{file_path}'"
            )
            return datasets_dict

        else:
            raise TypeError(
                f"`key` must be None, a string, or a list of strings but got key = '{key}'"
            )

def load_pt(file_path, weights_only=False):
    import torch

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")

    data = torch.load(file_path, weights_only=weights_only) 
    # The default behavior of torch.load is `weights_only=True` since PyTorch 2.6 (2025.01.29)
    # https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573
    # Because PtyRAD .pt isn't a true PyTorch model, so `weights_only=True` would break this critical loading function.
    # However, `weights_only=False` has potential risk if the .pt file contains malicious code, so please only use this `load_pt` for PtyRAD-generated .pt file.
    
    vprint("Success! Loaded .pt file path =", file_path)
    return data

def load_ptyrad(file_path: str) -> Dict[str, Any]:
    """
    Load PtyRAD reconstruction files based on their file extension.

    This function supports loading files with extensions `.h5`, `.hdf5`, and `.pt`.
    The file type is inferred from the extension, and the appropriate loader function is called.
    The suggested model output file type has changed to HDF5 since PtyRAD v0.1.0b7 for cross-platform interoperability.

    Args:
        file_path (str): Path to the file to be loaded.

    Returns:
        Any: The loaded data, typically as a numpy array or dictionary, depending on the file type.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file type is unsupported.

    Notes:
        - `.h5` and `.hdf5` files are loaded using the `load_hdf5` function.
        - `.pt` files are loaded using the `load_pt` function and converted to numpy arrays for backward compatibility.
        - Unsupported file types will raise a `ValueError`.

    Example:
        ```python
        data = load_ptyrad("example.h5")
        ```
    """
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    # Infer file type from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in [".h5", ".hdf5"]:
        return load_hdf5(file_path)

    elif ext == ".pt":
        vprint("WARNING: Loading PtyRAD reconstruction from .pt file is deprecated and will likely be removed by 2025 Aug.")
        vprint("INFO: PtyRAD reconstruction output has been using .hdf5 format since v0.1.0b7.")
        return tensors_to_ndarrays(load_pt(file_path)) # .pt is supported for backward compatibility before 0.1.0b7. (e.g. PtyRAD reconstructions used for the paper)
    
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported types are .h5, .hdf5, and .pt."
        )

###### These are params loading functions ######

def load_params(file_path):
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    vprint("### Loading params file ###")
    param_path, param_type = os.path.splitext(file_path)
    if param_type in (".yml", ".yaml"):
        params_dict = load_yml_params(file_path)
    elif param_type == ".toml":
        params_dict = load_toml_params(file_path)
    elif param_type == ".json":
        params_dict = load_json_params(file_path)
    elif param_type == ".py":
        params_dict =  load_py_params(param_path)
    else:
        raise ValueError("param_type needs to be either 'yml', 'json', or 'py'")
    
    # Add the file path to the params_dict while we save the params file to output folder
    params_dict['params_path'] = file_path
    
    vprint(" ")
    return params_dict

def load_json_params(file_path):
    import json
    
    with open(file_path, "r", encoding='utf-8') as file:
        params_dict = json.load(file)
    vprint("Success! Loaded .json file path =", file_path)
    return params_dict

def load_toml_params(file_path):
    """
    Load parameters from a TOML file.
    
    Parameters:
    file_path (str): The path to the TOML file to be loaded.
    
    Returns:
    dict: A dictionary containing the parameters loaded from the TOML file.
    
    Raises:
    FileNotFoundError: If the specified file does not exist.
    ImportError: If the tomli package is not installed for Python < 3.11.
    """

    try:
        # Read the file with utf-8
        # Note that "A TOML file must be a valid UTF-8 encoded Unicode document." per documentation.
        # Therefore, the toml file is read in binary mode ("rb") and the encoding is handled internally.
        # But I've observed some encoding mismatch when people run the script with terminal that has different default encoding.
        # Therefore, it is safer to read it with utf-8 encoding first and pass it to tomllib.
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
        
        try:
            # For Python 3.11+
            import tomllib
            params_dict = tomllib.loads(content)
        except ImportError:
            # For Python < 3.11
            import tomli # type: ignore
            params_dict = tomli.loads(content)
    except ImportError:
        raise ImportError("TOML support requires 'tomli' package for Python < 3.11 or built-in 'tomllib' for Python 3.11+. ")
    
    vprint("Success! Loaded .toml file path =", file_path)
    return params_dict

def load_yml_params(file_path):
    import yaml

    with open(file_path, "r", encoding='utf-8') as file:
        params_dict = yaml.safe_load(file)
    vprint("Success! Loaded .yml file path =", file_path)
    return params_dict

def load_py_params(file_path):
    import importlib

    params_module = importlib.import_module(file_path)
    params_dict = {
        name: getattr(params_module, name)
        for name in dir(params_module)
        if not name.startswith("__")
    }
    vprint("Success! Loaded .py file path =", file_path)
    return params_dict