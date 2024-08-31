import os

import h5py
import numpy as np
import scipy.io as sio

def load_hdf5(file_path, dataset_key="ds"):
    """
    Load data from an HDF5 file.

    Parameters:
    file_path (str): The full path to the HDF5 data file.
    dataset_key (str, optional): The key of the dataset to load from the HDF5 file.

    Returns:
    data (numpy.ndarray): The loaded data.

    Raises:
    FileNotFoundError: If the specified file does not exist.

    Example:
    file_path = 'data.h5'
    data, data_source = load_hdf5(file_path, dataset_key='ds')
    """

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError("Error: The specified file does not exist.")

    with h5py.File(file_path, "r") as hf:
        data = np.array(hf[dataset_key], dtype="float32")
        print("Success! .hdf5 file path =", file_path)
        print("Imported .hdf5 data shape =", data.shape)
        return data

def load_tif(file_path):
    from tifffile import imread

    data = imread(file_path)
    print("Success! .tif file path =", file_path)
    print("Imported .tif data shape =", data.shape)
    return data

def load_pt(file_path):
    import torch

    data = torch.load(file_path)
    print("Success! .pt file path =", file_path)
    return data

def load_params(file_path):
    print("### Loading params file ###")

    param_path, param_type = os.path.splitext(file_path)
    if param_type == ".yml":
        return load_yml_params(file_path)
    elif param_type == ".py":
        return load_py_params(param_path)
    else:
        raise ValueError("param_type needs to be either 'yml' or 'py'")

def load_yml_params(file_path):
    import yaml

    with open(file_path, "r") as file:
        params_dict = yaml.safe_load(file)
    print("Success! .yml file path =", file_path)
    params_dict['params_path'] = file_path
    return params_dict

def load_py_params(file_path):
    import importlib

    params_module = importlib.import_module(file_path)
    print("Success! .py file path =", file_path)
    params_dict = {
        name: getattr(params_module, name)
        for name in dir(params_module)
        if not name.startswith("__")
    }
    params_dict['params_path'] = file_path
    return params_dict


def load_fields_from_mat(file_path, target_field="All", squeeze_me=True, simplify_cells=True):
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
                print(f"Result {i + 1}: {result}")
    """

    result_list = []

    # Load entire .mat
    if target_field == "All":
        try:
            mat_contents = sio.loadmat(
                file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells
            )
            print("Success! .mat File path =", file_path)
            return mat_contents
        except NotImplementedError:
            # If loading from MATLAB file complains, switch to HDF5
            print("Can't load .mat v7.3 with scipy. Switching to h5py.")
            mat_contents = {}
            with h5py.File(file_path, "r") as hdf_file:
                for key in hdf_file.keys():
                    mat_contents[key] = hdf_file[key][()]
            print("Success! .mat file path =", file_path)
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
                    print(f"Field '{field}' not found in file {file_path}")
                    result_list.append(None)
                    break
            else:
                result_list.append(outputs)
        except NotImplementedError:
            # If loading from MATLAB file complains, switch to HDF5
            print("Can't load .mat v7.3 with scipy. Switching to h5py.")
            with h5py.File(file_path, "r") as hdf_file:
                result_list.append(hdf_file[name][()])
    print("Success! .mat file path =", file_path)
    return result_list