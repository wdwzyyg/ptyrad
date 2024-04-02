
import os
import numpy as np
import h5py
import scipy.io as sio

def load_empad_as_4D(file_path, dim_x, dim_y, N_scan_x, N_scan_y, order):
    """
    Load binary data from a specified file and return it as a 4D NumPy array.

    Parameters:
    file_path (str): The full path to the binary data file.
    dim_x (int): Number of DP pixels in the Exp X (horizontal) dimension.
    dim_y (int): Number of DP pixels in the Exp Y (vertical) dimension (Usually it's 130 rows with 2 extra rows).
    N_scan_x (int): Number of real space scan positions in the Exp X (horizontal) direction.
    N_scan_y (int): Number of real space scan positions in the Exp Y (vertical) direction.
    order (str): The order for numpy.reshape to place the elements. 'C', 'F'
    https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
     
    Returns:
    data4D (numpy.ndarray): A 4D NumPy array containing the loaded data with shape (dim_x, dim_y, N_scan_x, N_scan_y).

    Raises:
    FileNotFoundError: If the specified file does not exist.
    ValueError: If the calculated file size does not match the actual file size.

    Example:
    file_path = '00_data/0314_step128_14.5Mx_cl115mm_25.2mrad_df0_20pA/scan_x128_y128.raw'
    data = load_empad_as_4D(file_path, 128, 130, 128, 128)
    """

    
    # Define a constant for file bit depth
    FILE_BIT_DEPTH = 32 
    
    # Calculate the expected file size based on the provided dimensions and bit depth
    file_size_calc = dim_x * dim_y * N_scan_x * N_scan_y * FILE_BIT_DEPTH // 8 # Bit to bytes conversion

    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError("Error: The specified file does not exist.")

        # Get the actual file size
        file_size_get = os.path.getsize(file_path)

        # Read binary data
        if file_size_calc == file_size_get:
            with open(file_path, 'rb') as f:
                data = np.fromfile(f, dtype='<f4', count=-1)
                if order == 'F':
                    shape = (dim_x, dim_y, N_scan_x, N_scan_y)
                    shape_str = ("(dim_x, dim_y, N_scan_x, N_scan_y)")
                    data_reshape = data.reshape(shape, order=order)
                    data4D = data_reshape[:,:-2,:,:]

                else: # np.reshape default is 'C', and 'A'
                    shape = (N_scan_y, N_scan_x, dim_y, dim_x)
                    shape_str = ("(N_scan_y, N_scan_x, dim_y, dim_x)")
                    data_reshape = data.reshape(shape, order=order)
                    data4D = data_reshape[:,:,:-2,:]

                print(f"Loaded .raw before cropping extra DP rows has flags \n{data_reshape.flags}")
                print(f"Imported data dimension with {order} order {shape_str} = {data4D.shape} after cropping")
                print("Success! File path =", file_path)
                
                return data4D
        else:
            raise ValueError("Error: Calculated file size is different from the actual file size!")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    return None

def load_hdf5(file_path, dataset_key='ds'):
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

    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError("Error: The specified file does not exist.")

        with h5py.File(file_path, 'r') as hf:
            data = np.array(hf[dataset_key])
            print("Success! hdf5 File path =", file_path)
            print("Imported hdf5 data shape =", data.shape)
            return data

    except FileNotFoundError as e:
        print(e)
    return None

def load_tif(file_path):
    from tifffile import imread
    data = imread(file_path)
    return data

def load_pt(file_path):
    import torch
    data = torch.load(file_path)
    return data

def save_4D_as_hdf5(data4D, file_path, final_shape=None, options=None, overwrite=False, source_metadata=None):
    """
    Save a 4D NumPy array to an HDF5 file with optional settings and optional metadata.

    Parameters:
    data4D (ndarray): The 4D data array to be saved.
    file_path (str): The file path where the HDF5 file will be saved.
    final_shape (tuple, optional): The desired shape of the saved dataset. If not provided, it's determined from data4D.
    options (dict, optional): A dictionary of HDF5 dataset options (e.g., compression settings, chunking).
    overwrite (bool, optional): If True, overwrite the file if it already exists.
    source_metadata (str, optional): Metadata string describing the source or path of the data.

    Example:
    kx = 2
    ky = 3
    rx = 4
    ry = 5
    data4D = np.arange(kx * ky * rx * ry).reshape(kx, ky, rx, ry)

    # Define options (optional)
    hdf5_options = {'compression': 'gzip', 'compression_opts': 9, 'chunks': (2, 3, 4, 5)}

    # Save the 4D array to an HDF5 file, overwriting if it exists, and include source metadata
    save_4D_as_hdf5(data4D, 'data.h5', final_shape=(kx, ky, rx, ry), options=hdf5_options, overwrite=True, source_metadata="Collected from experiment X.")
    """
    
    if options is None:
        options = {}

    if final_shape is None:
        final_shape = data4D.shape

    file_directory = os.path.dirname(file_path)
    
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
        print(f"Creating folder '{file_directory}'")
    mode = 'w' if overwrite else 'w-'
    
    with h5py.File(file_path, mode) as hf:
        hf.create_dataset('ds', data=np.reshape(np.ravel(data4D, order='F'), final_shape, order='F'), **options)
        
        if source_metadata is not None:
            hf['ds'].attrs['source_metadata'] = source_metadata
        
    print(f"File '{file_path}' saved successfully.")
    return
    
import scipy.io as sio
import h5py
def load_fields_from_mat(file_path, target_field='All', squeeze_me=True, simplify_cells=True):
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
            mat_contents = sio.loadmat(file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells)
            print("Success! .mat File path =", file_path)
            return mat_contents
        except NotImplementedError:
            # If loading from MATLAB file complains, switch to HDF5
            print(f"Can't load .mat v7.3 with scipy. Switching to h5py.")
            mat_contents = {}
            with h5py.File(file_path, 'r') as hdf_file:
                for key in hdf_file.keys():
                    mat_contents[key] = hdf_file[key][()]
            print("Success! .mat File path =", file_path)
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
            mat_contents = sio.loadmat(file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells)
            fields = name.split('.')
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
            print(f"Can't load .mat v7.3 with scipy. Switching to h5py.")
            with h5py.File(file_path, 'r') as hdf_file:
                result_list.append(hdf_file[name][()])
    print("Success! .mat File path =", file_path)
    return result_list






# ARCHIVE, load_hdf5 works fine for .matv7.3

# This is quite similar to `load_hdf5` and might be combined in the future
# While `load_hdf5` is designed to read dataset as numpy array, 
# `read_matv7_3` is intended to read the .mat file as a dict.
def read_matv7_3(file_path, target_field):
    try:
        with h5py.File(file_path, 'r') as file:
            # Accessing the target field and assigning it to a variable
            if target_field in file:
                target_data = file[target_field][()]
                return target_data
            else:
                print(f"Error: Target field '{target_field}' not found in the file.")

    except IOError:
        print("Error: File not found or could not be opened.")
        return None
    
def load_fields_from_mat_archive(file_path, target_field='All', squeeze_me = True, simplify_cells= True):
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
    try:
        if target_field == "All":
            mat_contents = sio.loadmat(file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells)
            print("Success! .mat File path =", file_path)
            return mat_contents

        if isinstance(target_field, str):
            target_fields = [target_field]
        elif isinstance(target_field, list):
            target_fields = target_field
        else:
            raise ValueError("target_field must be a string or a list of strings")

        result_list = []

        for name in target_fields:
            mat_contents = sio.loadmat(file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells)
            fields = name.split('.')
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
        print("Success! .mat File path =", file_path)
        return result_list

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return [None] * len(target_fields)