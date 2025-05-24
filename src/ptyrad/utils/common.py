import io
import logging
import os
import platform
from time import perf_counter

import torch
import torch.distributed as dist


# Only used in run_ptyrad.py, might have a better place
def set_accelerator():

    try:
        from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
        from accelerate.state import DistributedType
        dataloader_config  = DataLoaderConfiguration(split_batches=True) # This supress the warning when we do `Accelerator(split_batches=True)`
        kwargs_handlers    = [DistributedDataParallelKwargs(find_unused_parameters=False)] # This avoids the error `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss.` We don't necessarily need this if we carefully register parameters (used in forward) and buffer in the `model`.
        accelerator        = Accelerator(dataloader_config=dataloader_config, kwargs_handlers=kwargs_handlers)
        vprint("### Initializing HuggingFace accelerator ###")
        vprint(f"Accelerator.distributed_type = {accelerator.distributed_type}")
        vprint(f"Accelerator.num_process      = {accelerator.num_processes}")
        vprint(f"Accelerator.mixed_precision  = {accelerator.mixed_precision}")
        
        if accelerator.distributed_type == DistributedType.NO and accelerator.mixed_precision == "no":
            vprint("'accelerate' is available but NOT using distributed mode or mixed precision")
            vprint("If you want to utilize 'accelerate' for multiGPU or mixed precision, ")
            vprint("Run `accelerate launch --multi_gpu --num_processes=2 --mixed_precision='no' -m ptyrad run <PTYRAD_ARGUMENTS>` in your terminal")
    except ImportError:
        vprint("### HuggingFace accelerator is not available, no multi-GPU or mixed-precision ###")
        accelerator = None
        
    vprint(" ")
    return accelerator

# System level utils
class CustomLogger:
    def __init__(self, log_file='output.log', log_dir='auto', prefix_date=True, prefix_jobid=0, append_to_file=True, show_timestamp=True):
        self.logger = logging.getLogger('PtyRAD')
        self.logger.setLevel(logging.INFO)
        
        # Clear all existing handlers to re-instantiate the logger
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.log_file       = log_file
        self.log_dir        = log_dir
        self.flush_file     = log_file is not None
        self.prefix_date    = prefix_date
        self.prefix_jobid   = prefix_jobid
        self.append_to_file = append_to_file
        self.show_timestamp = show_timestamp

        # Create console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s' if show_timestamp else '%(message)s')
        self.console_handler.setFormatter(formatter)
        
        # Create a buffer for file logs
        self.log_buffer = io.StringIO()
        self.buffer_handler = logging.StreamHandler(self.log_buffer)
        self.buffer_handler.setLevel(logging.INFO)
        self.buffer_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.buffer_handler)
        
        # Print logger information
        vprint("### PtyRAD Logger configuration ###")
        vprint(f"log_file       = '{self.log_file}'. If log_file = None, no log file will be created.")
        vprint(f"log_dir        = '{self.log_dir}'. If log_dir = 'auto', then log will be saved to `output_path` or 'logs/'.")
        vprint(f"flush_file     = {self.flush_file}. Automatically set to True if `log_file is not None`")
        vprint(f"prefix_date    = {self.prefix_date}. If true, a datetime str is prefixed to the `log_file`.")
        vprint(f"prefix_jobid   = '{self.prefix_jobid}'. If not 0, it'll be prefixed to the log file. This is used for hypertune mode with multiple GPUs.")
        vprint(f"append_to_file = {self.append_to_file}. If true, logs will be appended to the existing file. If false, the log file will be overwritten.")
        vprint(f"show_timestamp = {self.show_timestamp}. If true, the printed information will contain a timestamp.")
        vprint(' ')

    def flush_to_file(self, log_dir=None, append_to_file=None):
        """
        Flushes buffered logs to a file based on user-defined file mode (append or write)
        """
        
        # Set log_dir
        if log_dir is None:
            if self.log_dir == 'auto':
                log_dir = 'logs'
            else:
                log_dir = self.log_dir

        # Set file_mode
        if append_to_file is None:
            append_to_file = self.append_to_file
        file_mode = 'a' if append_to_file else 'w'
        
        # Set file name
        log_file = self.log_file
        if self.prefix_jobid != 0:
            log_file = str(self.prefix_jobid).zfill(2) + '_' + log_file
        if self.prefix_date:
            log_file = get_date() + '_' + log_file
        
        show_timestamp = self.show_timestamp
        
        if self.flush_file:
            # Ensure the log directory exists
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, log_file)

            # Write the buffered logs to the specified file
            with open(log_file_path, file_mode) as f:
                f.write(self.log_buffer.getvalue())

            # Clear the buffer
            self.log_buffer.truncate(0)
            self.log_buffer.seek(0)

            # Set up a file handler for future logging to the file
            self.file_handler = logging.FileHandler(log_file_path, mode='a')  # Always append after initial flush
            self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s' if show_timestamp else '%(message)s'))
            self.logger.addHandler(self.file_handler)
            vprint(f"### Log file is flushed (created) as {log_file_path} ###")
        else:
            self.file_handler = None
            vprint(f"### Log file is not flushed (created) because log_file is set to {self.log_file} ###")
        vprint(' ')
        
    def close(self):
        """Closes the file handler if it exists."""
        if self.file_handler is not None:
            self.file_handler.flush()
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
            self.file_handler = None

def print_system_info():
    import os
    import platform
    import sys
    
    vprint("### System information ###")
    
    # Operating system information
    vprint(f"Operating System: {platform.system()} {platform.release()}")
    vprint(f"OS Version: {platform.version()}")
    vprint(f"Machine: {platform.machine()}")
    vprint(f"Processor: {platform.processor()}")
    
    # CPU cores
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        cpus =  int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        # Fallback to the total number of CPU cores on the node
        cpus = os.cpu_count()
    vprint(f"Available CPU cores: {cpus}")
    
    # Memory information
    if 'SLURM_MEM_PER_NODE' in os.environ:
        # Memory allocated per node by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_NODE']) / 1024  # Convert MB to GB
        vprint(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    elif 'SLURM_MEM_PER_CPU' in os.environ:
        # Memory allocated per CPU by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_CPU']) * cpus / 1024  # Convert MB to GB
        vprint(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    else:
        try:
            import psutil
            # Fallback to system memory information
            mem = psutil.virtual_memory()
            vprint(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
            vprint(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
        except ImportError:
            vprint("Memory information will be available after `conda install conda-forge::psutil`")
    vprint(" ")
            
    # GPU information
    print_gpu_info()
    vprint(" ")
    
    # Python version and executable
    vprint("### Python information ###")
    vprint(f"Python Executable: {sys.executable}")
    vprint(f"Python Version: {sys.version}")
    vprint(" ")
    
    # Packages information (numpy, PyTorch, Optuna, Accelerate, PtyRAD)
    print_packages_info()
    vprint(" ")

def print_gpu_info():
    vprint("### GPU information ###")
    try:
        import torch
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            vprint(f"CUDA Available: {torch.cuda.is_available()}")
            vprint(f"CUDA Version: {torch.version.cuda}")
            vprint(f"Available CUDA GPUs: {[torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]}")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            vprint(f"MPS Available: {torch.backends.mps.is_available()}")
        elif torch.backends.cuda.is_built() or torch.backends.mps.is_built():
            vprint("WARNING: GPU support built with PyTorch, but could not find any existing / compatible GPU device.")
            vprint("         PtyRAD will fall back to CPU which is much slower in performance")
            vprint("         → If you're using a CPU-only machine, you can safely ignore this.")
            vprint("         → If you believe you *do* have a GPU, please check the compatibility:")
            vprint("           - Are the correct NVIDIA drivers installed?")
            vprint("           - Is your CUDA runtime version compatible with PyTorch?")
            vprint("           Tips: Run `nvidia-smi` in your terminal for NVIDIA driver and CUDA runtime information.")
            vprint("           Tips: Run `conda list torch` in your terminal (with `ptyrad` environment activated) to check the installed PyTorch version.")
        else:
            vprint("WARNING: No GPU backend (CUDA or MPS) built into this PyTorch install.")
            vprint("         PtyRAD will fall back to CPU which is much slower in performance")
            vprint("         Please consider reinstalling PyTorch with GPU support if available.")
            vprint("         See https://github.com/chiahao3/ptyrad for PtyRAD installation guide.")
    except ImportError:
        vprint("WARNING: No GPU information because PyTorch can't be imported.")
        vprint("         Please install PyTorch because it's the crucial dependency of PtyRAD.")
        vprint("         See https://github.com/chiahao3/ptyrad for PtyRAD installation guide.")
    
def print_packages_info():
    import importlib
    import importlib.metadata
    vprint("### Packages information ###")
    
    # Print package versions
    packages = [
        ("Numpy", "numpy"),
        ("PyTorch", "torch"),
        ("Optuna", "optuna"),
        ("Accelerate", "accelerate"),
    ]

    # Check versions for relevant packages
    for display_name, module_name in packages:
        try:
            # Try to get the version from package metadata (installed version)
            version = importlib.metadata.version(module_name)
            vprint(f"{display_name} Version (metadata): {version}")
        except importlib.metadata.PackageNotFoundError:
            vprint(f"{display_name} not found in the environment.")
        except Exception as e:
            vprint(f"Error retrieving version for {display_name}: {e}")
    
    # Check the version and path of the used PtyRAD package
    # Note that we're focusing on the version/path of the actual imported PtyRAD.
    # If there are both an installed version of PtyRAD in the environment and a local copy in the working directory,
    # Python will prioritize the version in the working directory.
    #
    # When using `pip install -e .`, only the version metadata gets recorded, which won't be updated until you reinstall.
    # As a result, a user who pulls new code from the repo will have their `__init__.py` updated, but the version metadata recorded by pip will remain unchanged.
    # Therefore, it is better to retrieve the version directly from `module.__version__` for now, as this will reflect the actual local version being used.
    # Once we transition to using pip/conda for installation, all code updates will be paired with an installation, 
    # and we can safely switch to retrieving the version via `importlib.metadata.version`.
    try:
        # Import ptyrad (which will prioritize the local version if available)
        module = importlib.import_module('ptyrad')
        vprint(f"PtyRAD Version (direct import): {module.__version__}") # This version is defined in __init__.py
        vprint(f"PtyRAD is located at: {module.__file__}")
    except ImportError:
        vprint("PtyRAD not found locally")
    except AttributeError:
        vprint("PtyRAD imported, but no __version__ attribute found.")
    except Exception as e:
        vprint(f"Error retrieving version for PtyRAD: {e}")

def set_gpu_device(gpuid=0):
    """
    Sets the GPU device based on the input. If 'acc' is passed, it returns None to defer to accelerate.
    
    Args:
        gpuid (str or int): The GPU ID to use. Can be:
            - "acc": Defer device assignment to accelerate.
            - "cpu": Use CPU.
            - An integer (or string representation of an integer) for a specific GPU ID. This only has effect on NVIDIA GPUs.
    
    Returns:
        torch.device or None: The selected device, or None if deferred to accelerate.
    """
    vprint("### Setting GPU Device ###")

    if gpuid == "acc":
        vprint("Specified to use accelerate device (gpuid='acc')")
        vprint(" ")
        return None
    
    if gpuid == "cpu":
        device = torch.device("cpu")
        torch.set_default_device(device)
        vprint("Specified to use CPU (gpuid='cpu').")
        vprint(" ")
        return device

    try:
        gpuid = int(gpuid)
        if torch.cuda.is_available():
            num_cuda_devices = torch.cuda.device_count()
            if gpuid < num_cuda_devices:
                device = torch.device(f"cuda:{gpuid}")
                torch.set_default_device(device)
                vprint(f"Selected GPU device: {device} ({torch.cuda.get_device_name(gpuid)})")
                vprint(" ")
                return device
            
            else:
                device = torch.device("cuda")
                vprint(f"Requested CUDA device cuda:{gpuid} is out of range (only {num_cuda_devices} available). " 
                    f"Fall back to GPU device: {device}")
                vprint(" ")
                return device
            
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            torch.set_default_device(device)
            vprint("Selected GPU device: MPS (Apple Silicon)")
            vprint(" ")
            return device
        
        else:
            device = torch.device("cpu")
            torch.set_default_device(device)
            vprint(f"GPU ID specifed as {gpuid} but no GPU found. Using CPU instead.")
            vprint(" ")
            return device
        
    except ValueError:
        raise ValueError(f"Invalid gpuid '{gpuid}'. Expected 'acc', 'cpu', or an integer.")

def vprint(*args, verbose=True, **kwargs):
    """Verbose print/logging with individual control, only for rank 0 in DDP."""
    if verbose and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
        logger = logging.getLogger('PtyRAD')
        if logger.hasHandlers():
            logger.info(' '.join(map(str, args)), **kwargs)
        else:
            print(*args, **kwargs)

def vprint_nested_dict(d, indent=0, verbose=True, leaf_inline_threshold=6):
    indent_str = "    " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            # Check if this is a flat leaf dict
            is_flat_leaf = all(not isinstance(v, (dict, list)) for v in value.values())
            if is_flat_leaf and len(value) <= leaf_inline_threshold:  # Determine whether to print inline or not
                flat = ", ".join(f"{k}: {repr(v)}" for k, v in value.items())
                vprint(f"{indent_str}{key}: {{{flat}}}", verbose=verbose)
            else:
                vprint(f"{indent_str}{key}:", verbose=verbose)
                vprint_nested_dict(value, indent + 1, verbose=verbose)
        elif isinstance(value, list) and all(not isinstance(i, (dict, list)) for i in value):
            vprint(f"{indent_str}{key}: {value}", verbose=verbose)
        else:
            vprint(f"{indent_str}{key}: {repr(value)}", verbose=verbose)

def safe_get_nested(d, keys, default=None):
    """
    Safely get a value from a nested dictionary.
    
    Parameters:
    - d (dict): The dictionary to traverse.
    - keys (list or tuple): A sequence of keys to access nested values.
    - default: The value to return if any key is missing or intermediate value is None.
    
    Returns:
    - The nested value if found, otherwise `default`.
    """
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
        if d is None:
            return default
    return d

def get_date(date_format='%Y%m%d'):
    from datetime import date, datetime
    
    # If the format includes time-specific placeholders, return full datetime
    if any(fmt in date_format for fmt in ['%H', '%M', '%S']):
        return datetime.now().strftime(date_format)
    
    # Otherwise, just return the date
    return date.today().strftime(date_format)

def time_sync():
    # PyTorch doesn't have a direct exposed API to check the selected default device 
    # so we'll be checking these .is_available() just to prevent error.
    # Luckily these checks won't really affect the performance.
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Check if MPS (Metal Performance Shaders) is available (macOS only)
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    # Measure the time
    t = perf_counter()
    return t

def parse_sec_to_time_str(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if days > 0:
        return f"{int(days)} day {int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif hours > 0:
        return f"{int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif minutes > 0:
        return f"{int(minutes)} min {secs:.3f} sec"
    else:
        return f"{secs:.3f} sec"

def parse_hypertune_params_to_str(hypertune_params):
    
    hypertune_str = ''
    for key, value in hypertune_params.items():
        if key[-2:].lower() == "lr":
            hypertune_str += f"_{key}_{value:.1e}"
        elif isinstance(value, (int, float)):
            hypertune_str += f"_{key}_{value:.3g}"
        else:
            hypertune_str += f"_{key}_{value}"
    
    return hypertune_str

def safe_filename(filepath, verbose=False):
    """
    Ensures a filepath is safe across platforms by:
    1. Converting relative paths to absolute
    2. Limiting individual components to 255 characters
    3. Handling total path length restrictions
    4. Providing feedback when corrections are made
    
    Args:
        filepath: The original filepath to make safe
        verbose: Whether to print messages about corrections (default: False)
    
    Returns:
        A modified filepath that should work across platforms
    """
    # Store original path for reporting
    original_path = filepath
    
    # Handle relative paths by converting to absolute
    filepath = os.path.abspath(filepath)
    
    # Platform detection
    is_windows = platform.system() == 'Windows'
    
    # Check if path already has long path prefix on Windows
    has_long_prefix = is_windows and filepath.startswith("\\\\?\\")
    
    # Early return if path is already valid
    if not has_long_prefix:
        # Check individual component limit (255 chars)
        components_valid = True
        sep = '\\' if is_windows else '/'
        parts = filepath.split(sep)
        for part in parts:
            if len(part) > 255:
                components_valid = False
                break
        
        # Check total path length limit
        length_valid = (len(filepath) <= 260) if is_windows else True
        
        # If everything is valid, return the absolute path
        if components_valid and length_valid:
            return filepath
    
    # Path requires correction - continue with fixing logic
    # Path separator based on platform
    sep = '\\' if is_windows else '/'
    
    # Split path into directory and filename
    directory, filename = os.path.split(filepath)
    
    # Track if any changes were made
    changes_made = False
    
    # Limit filename component to 255 chars (preserve extension)
    if len(filename) > 255:
        changes_made = True
        name, ext = os.path.splitext(filename)
        max_name_length = 255 - len(ext)
        filename = name[:max_name_length] + ext
    
    # Handle directory components (limit each to 255 chars)
    if directory:
        parts = directory.split(sep)
        for i, part in enumerate(parts):
            if len(part) > 255:
                changes_made = True
                parts[i] = part[:255]
        directory = sep.join(parts)
    
    # Recombine path
    result_path = os.path.join(directory, filename)
    
    # Handle Windows total path length
    if is_windows and len(result_path) > 260:
        changes_made = True
        # If still too long, apply the \\?\ prefix for long path support
        if not result_path.startswith("\\\\?\\"):
            # Ensure we're working with an absolute path for the \\?\ prefix
            result_path = "\\\\?\\" + os.path.abspath(result_path)
    
    # Provide feedback if corrections were made
    if changes_made and verbose:
        print("Path corrected for compatibility:")
        print(f"  Original: {original_path}")
        print(f"  Corrected: {result_path}")
    
    return result_path