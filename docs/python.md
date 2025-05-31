# Python Interface

*PtyRAD* provides an easy Python interface for programmatically control of  `ptyrad` in your Python environment.

**Run reconstructions**
```python
from ptyrad.load import load_params
from ptyrad.reconstruction import PtyRADSolver
from ptyrad.utils import print_system_info, set_gpu_device, CustomLogger

logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_date=True, append_to_file=True, show_timestamp=True)

params_path = "params/tBL_WSe2_reconstruct.yml"

print_system_info()
params = load_params(params_path)
device = set_gpu_device(gpuid=0)

ptycho_solver = PtyRADSolver(params, device=device, logger=logger)

ptycho_solver.run()
```

This is the same example of `demo/scripts/run_ptyrad_quick_example.ipynb`.