## Preprocess, parsing relevant functions

import numpy as np

def preprocess_ptycho_output_dict(ptycho_output_dict):
    probe  = ptycho_output_dict['probe']
    object = ptycho_output_dict['object']
    
    if 'p' in ptycho_output_dict and 'multi_slice_param' in ptycho_output_dict['p']:
        print("Handling multislice slice ptycho .mat")
        
        exp_params = {
        'lambd':           ptycho_output_dict['p']['lambda'],
        'dx_spec':         ptycho_output_dict['p']['dx_spec'],
        'z_distance_arr':  ptycho_output_dict['p']['multi_slice_param']['z_distance'],
        'N_scans':         ptycho_output_dict['outputs']['probe_positions'].shape[0],
        'probe_positions': ptycho_output_dict['outputs']['probe_positions'],
        'Nlayer':      len(ptycho_output_dict['p']['multi_slice_param']['z_distance']) - 1,
        }
        
    else:
        print("Handling single slice ptycho .mat")
        print(f"`z_distance_arr` and `Nlayer` is populted with 1 and 1")
    
        exp_params = {
        'lambd':           ptycho_output_dict['p']['lambda'],
        'dx_spec':         ptycho_output_dict['p']['dx_spec'],
        'z_distance_arr':  1,
        'N_scans':         ptycho_output_dict['outputs']['probe_positions'].shape[0],
        'probe_positions': ptycho_output_dict['outputs']['probe_positions'],
        'Nlayer':          1,
        }
    
    return probe, object, exp_params

def preprocess_CBED(CBED):
    
    # For abTEM simulated data there's no preprocessing needed
        
    # Config: Exp PSO    
    processed_CBED = np.transpose(CBED, axes=(0,2,1))
    print("Transposing from Matlab-generated HDF5 to Python orientation (N, ky, kx)")
    
    input_shape = processed_CBED.shape
    center = (input_shape[1]//2, input_shape[2]//2)
    height = input_shape[1]//2
    width = input_shape[2]//2
    row_start, row_end = center[0] - height//2, center[0] + height//2
    column_start, column_end = center[1] - width//2, center[1] + width//2
    
    processed_CBED = processed_CBED[:,row_start:row_end, column_start:column_end]
    print(f"Cropping padded CBED back to original dimension as {processed_CBED.shape}")
    
    return processed_CBED