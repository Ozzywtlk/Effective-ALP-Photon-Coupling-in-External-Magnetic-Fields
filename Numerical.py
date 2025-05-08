#%%
from numpy import degrees , sqrt, cos, sin
import numpy as np
import vegas
from math import pi
import matplotlib.pyplot as plt
import ctypes
import os
import multiprocessing
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset # Import necessary functions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from fractions import Fraction
import matplotlib.ticker as mticker
#%%
class ComplexDouble(ctypes.Structure):
    """Represents C double complex using ctypes."""
    _fields_ = [("real", ctypes.c_double),
                ("imag", ctypes.c_double)]
    # --- Constructor and methods as provided by user ---
    def __init__(self, value=0+0j):
        """Initialize from Python complex, float, int, or another ComplexDouble."""
        if isinstance(value, (complex, int, float)):
            py_complex = complex(value)
            super().__init__(py_complex.real, py_complex.imag)
        elif isinstance(value, ComplexDouble):
            super().__init__(value.real, value.imag)
        elif isinstance(value, tuple) and len(value) == 2:
            super().__init__(float(value[0]), float(value[1]))
        else:
             super().__init__() # Initialize to 0.0, 0.0


    def to_python(self):
        """Convert ctypes structure back to a Python complex number."""
        return complex(self.real, self.imag)

    def __complex__(self):
        """Allows direct conversion using complex()."""
        return self.to_python()

    def __repr__(self):
        """Provide a helpful string representation."""
        # Check if _b_base_ is None which happens before __init__ completes in some cases
        if getattr(self, '_b_base_', None) is None and not hasattr(self, 'real'):
             return f"ComplexDouble(<pre-init>)"
        # Check if fields are initialized
        # A simple check might be if they are not the default 0.0 - adjust if needed
        # Or rely on a flag set during proper init, but simple check for now:
        try:
             # Attempt to access to ensure initialized
             _ = self.real
             _ = self.imag
             return f"ComplexDouble({self.to_python()})"
        except ValueError: # This might occur if fields are not properly initialized
             return f"ComplexDouble(<uninitialized>)"
        except AttributeError: # If fields don't exist yet
             return f"ComplexDouble(<fields-missing>)"


    def __str__(self):
         return str(self.to_python())
#These below represent K vectors
def sec(angle):
    return 1.0 / cos(angle)

def QK(Ea,Ma,theta):
    return [Ea/sqrt(Ma**2), sqrt((Ea**2 - 1.0*Ma**2)*cos(theta)**2)/sqrt(Ma**2), 0.0, sqrt((Ea**2 - 1.0*Ma**2)*sin(theta)**2)/sqrt(Ma**2)]

def LK(Ea,Ma,theta):
    return [0,0,1,0]

def LTK(Ea,Ma,theta):
    return [-1.0*sqrt((Ea**2 - 1.0*Ma**2)*sin(theta)**2)/sqrt(Ea**2 - 1.0*(Ea**2 - 1.0*Ma**2)*sin(theta)**2), 0.0, 0.0, Ea/sqrt(Ea**2 - 1.0*(Ea**2 - 1.0*Ma**2)*sin(theta)**2)]

def GK(Ea,Ma,theta):
    return (Ea/sqrt(Ma**4*sec(theta)**2/(Ea**2 - Ma**2) + Ma**2), sqrt((Ea - Ma)*(Ea + Ma)*cos(theta)**2)*sqrt(Ma**4*sec(theta)**2/(Ea**2 - Ma**2) + Ma**2)/Ma**2, 0, sqrt((Ea - Ma)*(Ea + Ma)*sin(theta)**2)/sqrt(Ma**4*sec(theta)**2/(Ea**2 - Ma**2) + Ma**2))

#Below are P1 vectors
def LP1(Ea,Ma,theta,psi,phi):
    return [0.0, 1.0*Ma**2*sin(phi)*cos(psi)/(sqrt(Ma**4*cos(psi)**2/(1.0*Ea - 1.0*sqrt(-(-1.0*Ea**2 + Ma**2)*sin(theta)**2)*sin(psi) - 1.0*sqrt(-(-1.0*Ea**2 + Ma**2)*cos(theta)**2)*cos(phi)*cos(psi))**2)*(-1.0*Ea + 1.0*sqrt(-(-1.0*Ea**2 + Ma**2)*sin(theta)**2)*sin(psi) + 1.0*sqrt(-(-1.0*Ea**2 + Ma**2)*cos(theta)**2)*cos(phi)*cos(psi))), -1.0*Ma**2*cos(phi)*cos(psi)/(sqrt(Ma**4*cos(psi)**2/(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi))**2)*(-1.0*Ea + 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) + 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi))), 0.0]

def LTP1(Ea,Ma,theta,psi,phi):
    return [1.0*Ma**2*sin(psi)/(sqrt(-Ma**4*(sin(psi)**2 - 1.0)/(-1.0*Ea + 1.0*sqrt(-(-1.0*Ea**2 + Ma**2)*sin(theta)**2)*sin(psi) + 1.0*sqrt(-(-1.0*Ea**2 + Ma**2)*cos(theta)**2)*cos(phi)*cos(psi))**2)*(-1.0*Ea + 1.0*sqrt(-(-1.0*Ea**2 + Ma**2)*sin(theta)**2)*sin(psi) + 1.0*sqrt(-(-1.0*Ea**2 + Ma**2)*cos(theta)**2)*cos(phi)*cos(psi))), 0.0, 0.0, 0.5*Ma**2/(sqrt(Ma**4*(0.25 - 0.25*sin(psi)**2)/(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi))**2)*(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi)))]

#Below are P2 vectors
def LP2(Ea,Ma,theta,psi,phi):
    return [0.0, -0.5*Ma**2*sin(phi)*cos(psi)/(sqrt(0.25*Ma**4*sin(phi)**2*cos(psi)**2/(Ea - sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi))**2 + 1.0*(1.0*sqrt((Ea**2 - 1.0*Ma**2)*cos(theta)**2)*(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi)) + (-0.5*Ma**2 + (-1.0*Ea**2 + 1.0*Ma**2)*cos(theta)**2)*cos(phi)*cos(psi))**2/(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi))**2)*(-1.0*Ea + 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) + 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi))), (0.5*Ma**2*cos(phi)*cos(psi)/(-1.0*Ea + 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) + 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi)) + sqrt((Ea**2 - 1.0*Ma**2)*cos(theta)**2))/sqrt(0.25*Ma**4*sin(phi)**2*cos(psi)**2/(Ea - sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi))**2 + 1.0*(1.0*sqrt((Ea**2 - 1.0*Ma**2)*cos(theta)**2)*(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi)) + (-0.5*Ma**2 + (-1.0*Ea**2 + 1.0*Ma**2)*cos(theta)**2)*cos(phi)*cos(psi))**2/(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi))**2), 0.0]

def LTP2(Ea,Ma,theta,psi,phi):
    return [(Ma**2*sin(psi)/(2.0*Ea - 2.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 2.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi)) - 1.0*sqrt((Ea**2 - 1.0*Ma**2)*sin(theta)**2))/sqrt((Ea - 0.5*Ma**2/(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi)))**2 - 1.0*(1.0*Ea*sqrt((Ea**2 - 1.0*Ma**2)*sin(theta)**2) - 0.5*Ma**2*sin(psi) - 1.0*sqrt((Ea**2 - 1.0*Ma**2)*sin(theta)**2)*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi) - 1.0*(Ea**2 - 1.0*Ma**2)*sin(psi)*sin(theta)**2)**2/(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi))**2), 0.0, 0.0, (Ea - 0.5*Ma**2/(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi)))/sqrt((Ea - 0.5*Ma**2/(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi)))**2 - 1.0*(1.0*Ea*sqrt((Ea**2 - 1.0*Ma**2)*sin(theta)**2) - 0.5*Ma**2*sin(psi) - 1.0*sqrt((Ea**2 - 1.0*Ma**2)*sin(theta)**2)*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi) - 1.0*(Ea**2 - 1.0*Ma**2)*sin(psi)*sin(theta)**2)**2/(1.0*Ea - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*sin(theta)**2)*sin(psi) - 1.0*sqrt((Ea - 1.0*Ma)*(Ea + Ma)*cos(theta)**2)*cos(phi)*cos(psi))**2)]
#%%
num_psi=6
num_phi=6
num_theta_points = 101
#%%
# --- DLL Loading ---
os.add_dll_directory(r"path/to/dlls")

# Load the DLLs using ctypes.
dll_pathTHETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "dirname", "name_of_your_dll.dll")



my_libTHETA_PSI_PHI = ctypes.WinDLL(dll_pathCHETA_PSI_PHI)



# --- End DLL Loading ---

# Create function pointers for the C functions in the DLL, make one for each type of function. Here we have left a single example
int5ea1936eBTHPSIPHI = [[[[None for _ in range(num_phi+1)] for _ in range(num_psi+1)] for _ in range(num_theta_points)] for _ in range(5)]



try:
    for i in range(1,5):
        for j in range(0,num_theta_points):
            for psi in range(0,num_psi+1):
                for nu in range(1,5):
                    for rho in range(1,5):
                        int5ea1936eBTHPSIPHI[i][nu][rho][j][psi].argtypes = [ctypes.c_double] * 2
                        intcea1936eBTHPSIPHI[i][nu][rho][j][psi].restype = ComplexDouble


except OSError as e:
    print(f"Error loading DLL: {e}")
    print("Make sure the DLL exists and that both Python and the DLL are 64-bit.")
    raise
except Exception as e:
    print(f"An unexpected error occurred during DLL loading: {e}")
    raise
# --- End Function Pointer Setup ---

#%%
#----------------Function Definitions------------------
def make_vegas_integrands_real_imag(c_func_pointer):
    """ Creates real/imaginary integrand wrappers for a C function pointer """
    if c_func_pointer is None:
         raise ValueError("Provided C function pointer is None")

    # These inner functions should be pickleable
    def vegas_integrand_real_part(u_batch: np.ndarray):
        # Directly call the C function pointer with unpacked arguments
        c_result = c_func_pointer(*u_batch)
        # Convert C complex result to Python float (real part)
        return c_result.real

    def vegas_integrand_imag_part(u_batch: np.ndarray):
        # Directly call the C function pointer with unpacked arguments
        c_result = c_func_pointer(*u_batch)
        # Convert C complex result to Python float (imaginary part)
        return c_result.imag

    return vegas_integrand_real_part, vegas_integrand_imag_part


def process_one_theta(task_args):
    """Worker function for multiprocessing. Integrates case for one theta value."""
    i, current_theta_rad, psi_idx, phi_idx, ndim, num_iterations, neval_per_iter, func_type = task_args


    # Results for this specific theta (indices unused)
    local_means = [[[np.nan + 0j for _ in range(5)] for _ in range(5)] for _ in range(5)]
    local_sdevs = [[[np.nan + 0j for _ in range(5)] for _ in range(5)] for _ in range(5)]
    result_real_vegas = None # Keep track of one vegas result for return

    for mu in range(1, 5):
        for nu in range(1, 5):
            for rho in range(1, 5):
                result_real = None
                result_imag = None
                integration_success = False
                c_func = None

                # Select the specific C function pointer
                try:
                    match func_type:
                        case "example":
                            c_func = int5ea1936eBTHPSIPHI[mu][nu][rho][i][psi_idx] 
                except IndexError:
                    continue # Skip if indices are invalid
                except Exception as e:
                    print(f"  [Proc {os.getpid()}] Error accessing function pointer for, mu={mu}, nu={nu}, rho={rho}, i={i}: {e}. Skipping.")
                    continue

                if c_func is None:
                    continue # Skip this combination if pointer is None

                # --- Integration ---
                max_retries = 15 # Add a limit to prevent infinite loops
                retries = 0
                integration_success = False

                while not integration_success and retries < max_retries:
                    try:
                        # Create integrand wrappers
                        integrand_real, integrand_imag = make_vegas_integrands_real_imag(c_func)

                        # Create NEW Integrator instances for this attempt
                        integ_real = vegas.Integrator([[0.05, 0.9]] * ndim)
                        integ_imag = vegas.Integrator([[0.05, 0.9]] * ndim)

                        # Integrate Real Part
                        result_real = integ_real(integrand_real, nitn=num_iterations, neval=neval_per_iter)
                        if mu == 1 and nu == 1 and rho == 1 and retries == 0: # Store first attempt's result
                             result_real_vegas = result_real

                        # Integrate Imaginary Part
                        result_imag = integ_imag(integrand_imag, nitn=num_iterations, neval=neval_per_iter)

                        # Check if results are valid vegas objects and contain non-NaN mean/sdev
                        if (hasattr(result_real, 'mean') and hasattr(result_imag, 'mean') and
                            hasattr(result_real, 'sdev') and hasattr(result_imag, 'sdev') and
                            not np.isnan(result_real.mean) and not np.isnan(result_imag.mean) and
                            not np.isnan(result_real.sdev) and not np.isnan(result_imag.sdev)):
                            integration_success = True # Exit loop if successful
                        else:
                            retries += 1

                    except Exception as e:
                        retries += 1
                        result_real = None
                        result_imag = None


                # Store results only if successful (non-NaN) after retries
                if integration_success:
                    local_means[mu][nu][rho] = result_real.mean + 1j * result_imag.mean
                    local_sdevs[mu][nu][rho] = result_real.sdev + 1j * result_imag.sdev


    # Return tuple includes indices and results for this task
    return (i, psi_idx, phi_idx, local_means, local_sdevs, result_real_vegas)

#%%

# ---------------------------------------INTEGRATION-----------------------------------


if __name__ == '__main__':
    multiprocessing.freeze_support() # For Windows compatibility / frozen apps

    EPS=np.finfo(np.float64).eps
    ndim = 2 # Number of dimensions for Vegas integration (should match C func args)
    neval_per_iter = 200 #672
    num_iterations = 24 # User's value
     # User's value
    theta_values_rad = (np.exp(-3.5*np.linspace(0+EPS, 1-EPS, num_theta_points))-np.exp(-3.5))* (pi / 2.0)/(np.exp(-3.5)-1)  

    # Initialize result storage (filled with NaN initially)

    results_mean_complex=[[[[[np.nan + 0j] * num_theta_points for _ in range(5)] for _ in range(5)] for _ in range(5)] for _ in range(num_psi+1)]

    # 1. Prepare Tasks for Multiprocessing

    tasks=[
        (i, current_theta_rad, current_psi_idx, current_phi_idx, ndim, num_iterations, neval_per_iter, "example")# change "example" with the functions you want to test
        for i, current_theta_rad in enumerate(theta_values_rad)
        for current_psi_idx in range(num_psi + 1) # Iterate psi from 0 to 5
        for current_phi_idx in range(6, 7) # Iterate phi from 1 to 5
    ]
    print(f"Prepared {len(tasks)} tasks for parallel execution.")


    
# 2. Determine number of processes
    n_cores = multiprocessing.cpu_count()
    print(f"Starting parallel Vegas integration using {n_cores} processes.")
    print(f"Vegas settings per task: nitn={num_iterations}, neval={neval_per_iter}")

    # 3. Run in Parallel (or sequentially if n_processes=1)
    start_time = time.time()
    all_pool_results = []

    with multiprocessing.Pool(processes=n_cores) as pool: # Use n_cores for actual parallel
        # Use map instead of starmap since worker takes a single tuple argument
        all_pool_results = pool.map(process_one_theta, tasks)
    end_time = time.time()
    print(f"\nParallel integration finished in {end_time - start_time:.2f} seconds.")


   
    # 4. Collect and Organize Results
    print("Organizing results...")
    for result_tuple in all_pool_results:
        # Place results into the correct index 'i'
        i,psi_i,phi_i, local_means, local_sdevs,resreal = result_tuple
        results_found_for_i = False
        for mu in range(1, 5):
            for nu in range(1, 5):
                for rho in range(1, 5):
                    results_mean_complex[psi_i][mu][nu][rho][i] = 2*local_means[mu][nu][rho]
    
    #####IMPORTANT: If the intagral over the imaginary line was done in Mathematica, combine the results here before proceeding.
    
    #%%

    # --- Plotting (using the collected results) ---
    print("Generating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 7)) # Create a single figure and 
    colors = plt.cm.viridis(np.linspace(0, 1, num_psi + 1)) # Get distinct colors

    fixed_phi_idx = 5 # User specified phi=5
    lines_data_for_inset = []
    lines_data_for_inset_raw = [] # Store raw data for inset fallback
    smoothed_lines_data_for_inset = [] # Store smoothed data for inset
    # Loop through each psi index
    for psi_idx in range(0,num_psi + 1): # psi = 0, 1, 2, 3, 4, 5
        abs_result_psi = [np.nan] * num_theta_points # Initialize results for this psi
        std_result_psi = [np.nan] * num_theta_points # Initialize std results for this psi

        # Loop through each theta index 'l'
        for l in range(num_theta_points):
            term_qk_LL = 0.0
            term_lk_LL = 0.0
            term_ltk_LL = 0.0
            term_gk_LL = 0.0
            term_qk_LTLT = 0.0
            term_lk_LTLT = 0.0
            term_ltk_LTLT = 0.0
            term_gk_LTLT = 0.0
            term_qk_SP = 0.0
            term_lk_SP = 0.0
            term_ltk_SP = 0.0
            term_gk_SP = 0.0

            std_qk_LL_Re=0.0
            std_lk_LL_Re=0.0
            std_ltk_LL_Re=0.0
            std_gk_LL_Re=0.0
            std_qk_LTLT_Re=0.0
            std_lk_LTLT_Re=0.0
            std_ltk_LTLT_Re=0.0
            std_gk_LTLT_Re=0.0
            std_qk_SP_Re=0.0
            std_lk_SP_Re=0.0
            std_ltk_SP_Re=0.0
            std_gk_SP_Re=0.0

            std_qk_LL_Im=0.0
            std_lk_LL_Im=0.0
            std_ltk_LL_Im=0.0
            std_gk_LL_Im=0.0
            std_qk_LTLT_Im=0.0
            std_lk_LTLT_Im=0.0
            std_ltk_LTLT_Im=0.0
            std_gk_LTLT_Im=0.0
            std_qk_SP_Im=0.0
            std_lk_SP_Im=0.0
            std_ltk_SP_Im=0.0
            std_gk_SP_Im=0.0


            valid_data_for_theta = True # Flag to check if data exists
            Ea_val = 5
            Ma_val = 1
            current_theta = theta_values_rad[l]
            qk_vec = QK(Ea_val,Ma_val, current_theta)
            lk_vec = LK(Ea_val,Ma_val, current_theta)
            ltk_vec = LTK(Ea_val,Ma_val, current_theta)
            gk_vec = GK(Ea_val,Ma_val, current_theta)

            lp1_vec = LP1(Ea_val, Ma_val, current_theta, psi_idx, fixed_phi_idx)
            lp2_vec = LP2(Ea_val, Ma_val, current_theta, psi_idx, fixed_phi_idx)
            ltp1_vec = LTP1(Ea_val, Ma_val, current_theta, psi_idx, fixed_phi_idx)
            ltp2_vec = LTP2(Ea_val, Ma_val, current_theta, psi_idx, fixed_phi_idx)
            
            # Loop through each result component 'k'
            for k in range(1, 5): # k = 1, 2, 3, 4
                
                # Get the complex mean result for this theta(l), component(k), psi_idx, and fixed phi_idx
                

                sdev_valLL = 0
                sdev_valLTLT = 0
                sdev_valSP = 0
                

                for nu in range(1, 5):
                    for rho in range(1, 5):
                        mean_val = results_mean_complex[psi_idx][k][nu][rho][l]
                        if np.isnan(mean_val):
                            valid_data_for_theta = False
                            break
                        if 0 <= k-1 < len(lp1_vec):
                            if 0 <= nu-1 < len(lp1_vec):
                                if 0 <= rho-1 < len(lp1_vec):
                                    ############IMPORTANT: CALCULATE THE ERRORS FROM THE NUMERICAL INTEGRATION HERE. THE TERMS HERE ARE SIMPLY EXAMPLES
                                    term_qk_LL += (mean_val) * qk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1]
                                    term_lk_LL += (mean_val) * lk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1]
                                    term_ltk_LL += (mean_val) * ltk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1]
                                    term_gk_LL += (mean_val) * gk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1]
                                    std_qk_LL_Re += np.real(sdev_val * qk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1])**2
                                    std_lk_LL_Re += np.real(sdev_val * lk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1])**2
                                    std_ltk_LL_Re += np.real(sdev_val * ltk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1])**2
                                    std_gk_LL_Re += np.real(sdev_val * gk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1])**2
                                    std_qk_LL_Im += np.imag(sdev_val * qk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1])**2
                                    std_lk_LL_Im += np.imag(sdev_val * lk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1])**2
                                    std_ltk_LL_Im += np.imag(sdev_val * ltk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1])**2
                                    std_gk_LL_Im += np.imag(sdev_val * gk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1])**2

                                    term_qk_LTLT += (mean_val) * qk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1]
                                    term_lk_LTLT += (mean_val) * lk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1]
                                    term_ltk_LTLT += (mean_val) * ltk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1]
                                    term_gk_LTLT += (mean_val) * gk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1]            
                                    std_qk_LTLT_Re += np.real(sdev_val * qk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1])**2
                                    std_lk_LTLT_Re += np.real(sdev_val * lk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1])**2
                                    std_ltk_LTLT_Re += np.real(sdev_val * ltk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1])**2
                                    std_gk_LTLT_Re += np.real(sdev_val * gk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1])**2
                                    std_qk_LTLT_Im += np.imag(sdev_val * qk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1])**2
                                    std_lk_LTLT_Im += np.imag(sdev_val * lk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1])**2
                                    std_ltk_LTLT_Im += np.imag(sdev_val * ltk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1])**2
                                    std_gk_LTLT_Im += np.imag(sdev_val * gk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1])**2

                                    term_qk_SP += (mean_val) * qk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2))
                                    term_lk_SP += (mean_val) * lk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2))
                                    term_ltk_SP += (mean_val) * ltk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2))
                                    term_gk_SP += (mean_val) * gk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2))
                                    std_qk_SP_Re += np.real(sdev_val * qk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2)))**2
                                    std_lk_SP_Re += np.real(sdev_val * lk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2)))**2
                                    std_ltk_SP_Re += np.real(sdev_val * ltk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2)))**2
                                    std_gk_SP_Re += np.real(sdev_val * gk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2)))**2
                                    std_qk_SP_Im += np.imag(sdev_val * qk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2)))**2
                                    std_lk_SP_Im += np.imag(sdev_val * lk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2)))**2
                                    std_ltk_SP_Im += np.imag(sdev_val * ltk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2)))**2
                                    std_gk_SP_Im += np.imag(sdev_val * gk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2)))**2
                                    

            # Store the sum for this theta point if data was valid
            if valid_data_for_theta:
                #Mean#
                abs_result_psi[l] = np.abs(term_qk_LL)**2 + np.abs(term_lk_LL)**2 + np.abs(term_ltk_LL)**2 + np.abs(term_gk_LL)**2 + np.abs(term_qk_LTLT)**2 + np.abs(term_lk_LTLT)**2 + np.abs(term_ltk_LTLT)**2 + np.abs(term_gk_LTLT)**2 + np.abs(term_qk_SP)**2 + np.abs(term_lk_SP)**2 + np.abs(term_ltk_SP)**2 + np.abs(term_gk_SP)**2
                ##Standard Deviation#
                std_result_psi[l] =sqrt(2*(std_qk_LL_Re**2+std_qk_LL_Im**2+std_lk_LL_Re**2+std_lk_LL_Im**2+std_ltk_LL_Re**2+std_ltk_LL_Im**2+std_gk_LL_Re**2+std_gk_LL_Im**2+std_qk_LTLT_Re**2+std_qk_LTLT_Im**2+std_lk_LTLT_Re**2+std_lk_LTLT_Im**2+std_ltk_LTLT_Re**2+std_ltk_LTLT_Im**2+std_gk_LTLT_Re**2+std_gk_LTLT_Im**2+std_qk_SP_Re**2+std_qk_SP_Im**2+std_lk_SP_Re**2+std_lk_SP_Im**2+std_ltk_SP_Re**2+std_ltk_SP_Im**2+std_gk_SP_Re**2+std_gk_SP_Im**2)+4*(std_qk_LL_Re*np.real(term_qk_LL)**2+std_qk_LL_Im*np.imag(term_qk_LL)**2+std_lk_LL_Re*np.real(term_lk_LL)**2+std_lk_LL_Im*np.imag(term_lk_LL)**2+std_ltk_LL_Re*np.real(term_ltk_LL)**2+std_ltk_LL_Im*np.imag(term_ltk_LL)**2+std_gk_LL_Re*np.real(term_gk_LL)**2+std_gk_LL_Im*np.imag(term_gk_LL)**2+std_qk_LTLT_Re*np.real(term_qk_LTLT)**2+std_qk_LTLT_Im*np.imag(term_qk_LTLT)**2+std_lk_LTLT_Re*np.real(term_lk_LTLT)**2+std_lk_LTLT_Im*np.imag(term_lk_LTLT)**2+std_ltk_LTLT_Re*np.real(term_ltk_LTLT)**2+std_ltk_LTLT_Im*np.imag(term_ltk_LTLT)**2+std_gk_LTLT_Re*np.real(term_gk_LTLT)**2+std_gk_LTLT_Im*np.imag(term_gk_LTLT)**2+std_qk_SP_Re*np.real(term_qk_SP)**2+std_qk_SP_Im*np.imag(term_qk_SP)**2+std_lk_SP_Re*np.real(term_lk_SP)**2+std_lk_SP_Im*np.imag(term_lk_SP)**2+std_ltk_SP_Re*np.real(term_ltk_SP)**2+std_ltk_SP_Im*np.imag(term_ltk_SP)**2+std_gk_SP_Re*np.real(term_gk_SP)**2+std_gk_SP_Im*np.imag(term_gk_SP)**2))
            # else it remains NaN
        
        # Prepare data for plotting (concatenate with mirrored part)
        # Filter out NaN values before flipping and concatenating to avoid issues
        valid_indices_raw = ~np.isnan(abs_result_psi) & ~np.isnan(std_result_psi) & (np.array(std_result_psi) > 1e-15)
        raw_data_exists = np.sum(valid_indices_raw) > 0
        valid_indices = ~np.isnan(abs_result_psi)
        valid_abs_result_psi = np.array(abs_result_psi)[valid_indices]
        valid_std_result_psi = np.array(std_result_psi)[valid_indices]
        valid_theta_values_rad = theta_values_rad[valid_indices]
        plot_theta_values_psi_raw = None
        plot_abs_results_psi_raw = None
        plot_std_results_psi_raw = None

        if raw_data_exists:
            valid_abs_result_psi_raw = np.array(abs_result_psi)[valid_indices_raw]
            valid_std_result_psi_raw = np.array(std_result_psi)[valid_indices_raw]
            valid_theta_values_rad_raw = theta_values_rad[valid_indices_raw]
            # Mirror only the valid raw results
            plot_abs_results_psi_raw =  np.concatenate((valid_abs_result_psi_raw, np.flip(valid_abs_result_psi_raw)[1:]))
            plot_std_results_psi_raw = 5*np.concatenate((valid_std_result_psi_raw, np.flip(valid_std_result_psi_raw)[1:])) # Apply scaling if needed
            plot_theta_values_psi_raw = np.concatenate((valid_theta_values_rad_raw, -np.flip(valid_theta_values_rad_raw)[1:]))

        # --- Calculate psi label string ---
        angle_rad = -pi/2 + psi_idx * (pi / num_psi)
        psi_label_val_str = ""
        tolerance = 1e-9 # Tolerance for floating point comparison

        if abs(angle_rad + pi/2) < tolerance:
            psi_label_val_str = r"-\pi/2"
        elif abs(angle_rad - pi/2) < tolerance:
            psi_label_val_str = r"\pi/2"
        elif abs(angle_rad) < tolerance:
            psi_label_val_str = r"0"
        else:
            # Calculate fraction of pi
            from fractions import Fraction # Import here if not done globally
            frac = Fraction(angle_rad / pi).limit_denominator()
            num = frac.numerator
            den = frac.denominator
            # Format the fraction string
            if den == 1:
                if num == 1: psi_label_val_str = r"\pi"
                elif num == -1: psi_label_val_str = r"-\pi"
                else: psi_label_val_str = fr"{num}\pi"
            elif num == 1:
                psi_label_val_str = fr"\pi/{den}"
            elif num == -1:
                psi_label_val_str = fr"-\pi/{den}"
            else:
                psi_label_val_str = fr"{num}\pi/{den}"
        # --- Gaussian Process Smoothing ---
        valid_indices_smooth = ~np.isnan(abs_result_psi) & ~np.isnan(std_result_psi) & \
                            (np.array(abs_result_psi) > 1e-15) & (np.array(std_result_psi) > 1e-15)
        can_smooth = np.sum(valid_indices_smooth) >= 3 # Need a few points
        valid_abs_result_psi = np.array(abs_result_psi)[valid_indices_smooth]
        valid_std_result_psi = np.array(std_result_psi)[valid_indices_smooth]
        valid_theta_values_rad = theta_values_rad[valid_indices_smooth]
        plot_successful = False # Flag to track if anything was plotted

        if not can_smooth:
            print(f"Psi={psi_idx}: Skipping smoothing (insufficient points).")
            if raw_data_exists:
                print(f"Psi={psi_idx}: Plotting RAW data (smoothing skipped).")
                line, caps, bars = ax.errorbar(plot_theta_values_psi_raw, plot_abs_results_psi_raw, yerr=plot_std_results_psi_raw,
                                            label=fr"$\psi = {psi_label_val_str}$ (Raw)", color=colors[psi_idx],
                                            fmt='.', markersize=3, alpha=0.5, capsize=2)
                lines_data_for_inset_raw.append((plot_theta_values_psi_raw, plot_abs_results_psi_raw, plot_std_results_psi_raw, line.get_color()))
                plot_successful = True
            else:
                print(f"Psi={psi_idx}: No valid raw data to plot either.")

        else: # Attempt smoothing
            print(f"Psi={psi_idx}: Attempting GPR smoothing...")
            # Prepare data for GPR (unmirrored, valid points only)
            X_train = np.concatenate((valid_theta_values_rad, -np.flip(valid_theta_values_rad)[1:])).reshape(-1, 1)
            y_train_raw = np.concatenate((valid_abs_result_psi, np.flip(valid_abs_result_psi)[1:]))
            y_train_err_raw = np.concatenate((valid_std_result_psi, np.flip(valid_std_result_psi)[1:]))
            theta_smooth_pos = np.linspace(X_train.min(), X_train.max(), 200)

            # --- Log Transform ---
            log_y_train = np.log(y_train_raw)
            log_y_train_err = y_train_err_raw / y_train_raw
            alpha_log = log_y_train_err**2 + 1e-12 # Variance in log space

            # --- Define Kernel (Adjusted) ---
            # --- Define Kernel (Adjusted Bounds) ---
            initial_length_scale = 0.1 # Keep initial guess
            median_alpha_log = np.median(alpha_log)
            print(f"Psi={psi_idx}: Median alpha_log = {median_alpha_log:.2e}")

            # Increase upper bounds for C and RBF length_scale
            kernel = C(1.0, (1e-13, 1e25)) * RBF(length_scale=initial_length_scale, length_scale_bounds=(0.05, 5.0)) \
                   + WhiteKernel(noise_level=median_alpha_log, noise_level_bounds=(1e-10, 1e+3))

            # --- Instantiate GPR ---
            gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_log, n_restarts_optimizer=65, normalize_y=False)


            try:
                # Fit the GP to the LOGARITHM of the data
                
                
                
                gp.fit(X_train, log_y_train)
                print(f"Psi={psi_idx}: GPR fitted. Final kernel: {gp.kernel_}")

                # Create points for prediction (denser grid over positive theta range)
                
                X_smooth_pos = theta_smooth_pos.reshape(-1, 1)

                # Predict mean and std deviation IN LOG SPACE
                log_y_smooth_pos, log_sigma_smooth_pos = gp.predict(X_smooth_pos, return_std=True)
                # --- Transform back to original scale ---
                y_smooth_pos = (10**70)*1936**2*pi**2*np.exp(log_y_smooth_pos)/(4*10**32*10**72)
                y_smooth_lower = (10**70)*1936**2*pi**2*np.exp(log_y_smooth_pos - log_sigma_smooth_pos)/(4*10**32*10**72)
                y_smooth_upper = (10**70)*1936**2*pi**2*np.exp(log_y_smooth_pos + log_sigma_smooth_pos)/(4*10**32*10**72)
                y_smooth_lower = np.maximum(y_smooth_lower, 1e-15) # Ensure positive for log plot

                # --- Mirror the smoothed results ---
                if abs(theta_smooth_pos[0]) < 1e-9:
                    theta_smooth_full = theta_smooth_pos
                    y_smooth_full = y_smooth_pos
                    y_smooth_lower_full = y_smooth_lower
                    y_smooth_upper_full = y_smooth_upper
                else:
                    theta_smooth_full = theta_smooth_pos
                    y_smooth_full = y_smooth_pos
                    y_smooth_lower_full = y_smooth_lower
                    y_smooth_upper_full = y_smooth_upper

                # --- Diagnostic Print ---
                print(f"Psi={psi_idx}: Smoothed y range: [{np.min(y_smooth_full):.2e}, {np.max(y_smooth_full):.2e}]")

                # --- Plotting Smoothed Results (Separate Positive/Negative) ---
                print(f"Psi={psi_idx}: Plotting SMOOTHED data.")

                # Separate positive and negative data for plotting
                theta_pos = theta_smooth_pos
                y_pos = y_smooth_pos
                y_lower_pos = y_smooth_lower
                y_upper_pos = y_smooth_upper

                if abs(theta_smooth_pos[0]) < 1e-9: # Check if zero is included
                    theta_neg = -np.flip(theta_smooth_pos)[1:] # Exclude zero from negative side
                    y_neg = np.flip(y_smooth_pos)[1:]
                    y_lower_neg = np.flip(y_smooth_lower)[1:]
                    y_upper_neg = np.flip(y_smooth_upper)[1:]
                else:
                    theta_neg = -np.flip(theta_smooth_pos)
                    y_neg = np.flip(y_smooth_pos)
                    y_lower_neg = np.flip(y_smooth_lower)
                    y_upper_neg = np.flip(y_smooth_upper)

                # Plot positive side (this one gets the label)
                line_pos, = ax.plot(theta_pos, y_pos, color=colors[psi_idx], linestyle='-', linewidth=1, label=fr"$\psi = {psi_label_val_str}$")
                ax.fill_between(theta_pos, y_lower_pos, y_upper_pos, color=colors[psi_idx], alpha=0.2)

                # Plot negative side (same color, no label)
                ax.plot(theta_neg, y_neg, color=colors[psi_idx], linestyle='-', linewidth=1)
                ax.fill_between(theta_neg, y_lower_neg, y_upper_neg, color=colors[psi_idx], alpha=0.2)

                # Store FULL smoothed data for inset (using the original full arrays)
                smoothed_lines_data_for_inset.append((theta_smooth_full, y_smooth_full, y_smooth_lower_full, y_smooth_upper_full, line_pos.get_color())) # Use color from labeled line
                plot_successful = True

            except Exception as e:
                print(f"Psi={psi_idx}: ERROR during GPR: {e}.")
                if raw_data_exists:
                    print(f"Psi={psi_idx}: Plotting RAW data (smoothing failed).")
                    line, caps, bars = ax.errorbar(plot_theta_values_psi_raw, plot_abs_results_psi_raw, yerr=plot_std_results_psi_raw,
                                                label=fr"$\psi = {psi_label_val_str}$ (Raw - Smooth Failed)", color=colors[psi_idx],
                                                fmt='.', markersize=3, alpha=0.5, capsize=2)
                    lines_data_for_inset_raw.append((plot_theta_values_psi_raw, plot_abs_results_psi_raw, plot_std_results_psi_raw, line.get_color()))
                    plot_successful = True
                else:
                    print(f"Psi={psi_idx}: No valid raw data to plot on failure.")

        if not plot_successful:
            print(f"Psi={psi_idx}: Nothing was plotted.")


    # --- Create Inset Axes ---
    axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
    min_y_inset, max_y_inset = np.inf, -np.inf

    # Plot the SMOOTHED data onto the inset axes if available, otherwise RAW
    if smoothed_lines_data_for_inset:
        print("Plotting smoothed data in inset.")
        for theta_vals, smooth_vals, smooth_lower, smooth_upper, color in smoothed_lines_data_for_inset:
            axins.plot(theta_vals, smooth_vals, color=color, linestyle='-', linewidth=1.0)
            axins.fill_between(theta_vals, smooth_lower, smooth_upper, color=color, alpha=0.2)
            # Find data within the zoom range to set ylim based on smoothed data
            mask = (theta_vals >= -pi/6) & (theta_vals <= pi/6) # Adjust zoom range if needed
            vals_in_range = smooth_vals[mask]
            if vals_in_range.size > 0:
                min_y_inset = min(min_y_inset, np.min(vals_in_range))
                max_y_inset = max(max_y_inset, np.max(vals_in_range))
    elif lines_data_for_inset_raw: # Check if there's raw data to plot as fallback
        print("Plotting raw data in inset (fallback).")
        for theta_vals, abs_vals, std_vals, color in lines_data_for_inset_raw:
            axins.errorbar(theta_vals, abs_vals, yerr=std_vals, color=color,
                        linestyle='-', linewidth=1.0, marker='o', markersize=0, capsize=1)
            mask = (theta_vals >= -pi/6) & (theta_vals <= pi/6) # Adjust zoom range if needed
            valid_vals_in_range = abs_vals[mask]
            valid_positive_vals = valid_vals_in_range[valid_vals_in_range > 0]
            if valid_positive_vals.size > 0:
                min_y_inset = min(min_y_inset, np.min(valid_positive_vals))
                max_y_inset = max(max_y_inset, np.max(valid_positive_vals))
    else:
        print("No data available for inset plot.")


    # Set inset limits
    axins.set_xlim(-pi/6, pi/6) # Adjust zoom range if needed
    if np.isfinite(min_y_inset) and np.isfinite(max_y_inset) and min_y_inset > 0:
        axins.set_ylim(min_y_inset * 0.8, max_y_inset * 1.2) # Adjust padding
    else:
        print("Warning: Could not determine appropriate Y limits for inset.")
        main_ymin, main_ymax = ax.get_ylim()
        if np.isfinite(main_ymin) and np.isfinite(main_ymax):
            axins.set_ylim(main_ymin, main_ymax*0.1) # Fallback
        else:
            axins.set_ylim(1e-9, 1e-8) # Absolute fallback

    #axins.set_yscale('log') # Match main plot scale for inset
    axins.grid(True, which="both", ls=":", lw=0.5)
    axins.tick_params(axis='both', which='major', labelsize=8)
    axins.tick_params(axis='both', which='minor', labelsize=6)

    # Mark the region corresponding to the inset on the main plot
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # --- Format X-axis ticks as multiples of pi ---
    

    def multiple_formatter(x, pos):
        """Formats radians into multiples of pi."""
        den = 12 # Denominator for fraction representation
        num = round(x / (np.pi / den))
        f = Fraction(num, den).limit_denominator() # Simplify fraction
        n, d = f.numerator, f.denominator

        if n == 0:
            return "0"
        elif n == 1 and d == 1:
            return r"$\pi$"
        elif n == -1 and d == 1:
            return r"$-\pi$"
        elif d == 1:
            return fr"${n}\pi$"
        elif n == 1:
            return fr"$\pi/{d}$"
        elif n == -1:
            return fr"$-\pi/{d}$"
        else:
            return fr"${n}\pi/{d}$"

    # Apply the formatter to the main plot's x-axis
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(multiple_formatter))
    # Optionally, set specific tick locations (e.g., every pi/2)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(np.pi / 4))

    # Apply the formatter to the inset plot's x-axis
    axins.xaxis.set_major_formatter(mticker.FuncFormatter(multiple_formatter))
    # Optionally, set specific tick locations for the inset (e.g., every 0.25 or pi/4)
    axins.xaxis.set_major_locator(mticker.MultipleLocator(np.pi / 12)) # Example
    # Or keep default locator but use the pi formatter

    # --- Final Plot Setup ---
    # --- Final Plot Setup ---
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$|\Gamma|^2~[\textrm{neV}^{-8}]$")
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
   
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout before saving
   
    # --- Save plot for LaTeX (using PGF backend) ---
    try:
            # Configure Matplotlib to use PGF backend and LaTeX for text rendering
            plt.rcParams.update({
                "pgf.texsystem": "pdflatex",  # or xelatex, lualatex
                "text.usetex": True,          # Use LaTeX for text rendering
                "font.family": "serif",       # Match LaTeX document font (optional)
                "font.size": 12,              # Increase default font size
                "axes.labelsize": 15,         # Increase axes label size
                "legend.fontsize": 10,        # Increase legend font size
                "xtick.labelsize": 12,        # Increase x-tick label size
                "ytick.labelsize": 12,        # Increase y-tick label size
                "pgf.rcfonts": False,         # Don't setup fonts from rc parameters (handled by preamble)
                "pgf.preamble": r"""
                    \usepackage[utf8x]{inputenc}
                    \usepackage[T1]{fontenc}
                    \usepackage{amsmath}
                    \usepackage{amssymb}
                    \usepackage{cmbright} % Example font package, adjust as needed
                """ # Add necessary LaTeX packages here
            })

            output_filename_pgf = "plot_output.pgf"
            plt.savefig(output_filename_pgf)
            print(f"Plot saved as '{output_filename_pgf}' for LaTeX (PGF format).")

            # Optional: Save as PDF as well (often useful)
            output_filename_pdf = "plot_output.pdf"
            plt.savefig(output_filename_pdf)
            print(f"Plot also saved as '{output_filename_pdf}' (PDF format).")

    except Exception as e:
        print(f"Error saving plot for LaTeX: {e}")
        print("Make sure LaTeX and required packages (like cmbright) are installed and in PATH.")
        # Fallback to saving as a standard format if PGF fails
        try:
            output_filename_png = "plot_output.png"
            plt.savefig(output_filename_png, dpi=300)
            print(f"Saved plot as '{output_filename_png}' as a fallback.")
        except Exception as e_fallback:
            print(f"Error saving fallback PNG: {e_fallback}")

    # --- Show plot interactively ---
    # Note: The appearance might change slightly due to text.usetex=True
    
    plt.show()
    print("\nScript finished.")
# %%
