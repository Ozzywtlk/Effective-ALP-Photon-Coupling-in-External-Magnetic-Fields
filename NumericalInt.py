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
num_psi=5
num_phi=5
num_theta_points = 81
#%%
# --- DLL Loading ---
os.add_dll_directory(r"C:\Users\ozzwt\mingw64\bin")
dll_pathLLTHETA = os.path.join(os.path.dirname(__file__), "intccll5ea44eBTHETA.dll")
dll_pathLLTHETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "intccll5ea44eBPsiPhiTHETA", "intccll5ea44eBPsiPhiTHETA.dll")
dll_pathLTLTTHETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "intccltlt5ea44eBPsiPhiTHETA", "intccltlt5ea44eBPsiPhiTHETA.dll")
dll_pathSPTHETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "intccsp5ea44eBPsiPhiTHETA", "intccsp5ea44eBPsiPhiTHETA.dll")


#dll_pathCCTHETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "intcc5ea44eBPsiPhiTHETA", "intcc5ea44eBPsiPhiTHETA.dll")
dll_pathDD0THETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "intdd5ea44eB0PsiPhiTHETA", "intdd5ea44eB0PsiPhiTHETA.dll")
dll_pathDD1THETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "intdd5ea44eB1PsiPhiTHETA", "intdd5ea44eB1PsiPhiTHETA.dll")
dll_pathDD2THETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "intdd5ea44eB2PsiPhiTHETA", "intdd5ea44eB2PsiPhiTHETA.dll")
dll_pathDD3THETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "intdd5ea44eB3PsiPhiTHETA", "intdd5ea44eB3PsiPhiTHETA.dll")
dll_pathDD4THETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "intdd5ea44eB4PsiPhiTHETA", "intdd5ea44eB4PsiPhiTHETA.dll")
dll_pathDD5THETA_PSI_PHI = os.path.join(os.path.dirname(__file__), "intdd5ea44eB5PsiPhiTHETA", "intdd5ea44eB5PsiPhiTHETA.dll")


my_libCCLLTHETA = ctypes.WinDLL(dll_pathLLTHETA)
my_libCCLLTHETA_PSI_PHI = ctypes.WinDLL(dll_pathLLTHETA_PSI_PHI)
my_libCCLTLTTHETA_PSI_PHI = ctypes.WinDLL(dll_pathLTLTTHETA_PSI_PHI)
my_libCCSPTHETA_PSI_PHI = ctypes.WinDLL(dll_pathSPTHETA_PSI_PHI)

my_libDD0THETA_PSI_PHI = ctypes.WinDLL(dll_pathDD0THETA_PSI_PHI)
my_libDD1THETA_PSI_PHI = ctypes.WinDLL(dll_pathDD1THETA_PSI_PHI)
my_libDD2THETA_PSI_PHI = ctypes.WinDLL(dll_pathDD2THETA_PSI_PHI)
my_libDD3THETA_PSI_PHI = ctypes.WinDLL(dll_pathDD3THETA_PSI_PHI)
my_libDD4THETA_PSI_PHI = ctypes.WinDLL(dll_pathDD4THETA_PSI_PHI)
my_libDD5THETA_PSI_PHI = ctypes.WinDLL(dll_pathDD5THETA_PSI_PHI)
# --- End DLL Loading ---

intccll5ea44eBTH = [[None for _ in range(41)] for _ in range(5)]
intccll5ea44eBTHPSIPHI = [[[[None for _ in range(num_phi+1)] for _ in range(num_psi+1)] for _ in range(81)] for _ in range(5)]
intccltlt5ea44eBTHPSIPHI = [[[[None for _ in range(num_phi+1)] for _ in range(num_psi+1)] for _ in range(81)] for _ in range(5)]
intccsp5ea44eBTHPSIPHI = [[[[None for _ in range(num_phi+1)] for _ in range(num_psi+1)] for _ in range(81)] for _ in range(5)]

#intcc5ea44eBTHPSIPHI = [[[[None for _ in range(81)] for _ in range(5)] for _ in range(5)] for _ in range(5)]
intdd5ea44eBTHPSIPHI = [[[[[None for _ in range(num_psi+1)] for _ in range(81)] for _ in range(5)] for _ in range(5)] for _ in range(5)]


try:
    for i in range(1,5):
        for j in range(0,81):
            for nu in range(1,5):
                for rho in range(1,5):
                    intdd5ea44eBTHPSIPHI[i][nu][rho][j][0]= getattr(my_libDD0THETA_PSI_PHI, f"intddll5ea44eB{0}Psi{5}Phi{i}{nu}{rho}THETA{j}")
                    intdd5ea44eBTHPSIPHI[i][nu][rho][j][1]= getattr(my_libDD1THETA_PSI_PHI, f"intddll5ea44eB{1}Psi{5}Phi{i}{nu}{rho}THETA{j}")
                    intdd5ea44eBTHPSIPHI[i][nu][rho][j][2]= getattr(my_libDD2THETA_PSI_PHI, f"intddll5ea44eB{2}Psi{5}Phi{i}{nu}{rho}THETA{j}")
                    intdd5ea44eBTHPSIPHI[i][nu][rho][j][3]= getattr(my_libDD3THETA_PSI_PHI, f"intddll5ea44eB{3}Psi{5}Phi{i}{nu}{rho}THETA{j}")
                    intdd5ea44eBTHPSIPHI[i][nu][rho][j][4]= getattr(my_libDD4THETA_PSI_PHI, f"intddll5ea44eB{4}Psi{5}Phi{i}{nu}{rho}THETA{j}")
                    intdd5ea44eBTHPSIPHI[i][nu][rho][j][5]= getattr(my_libDD5THETA_PSI_PHI, f"intddll5ea44eB{5}Psi{5}Phi{i}{nu}{rho}THETA{j}")
            for psi in range(0,num_psi+1):
                for phi in range(1,num_phi+1):
                    # --- LL ---
                    func_name_ll_psi_phi = f"intccll5ea44eB{psi}Psi{phi}Phi{i}THETA{j}"
                    intccll5ea44eBTHPSIPHI[i][j][psi][phi] = getattr(my_libCCLLTHETA_PSI_PHI, func_name_ll_psi_phi)
                    intccll5ea44eBTHPSIPHI[i][j][psi][phi].argtypes = [ctypes.c_double] * 2
                    intccll5ea44eBTHPSIPHI[i][j][psi][phi].restype = ComplexDouble

                    # --- LTLT ---
                    func_name_ltlt_psi_phi = f"intccltlt5ea44eB{psi}Psi{phi}Phi{i}THETA{j}"
                    intccltlt5ea44eBTHPSIPHI[i][j][psi][phi] = getattr(my_libCCLTLTTHETA_PSI_PHI, func_name_ltlt_psi_phi)
                    intccltlt5ea44eBTHPSIPHI[i][j][psi][phi].argtypes = [ctypes.c_double] * 2
                    intccltlt5ea44eBTHPSIPHI[i][j][psi][phi].restype = ComplexDouble

                    # --- SP  ---
                    func_name_sp_psi_phi = f"intccsp5ea44eB{psi}Psi{phi}Phi{i}THETA{j}"
                    intccsp5ea44eBTHPSIPHI[i][j][psi][phi] = getattr(my_libCCSPTHETA_PSI_PHI, func_name_sp_psi_phi)
                    intccsp5ea44eBTHPSIPHI[i][j][psi][phi].argtypes = [ctypes.c_double] * 2
                    intccsp5ea44eBTHPSIPHI[i][j][psi][phi].restype = ComplexDouble
                    for nu in range(1,5):
                        for rho in range(1,5):                            
                            intdd5ea44eBTHPSIPHI[i][nu][rho][j][psi].argtypes = [ctypes.c_double] * 2
                            intdd5ea44eBTHPSIPHI[i][nu][rho][j][psi].restype = ComplexDouble


except OSError as e:
    print(f"Error loading DLL {dll_pathLLTHETA_PSI_PHI}: {e}")
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

#%%
def process_one_theta(task_args):
    """Worker function for multiprocessing. Integrates LL case for one theta value."""
    i, current_theta_rad,psi_idx,phi_idx, ndim, num_iterations, neval_per_iter, func = task_args
    print(f"[Proc {os.getpid()}] Processing Theta index {i} ({degrees(current_theta_rad):.2f} deg)")

    # Results for this specific theta (k=0 index unused)
    local_means = [np.nan + 0j] * 5
    local_sdevs = [np.nan + 0j] * 5


    for k in range(1, 5): # Loop through k=1, 2, 3, 4
        result_real = None
        result_imag = None
        integration_success = False

        try:
            # Select the specific C function pointer for this k and theta index i
            match func:
                case "LL":
                    c_func = intccll5ea44eBTHPSIPHI[k][i][psi_idx][phi_idx]
                case "LTLT":
                    c_func = intccltlt5ea44eBTHPSIPHI[k][i][psi_idx][phi_idx]
                case "SP":
                    c_func = intccsp5ea44eBTHPSIPHI[k][i][psi_idx][phi_idx]

            if c_func is None:
                # print(f"  [Proc {os.getpid()}] Warning: C function LL not loaded for k={k}, i={i}. Skipping.")
                continue # Skip this k value

            # Create integrand wrappers
            integrand_real, integrand_imag = make_vegas_integrands_real_imag(c_func)

            # Create NEW Integrator instances for this k value IN THIS PROCESS
            integ_real = vegas.Integrator([[0.0001, 0.99]] * ndim)
            integ_imag = vegas.Integrator([[0.0001, 0.99]] * ndim)

            # --- Integrate Real Part ---
            result_real = integ_real(integrand_real, nitn=num_iterations, neval=neval_per_iter)

            # --- Integrate Imaginary Part ---
            result_imag = integ_imag(integrand_imag, nitn=num_iterations, neval=neval_per_iter)

            # Check if results are valid vegas objects (contain mean/sdev)
            if hasattr(result_real, 'mean') and hasattr(result_imag, 'mean'):
                 integration_success = True

        except Exception as e:
            print(f"  [Proc {os.getpid()}] ERROR during LL integration for theta_idx={i}, k={k}: {e}")
            # traceback.print_exc() # Avoid excessive printing in parallel run

        # Store results if successful
        if integration_success:
            local_means[k] = result_real.mean + 1j * result_imag.mean
            local_sdevs[k] = result_real.sdev + 1j * result_imag.sdev
    return (i, psi_idx, phi_idx, local_means, local_sdevs,result_real)

def process_one_thetaDD(task_args):
    """Worker function for multiprocessing. Integrates LL case for one theta value."""
    i, current_theta_rad,psi_idx,phi_idx, ndim, num_iterations, neval_per_iter, func = task_args
    print(f"[Proc {os.getpid()}] Processing Theta index {i} ({degrees(current_theta_rad):.2f} deg)")

    # Results for this specific theta (k=0 index unused)
    local_means = [[[np.nan + 0j for _ in range(5)] for _ in range(5)] for _ in range(5)]
    local_sdevs = [[[np.nan + 0j for _ in range(5)] for _ in range(5)] for _ in range(5)]


    for mu in range(1, 5): # Loop through k=1, 2, 3, 4
        for nu in range(1,5):
            for rho in range(1,5):
                result_real = None
                result_imag = None
                integration_success = False

                try:
                    # Select the specific C function pointer for this k and theta index i
                    c_func = intdd5ea44eBTHPSIPHI[mu][nu][rho][i][psi_idx]

                    if c_func is None:
                        # print(f"  [Proc {os.getpid()}] Warning: C function LL not loaded for k={k}, i={i}. Skipping.")
                        continue # Skip this k value

                    # Create integrand wrappers
                    integrand_real, integrand_imag = make_vegas_integrands_real_imag(c_func)

                    # Create NEW Integrator instances for this k value IN THIS PROCESS
                    integ_real = vegas.Integrator([[0.0001, 0.99]] * ndim)
                    integ_imag = vegas.Integrator([[0.0001, 0.99]] * ndim)

                    # --- Integrate Real Part ---
                    result_real = integ_real(integrand_real, nitn=num_iterations, neval=neval_per_iter)

                    # --- Integrate Imaginary Part ---
                    result_imag = integ_imag(integrand_imag, nitn=num_iterations, neval=neval_per_iter)

                    # Check if results are valid vegas objects (contain mean/sdev)
                    if hasattr(result_real, 'mean') and hasattr(result_imag, 'mean'):
                        integration_success = True

                except Exception as e:
                    print(f"  [Proc {os.getpid()}] ERROR during LL integration for theta_idx={i}, k={k}: {e}")
                    # traceback.print_exc() # Avoid excessive printing in parallel run

                # Store results if successful
                if integration_success:
                    local_means[mu][nu][rho] = result_real.mean + 1j * result_imag.mean
                    local_sdevs[mu][nu][rho] = result_real.sdev + 1j * result_imag.sdev
    return (i, psi_idx, phi_idx, local_means, local_sdevs,result_real)

#%%

# ---------------------------------------INTEGRATION-----------------------------------


if __name__ == '__main__':
    multiprocessing.freeze_support() # For Windows compatibility / frozen apps

    EPS=np.finfo(np.float64).eps
    ndim = 2 # Number of dimensions for Vegas integration (should match C func args)
    neval_per_iter = 1000
    num_iterations = 24 # User's value
     # User's value
    theta_values_rad = np.linspace(-1+EPS, 0-EPS, num_theta_points) * (pi / 2.0) 
    theta_values_rad_mirror = np.linspace(0+EPS, 1-EPS, num_theta_points) * (pi / 2.0 )


    # Initialize result storage (filled with NaN initially)
    results_mean_complex_LL = [[[[np.nan + 0j] * num_theta_points for _ in range(5)] for _ in range(6)] for _ in range(6)]
    results_sdev_complex_LL = [[[[np.nan + 0j] * num_theta_points for _ in range(5)] for _ in range(6)] for _ in range(6)]
    results_mean_complex_LTLT = [[[[np.nan + 0j] * num_theta_points for _ in range(5)] for _ in range(6)] for _ in range(6)]
    results_sdev_complex_LTLT =[[[[np.nan + 0j] * num_theta_points for _ in range(5)] for _ in range(6)] for _ in range(6)]
    results_mean_complex_SP = [[[[np.nan + 0j] * num_theta_points for _ in range(5)] for _ in range(6)] for _ in range(6)]
    results_sdev_complex_SP =[[[[np.nan + 0j] * num_theta_points for _ in range(5)] for _ in range(6)] for _ in range(6)]
    results_mean_complex_DD=[[[[[np.nan + 0j] * num_theta_points for _ in range(5)] for _ in range(5)] for _ in range(5)] for _ in range(6)]
    results_sdev_complex_DD=[[[[[np.nan + 0j] * num_theta_points for _ in range(5)] for _ in range(5)] for _ in range(5)] for _ in range(6)]

    # 1. Prepare Tasks for Multiprocessing
    tasksLL = [
        (i, current_theta_rad, current_psi_idx, current_phi_idx, ndim, num_iterations, neval_per_iter, "LL")
        for i, current_theta_rad in enumerate(theta_values_rad)
        for current_psi_idx in range(num_psi + 1) # Iterate psi from 0 to 5
        for current_phi_idx in range(3, 4) # Iterate phi from 1 to 5
    ]
    tasksLTLT = [
        (i, current_theta_rad, current_psi_idx, current_phi_idx, ndim, num_iterations, neval_per_iter, "LTLT")
        for i, current_theta_rad in enumerate(theta_values_rad)
        for current_psi_idx in range(num_psi + 1) # Iterate psi from 0 to 5
        for current_phi_idx in range(3,4) # Iterate phi from 1 to 5
    ]
    tasksSP = [
        (i, current_theta_rad, current_psi_idx, current_phi_idx, ndim, num_iterations, neval_per_iter, "SP")
        for i, current_theta_rad in enumerate(theta_values_rad)
        for current_psi_idx in range(num_psi + 1) # Iterate psi from 0 to 5
        for current_phi_idx in range(3,4) # Iterate phi from 1 to 5
    ]
    tasksDD=[
        (i, current_theta_rad, current_psi_idx, current_phi_idx, ndim, num_iterations, neval_per_iter, "DD")
        for i, current_theta_rad in enumerate(theta_values_rad)
        for current_psi_idx in range(num_psi + 1) # Iterate psi from 0 to 5
        for current_phi_idx in range(5, 6) # Iterate phi from 1 to 5
    ]
    print(f"Prepared {len(tasksSP)} tasks for parallel execution.")


    # 2. Determine number of processes
    n_cores = multiprocessing.cpu_count()
    print(f"Starting parallel Vegas integration using {n_cores} processes.")
    print(f"Vegas settings per task: nitn={num_iterations}, neval={neval_per_iter}")


    # 3. Run in Parallel (or sequentially if n_processes=1)
    start_time = time.time()
    all_pool_resultsLL = []
    all_pool_resultsLTLT = []
    all_pool_resultsSP = []
    all_pool_resultsDD = []
    with multiprocessing.Pool(processes=n_cores) as pool: # Use n_cores for actual parallel
        # Use map instead of starmap since worker takes a single tuple argument
        all_pool_resultsLL = pool.map(process_one_theta, tasksLL)
        all_pool_resultsLTLT = pool.map(process_one_theta, tasksLTLT)
        all_pool_resultsSP = pool.map(process_one_theta, tasksSP)
        all_pool_resultsDD = pool.map(process_one_thetaDD, tasksDD)
    end_time = time.time()
    print(f"\nParallel integration finished in {end_time - start_time:.2f} seconds.")



    # 4. Collect and Organize Results
    print("Organizing results...")
    for result_tuple in all_pool_resultsLL:
        # Place results into the correct index 'i'
        i,psi_i,phi_i, local_means, local_sdevs,resreal = result_tuple
        results_found_for_i = False
        for k in range(1, 5):
            results_mean_complex_LL[psi_i][phi_i][k][i] = local_means[k]
            results_sdev_complex_LL[psi_i][phi_i][k][i] = local_sdevs[k]
    for result_tuple in all_pool_resultsLTLT:
        # Place results into the correct index 'i'
        i,psi_i,phi_i, local_means, local_sdevs,resreal = result_tuple
        results_found_for_i = False
        for k in range(1, 5):
            results_mean_complex_LTLT[psi_i][phi_i][k][i] = local_means[k]
            results_sdev_complex_LTLT[psi_i][phi_i][k][i] = local_sdevs[k]
    for result_tuple in all_pool_resultsSP:
        # Place results into the correct index 'i'
        i,psi_i,phi_i, local_means, local_sdevs,resreal = result_tuple
        results_found_for_i = False
        for k in range(1, 5):
            results_mean_complex_SP[psi_i][phi_i][k][i] = local_means[k]
            results_sdev_complex_SP[psi_i][phi_i][k][i] = local_sdevs[k]

    for result_tuple in all_pool_resultsDD:
        # Place results into the correct index 'i'
        i,psi_i,phi_i, local_means, local_sdevs,resreal = result_tuple
        results_found_for_i = False
        for mu in range(1, 5):
            for nu in range(1, 5):
                for rho in range(1, 5):
                    results_mean_complex_DD[psi_i][mu][nu][rho][i] = local_means[mu][nu][rho]
                    results_sdev_complex_DD[psi_i][mu][nu][rho][i] = local_sdevs[mu][nu][rho]
    

    
    #%%
    # --- Plotting (using the collected results) ---
    print("Generating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 7)) # Create a single figure and axes
    fig.suptitle("Sum of |Integral * KVec|^2 vs. Theta for Phi=5", fontsize=14) # Adjusted title

    #plot_theta_values = np.concatenate((theta_values_rad, theta_values_rad_mirror[1:]))
    colors = plt.cm.viridis(np.linspace(0, 1, num_psi + 1)) # Get distinct colors

    fixed_phi_idx = 5 # User specified phi=5

    # Loop through each psi index
    for psi_idx in range(num_psi + 1): # psi = 0, 1, 2, 3, 4, 5
        abs_result_psi = [np.nan] * num_theta_points # Initialize results for this psi

        # Loop through each theta index 'l'
        for l in range(num_theta_points):
            sum_sq_for_l_psi = 0.0
            sum_sq_for_l_psi_DD = 0.0
            valid_data_for_theta = True # Flag to check if data exists

            # Loop through each result component 'k'
            for k in range(1, 5): # k = 1, 2, 3, 4
                try:
                    # Get the complex mean result for this theta(l), component(k), psi_idx, and fixed phi_idx
                    
                    mean_valLL = results_mean_complex_LL[psi_idx][fixed_phi_idx][k][l]
                    mean_valLTLT = results_mean_complex_LTLT[psi_idx][fixed_phi_idx][k][l]
                    mean_valSP = results_mean_complex_SP[psi_idx][fixed_phi_idx][k][l]
                    
                    
                    # Check if any result is NaN (integration might have failed)
                    '''
                    if np.isnan(mean_valLL) or np.isnan(mean_valLTLT) or np.isnan(mean_valSP):
                        valid_data_for_theta = False
                        break # Stop calculating for this theta if any part is NaN
                    '''
                    # Calculate K-vector components (using k-1 for 0-based index)
                    # Ensure Ea=5 is used consistently as in the K-vector functions
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

                    for nu in range(1, 5):
                        for rho in range(1, 5):
                            mean_valDD = results_mean_complex_DD[psi_idx][k][nu][rho][l]
                            if np.isnan(mean_valDD):
                                valid_data_for_theta = False
                                break
                            if 0 <= k-1 < len(lp1_vec):
                                if 0 <= nu-1 < len(lp1_vec):
                                    if 0 <= rho-1 < len(lp1_vec):
                                        term_qk_LL_DD = np.abs(mean_valDD * qk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1]+(mean_valLL * qk_vec[k-1]))**2
                                        term_lk_LL_DD = np.abs(mean_valDD * lk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1]+(mean_valLL * lk_vec[k-1]))**2
                                        term_ltk_LL_DD = np.abs(mean_valDD * ltk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1]+(mean_valLL * ltk_vec[k-1]))**2
                                        term_gk_LL_DD = np.abs(mean_valDD * gk_vec[k-1]*lp1_vec[nu-1]*lp2_vec[rho-1]+(mean_valLL * gk_vec[k-1]))**2

                                        term_qk_LTLT_DD = np.abs(mean_valDD * qk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1]+(mean_valLTLT * qk_vec[k-1]))**2
                                        term_lk_LTLT_DD = np.abs(mean_valDD * lk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1]+(mean_valLTLT * lk_vec[k-1]))**2
                                        term_ltk_LTLT_DD = np.abs(mean_valDD * ltk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1]+(mean_valLTLT * ltk_vec[k-1]))**2
                                        term_gk_LTLT_DD = np.abs(mean_valDD * gk_vec[k-1]*ltp1_vec[nu-1]*ltp2_vec[rho-1]+(mean_valLTLT * gk_vec[k-1]))**2            

                                        term_qk_SP_DD = np.abs(mean_valDD * qk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2))+(mean_valSP * qk_vec[k-1]))**2
                                        term_lk_SP_DD = np.abs(mean_valDD * lk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2))+(mean_valSP * lk_vec[k-1]))**2
                                        term_ltk_SP_DD = np.abs(mean_valDD * ltk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2))+(mean_valSP * ltk_vec[k-1]))**2
                                        term_gk_SP_DD = np.abs(mean_valDD * gk_vec[k-1]*((lp1_vec[nu-1]*ltp2_vec[rho-1]+ltp1_vec[nu-1]*lp2_vec[rho-1])/sqrt(2))+(mean_valSP * gk_vec[k-1]))**2

                                        sum_sq_for_l_psi_DD += (term_qk_LL_DD+ term_lk_LL_DD+term_ltk_LL_DD+term_gk_LL_DD + term_qk_LTLT_DD+ term_lk_LTLT_DD+term_ltk_LTLT_DD+term_gk_LTLT_DD+term_qk_SP_DD+ term_lk_SP_DD+term_ltk_SP_DD+term_gk_SP_DD)
                                        
                    # Calculate terms (ensure k-1 index is valid for vectors)
                    '''
                    if 0 <= k-1 < len(qk_vec):
                         term_qk_LL = np.abs(mean_valLL * qk_vec[k-1])**2
                         term_lk_LL = np.abs(mean_valLL * lk_vec[k-1])**2
                         term_ltk_LL = np.abs(mean_valLL * ltk_vec[k-1])**2
                         term_gk_LL = np.abs(mean_valLL * gk_vec[k-1])**2

                         term_qk_LTLT = np.abs(mean_valLTLT * qk_vec[k-1])**2
                         term_lk_LTLT = np.abs(mean_valLTLT * lk_vec[k-1])**2
                         term_ltk_LTLT = np.abs(mean_valLTLT * ltk_vec[k-1])**2
                         term_gk_LTLT = np.abs(mean_valLTLT * gk_vec[k-1])**2

                         term_qk_SP = np.abs(mean_valSP * qk_vec[k-1])**2
                         term_lk_SP = np.abs(mean_valSP * lk_vec[k-1])**2
                         term_ltk_SP = np.abs(mean_valSP * ltk_vec[k-1])**2
                         term_gk_SP = np.abs(mean_valSP * gk_vec[k-1])**2

                         # Add to the sum for this theta index 'l' and psi_idx
                         sum_sq_for_l_psi += (term_qk_LL + term_lk_LL + term_ltk_LL + term_gk_LL +
                                              term_qk_LTLT + term_lk_LTLT + term_ltk_LTLT + term_gk_LTLT +
                                              term_qk_SP + term_lk_SP + term_ltk_SP + term_gk_SP)
                    else:
                         print(f"Warning: Index k-1={k-1} out of bounds for K-vectors (length {len(qk_vec)}). Skipping term.")
                         valid_data_for_theta = False
                         break
                    '''

                except IndexError:
                    # This might happen if results arrays weren't fully populated for psi/phi
                    print(f"Warning: Data not found for psi={psi_idx}, phi={fixed_phi_idx}, k={k}, theta_idx={l}. Skipping theta point.")
                    valid_data_for_theta = False
                    break
                except Exception as e:
                    print(f"Error during calculation for psi={psi_idx}, phi={fixed_phi_idx}, k={k}, theta_idx={l}: {e}")
                    valid_data_for_theta = False
                    break

            # Store the sum for this theta point if data was valid
            if valid_data_for_theta:
                abs_result_psi[l] = sum_sq_for_l_psi_DD
                #abs_result_psi[l] = sum_sq_for_l_psi
            # else it remains NaN
        
        # Prepare data for plotting (concatenate with mirrored part)
        # Filter out NaN values before flipping and concatenating to avoid issues
        valid_indices = ~np.isnan(abs_result_psi)
        valid_abs_result_psi = np.array(abs_result_psi)[valid_indices]
        valid_theta_values_rad = theta_values_rad[valid_indices]
        
        if len(valid_abs_result_psi) > 0:
            # Mirror only the valid results
            plot_abs_results_psi =  np.concatenate((valid_abs_result_psi[20:], np.flip(valid_abs_result_psi)[1:][:-20]))
            # Mirror the corresponding theta values
            plot_theta_values_psi = np.concatenate((valid_theta_values_rad[20:], -np.flip(valid_theta_values_rad)[1:][:-20])) # Use negative for mirror

            # Plot the results for this psi_idx
            ax.plot( # Use plot for lines instead of scatter
                plot_theta_values_psi,
                plot_abs_results_psi,
                label=f"Psi = {psi_idx}",
                color=colors[psi_idx],
                marker='o', markersize=3, linestyle='-' # Add markers and lines
            )
        else:
            print(f"No valid data to plot for Psi = {psi_idx}")

    ax.set_xlabel("Theta (radians)")
    ax.set_ylabel("Sum |Integral * KVec|^2") # Updated label
    ax.set_title(f"Combined LL+LTLT+SP Results (Phi={fixed_phi_idx})")
    ax.legend()
    ax.grid(True)
    #ax.set_yscale('log') # Use log scale if values vary greatly

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show()

    print("\nScript finished.")
# %%
