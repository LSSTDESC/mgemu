# import regular cosmosis packages
from builtins import str
import os
from cosmosis.datablock import names, option_section
import sys
import traceback
# import packages for g(R) emulator
from mgemu import emu
import pyccl as ccl
import numpy as np

# These are pre-defined strings we use as datablock
# section names
modified_gravity_parameters = names.modified_gravity_parameters
cosmo = names.cosmological_parameters
distances = names.distances
matter_power_nl = names.matter_power_nl

# Read options from the ini file which are called once per processor
def setup(options):
    config = {
        'zmin': options.get_double(option_section, 'zmin', default=0.0),
        'zmax': options.get_double(option_section, 'zmax', default=1.0),
        'num_z': options.get_int(option_section, 'num_z', default=150),
        'do_distances': options.get_bool(option_section, 'do_distances', default = True),
        'transfer_function': options.get_string(option_section, 'transfer_function', default='boltzmann_class'),
    }
    return config

# Execute module is called every time for each set of cosmological parameters
def execute(block, config):

    # Get array of redshift to calculate P_MG(k)
    z = np.linspace(config['zmin'], config["zmax"], config['num_z'])

    # Call mgemu and ccl
    pk_mg, k, chi = get_observable(config,
                   Om  = block[cosmo, "omega_m"],
                   h   = block[cosmo, "h0"],
                   s8  = block[cosmo, "sigma_8"],
                   ns  = block[cosmo, "n_s"],
                   fR0 = block[modified_gravity_parameters, "fR0"],
                   n   = block[modified_gravity_parameters, "n"],
                   z   = z,
                   Omb  = block[cosmo, "omega_b"]
    )

    # Saving results in a grid of (k,z,pk). Here k and pk already in h/Mpc and (Mpc/h)^3 units
    block.put_grid("matter_power_nl", "k_h", k, "z", z, "p_k", pk_mg.T)

    # Pass chi to block[distances, 'd_m'], to use later in /structure/project_2d.py file.
    block[distances, 'z'] = z
    block[distances, 'a'] = 1./(1+z)
    block[distances, 'd_m'] = chi

    return 0

# Function to get pk_mg (not the ratio) and k arrays
def get_observable(config, Om, h, s8, ns, fR0, n, z, Omb=0.02203):

    # Call MG emulator
    Omh2 = (h**2)*Om
    pkratio = []
    for i in range(0,len(z)):
        pkratio_tmp, k_tmp = emu(Omh2, ns, s8, fR0, n, z[i])
        pkratio.append(pkratio_tmp)
    k=k_tmp # each array of k is the same (regardless of value of z)
    # Convert list to numpy arrays
    pkratio = np.array(pkratio)

    # Define CCL cosmology object
    cosmo = ccl.Cosmology(Omega_c = Om - Omb,
                            Omega_b = Omb,
                            h = h,
                            sigma8 = s8,
                            n_s = ns,
                            Neff = 3.04,
                            transfer_function = config['transfer_function'],
                            matter_power_spectrum = 'emu')
    # Get P_MG(k)
    a = 1./(1+z)
    #Now we have to be careful, because mgemu k-units are in h/Mpc, while CCL units are in 1/Mpc. convert
    kccl = k*h
    pk_nl = []
    for i in range(0,len(a)):
        pk_nl.append(ccl.nonlin_matter_power(cosmo, kccl, a[i]))
    pk_nl = np.array(pk_nl)
    #CCL output pk is in Mpc^3, convert to (Mpc/h)^3
    pk_nl *= h*h*h
    pk_mg = pk_nl*pkratio
    # Get radial comoving distance
    if config['do_distances']==True:
        chi = ccl.comoving_radial_distance(cosmo, 1/(1+z))

    return pk_mg, k, chi

# Not needed in Python (purpose is to free memory)
def cleanup(config):
    pass
