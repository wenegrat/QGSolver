#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:24:38 2017

@author: Jacob Wenegrat (wenegrat@umd.edu)
wenegrat.github.io
"""

"""
Dedalus script for 2 layer QG simulation.

Note that this has not been tested fully, so use at your own risk!

"""

import numpy as np
from mpi4py import MPI
CW = MPI.COMM_WORLD
import time
from pylab import *
from dedalus import public as de
from dedalus.extras import flow_tools
import scipy.integrate as integrate
import logging
logger = logging.getLogger(__name__)


# Parameters
#directoryname = '/scratch/jacob13/NLSIM/'
#ly_global = np.logspace(-5, -3, 192)*2*np.pi
#OD = False



nx = 256
ny = 256


f = 1e-4 # Coriolis parameter
Bo = 0*1.2e-11 # Beta Parameter
A4 = 1e-11
#A8 = 1e23
H1 = 500
H2 = 500
#rho1 = 1025
#rho2 = 1025.5

#Magnitude of shear flow
U1m = 0.025
U2m = 0.00

# Deformation Radius
Ld = 30e3


#Set the strength of the bottom drag (see Thompson and Young 2006)
rLU = 0.2

rek = U1m*rLU/Ld

gridratio = 1.7 # gridratio = nx/(2*pi*Lx/Ld)See Thompson and Young  2006

Lx, Ly = (nx/(gridratio/Ld),ny/(gridratio/Ld))

# Visc ratio, see Thompson and Young 2006
vr = 1e-13
L = Lx/(2*np.pi)
A8 = vr*(U1m*L**7)
#%% 3D PROBLEM
# Create basis and domain
start_init_time = time.time()
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)

domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, mesh=None)
y = domain.grid(1)
x = domain.grid(0)



# Define Fields (these can later define spatially varying velocities)
U1 = domain.new_field(name='U1')
U2 = domain.new_field(name='U2')

U1['g'] = U1m
U2['g'] = U2m



# set up IVP
problem = de.IVP(domain, variables=['q1', 'q2', 'psi1', 'psi2', 'u1', 'v1'])
#problem.meta[:]['z']['dirichlet'] = True

slices = domain.dist.grid_layout.slices(scales=1)

problem.parameters['delta'] = H1/H2
problem.parameters['DeltaU'] = U1m - U2m
problem.parameters['Bo'] = Bo
problem.parameters['Ld'] = Ld
problem.parameters['U1'] = U1
problem.parameters['U2'] = U2
problem.parameters['A4'] = A4
problem.parameters['A8'] = A8
problem.parameters['rek'] = rek
#Following notation in Arbic and Flierl
# define substitutions
problem.substitutions['L4(A)'] = '(dx(dx(dx(dx(A)))) + 2*dx(dx(dy(dy(A)))) + dy(dy(dy(dy(A)))))' #Horizontal biharmonic diff
problem.substitutions['HV(A)'] = '-A8*(L4(dx(dx(dx(dx(A))))) + L4(2*dx(dx(dy(dy(A))))) + L4(dy(dy(dy(dy(A))))))'
problem.substitutions['Jac(A,B)'] = 'dx(A)*dy(B) - dy(A)*dx(B) ' # Jacobian
problem.substitutions['Lap(A)'] = 'dx(dx(A)) + dy(dy(A))'
problem.substitutions['F1'] = '1/((delta+1)*Ld**2)'
problem.substitutions['F2'] = 'delta/((delta+1)*Ld**2)'
problem.substitutions['Qy1'] = 'Bo + F1*DeltaU'
problem.substitutions['Qy2'] = 'Bo - F2*DeltaU'

# define equations
problem.add_equation('dt(q1)  + dx(psi1)*Qy1  - HV(q1)  = - Jac(psi1, q1) - U1*dx(q1)', condition='(nx!=0) or (ny !=0)')
problem.add_equation('dt(q2)  + dx(psi2)*Qy2  - HV(q2)  = - Jac(psi2, q2) - U2*dx(q2) - rek*Lap(psi2) ', condition='(nx!=0) or (ny != 0)')

problem.add_equation('psi1=0', condition='(nx==0) and (ny==0)')
problem.add_equation('psi2=0', condition='(nx==0) and (ny==0)')

problem.add_equation('-q1 + Lap(psi1) +F1*(psi2-psi1)=0')
problem.add_equation('-q2 + Lap(psi2) +F2*(psi1-psi2)=0' )

problem.add_equation('u1 + dy(psi1) = 0') #Necessary only for the CFL condition
problem.add_equation('v1 - dx(psi1) = 0')

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')



#%%
# define initial condtions
q1 = solver.state['q1']
q2 = solver.state['q2']
psi1 = solver.state['psi1']
psi2 = solver.state['psi2']
# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

q1['g'] = 0
q1['g'] +=1e-7*noise
rand = np.random.RandomState(seed=24)
noise = rand.standard_normal(gshape)[slices]
q2['g'] = 1e-7*noise
psi1['g']=0
psi2['g']=0

#%%
# Integration parameters
#solver.stop_sim_time = 3600*24*50
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf
Ti = Ld/abs(U1m) #Eddy turnover timescale
dt = Ti/200 #First guess at time step

#solver.stop_sim_time = 500*86400
solver.stop_sim_time = 300*Ti

# Analysis
snap = solver.evaluator.add_file_handler('snapshots', sim_dt=3600*24*5, max_writes=24*1000, parallel=False)

# Basic Diagnostics
snap.add_task('q1', name='q1')
snap.add_task('q2', name='q2')
snap.add_task('Lap(psi1)', name='relv1')
snap.add_task('Lap(psi2)', name='relv2')
snap.add_task('psi1', name = 'psi1')
snap.add_task('psi2', name = 'psi2')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
## CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=2,
                     max_change=1.5, min_change=0, max_dt=Ti/2)
CFL.add_velocities(('u1', 'v1'))



# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)

        if (solver.iteration-1) % 500 == 1:
#        if True:
            logger.info('Iteration: %i, Days: %1.1f, dt: %e' %(solver.iteration, solver.sim_time/86400, dt))
            qtemp = solver.state['q1']
            if qtemp['g'].size > 0:
                qm = np.max(np.abs(qtemp['g']))
                logger.info('q1 Val: %f' % qm)
                if np.isnan(qm):
                    raise Exception('NaN encountered.')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))
