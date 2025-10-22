# blockSQP_reference_build - build system and Python interface for blockSQP
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
# Licensed under the zlib license. See LICENSE for more details.

import os
import sys
import datetime

try:
    cD = os.path.dirname(os.path.abspath(__file__))
except:
    cD = os.getcwd()
sys.path += [cD + "/..", cD + "/../.."]

import py_blockSQP_old
import OCP_experiment_old
import localcopy_OCProblems as OCProblems

Examples = [
            OCProblems.Batch_Reactor,
            OCProblems.Cart_Pendulum,
            OCProblems.Catalyst_Mixing,
            OCProblems.Cushioned_Oscillation,
            OCProblems.Egerstedt_Standard,
            OCProblems.Electric_Car,
            OCProblems.Goddard_Rocket,
            OCProblems.Hang_Glider,
            OCProblems.Hanging_Chain,
            OCProblems.Lotka_Volterra_Fishing,
            OCProblems.Particle_Steering,
            OCProblems.Quadrotor_Helicopter,
            OCProblems.Three_Tank_Multimode,
            OCProblems.Time_Optimal_Car,
            OCProblems.Tubular_Reactor,
            OCProblems.Lotka_OED,
            ]
OCProblems.Goddard_Rocket.__name__ = 'Goddard\'s Rocket'

opt_SR1_BFGS = py_blockSQP_old.SQPoptions()
opt_SR1_BFGS.maxTimeQP = 10.0

opt_conv_str_0 = py_blockSQP_old.SQPoptions()
opt_conv_str_0.maxTimeQP = 10.0
opt_conv_str_0.maxConvQP = 4


Experiments = [
               (opt_SR1_BFGS, "SR1-BFGS"),
               (opt_conv_str_0, "conv. str. 0"),
               ]


file_output = True
plot_folder = cD + "/out_old_blockSQP_experiments_RK4"
integrator = 'RK4'


nPert0 = 0
nPertF = 40
dirPath = plot_folder
if not os.path.exists(dirPath):
    os.makedirs(dirPath)
if file_output:
    date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
    sep = "" if dirPath[-1] == "/" else "/"
    pref = "blockSQP"
    filePath = dirPath + sep + pref + "_it_" + date_app + ".txt"
    out = open(filePath, 'w')
else:
    out = OCP_experiment_old.out_dummy()
titles = [EXP_name for _, EXP_name in Experiments]
OCP_experiment_old.print_heading(out, titles)

for OCclass in Examples:
    # if OCclass in (OCProblems.Hang_Glider, OCProblems.Batch_Reactor, OCProblems.Cart_Pendulum) and integrator == 'collocation':
    #     continue
    
    OCprob = OCclass(nt = 100, integrator = integrator, parallel = True)
    itMax = 200
    titles = []
    EXP_N_SQP = []
    EXP_N_secs = []
    EXP_type_sol = []
    n_EXP = 0
    for EXP_opts, EXP_name in Experiments:
        ret_N_SQP, ret_N_secs, ret_type_sol = OCP_experiment_old.perturbed_starts(OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
        EXP_N_SQP.append(ret_N_SQP)
        EXP_N_secs.append(ret_N_secs)
        EXP_type_sol.append(ret_type_sol)
        titles.append(EXP_name)
        n_EXP += 1
    ###############################################################################
    OCP_experiment_old.plot_successful(n_EXP, nPert0, nPertF,\
        titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
        suptitle = OCclass.__name__, dirPath = dirPath, savePrefix = "blockSQP")
    OCP_experiment_old.print_iterations(out, OCclass.__name__, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
out.close()
