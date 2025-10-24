# blockSQP_reference_build - build system and Python interface for blockSQP
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
# Licensed under the zlib license. See LICENSE for more details.


import localcopy_OCProblems as OCProblems
import py_blockSQP_old
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True


def create_prob_old(OCprob : OCProblems.OCProblem):    
    prob = py_blockSQP_old.Problemspec()
    prob.x_start = OCprob.start_point
    
    prob.nVar = OCprob.nVar
    prob.nCon = OCprob.nCon
    prob.f = OCprob.f
    prob.grad_f = OCprob.grad_f
    prob.g = OCprob.g
    prob.make_sparse(OCprob.jac_g_nnz, OCprob.jac_g_row, OCprob.jac_g_colind)
    prob.jac_g_nz = OCprob.jac_g_nz
    prob.hess = OCprob.hess_lag
    prob.set_blockIndex(np.array(OCprob.hessBlock_index, dtype = np.int32))
    prob.set_bounds(OCprob.lb_var, OCprob.ub_var, OCprob.lb_con, OCprob.ub_con)
    prob.lam_start = np.zeros(prob.nVar + prob.nCon, dtype = np.float64).reshape(-1)
    
    return prob

def perturbed_starts(OCprob : OCProblems.OCProblem, opts : py_blockSQP_old.SQPoptions, nPert0, nPertF, COND = False, itMax = 100):
    N_SQP = []
    N_secs = []
    type_sol = []
    for j in range(nPert0,nPertF):
        start_it = OCprob.perturbed_start_point(j)
        
        prob = create_prob_old(OCprob)
        prob.x_start = start_it
        prob.complete()
        stats = py_blockSQP_old.SQPstats("./solver_outputs")        
        t0 = time.monotonic()
        optimizer = py_blockSQP_old.SQPmethod(prob, opts, stats)
        optimizer.init()
        ret = optimizer.run(itMax)
        optimizer.finish()
        t1 = time.monotonic()
        
        N_SQP.append(stats.itCount)
        N_secs.append(t1 - t0)
        if int(ret) >= 0:
            type_sol.append(int(ret))
        else:
            type_sol.append(-1)    
    return N_SQP, N_secs, type_sol


def plot_successful(n_EXP, nPert0, nPertF, titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol, suptitle = None, dirPath = None, savePrefix = None):
    n_xticks = 10
    tdist = round((nPertF - nPert0)/n_xticks)
    tdist += (tdist==0)
    xticks = np.arange(nPert0, nPertF + tdist, tdist)
    ###############################################################################
    def F(x,r):
        if r == 0:
            return x
        else:
            return 0.00001    
    EXP_N_SQP_S = [[F(EXP_N_SQP[i][j], EXP_type_sol[i][j]) for j in range(nPertF - nPert0)] for i in range(n_EXP)]
    EXP_N_secs_S = [[F(EXP_N_secs[i][j], EXP_type_sol[i][j]) for j in range(nPertF - nPert0)] for i in range(n_EXP)]

    EXP_N_SQP_mu = [sum(EXP_N_SQP[i])/len(EXP_N_SQP[i]) for i in range(n_EXP)]
    EXP_N_SQP_sigma = [(sum((np.array(EXP_N_SQP[i]) - EXP_N_SQP_mu[i])**2)/len(EXP_N_SQP[i]))**(0.5) for i in range(n_EXP)]
    EXP_N_secs_mu = [sum(EXP_N_secs[i])/len(EXP_N_secs[i]) for i in range(n_EXP)]
    EXP_N_secs_sigma = [(sum((np.array(EXP_N_secs[i]) - EXP_N_secs_mu[i])**2)/len(EXP_N_secs[i]))**(0.5) for i in range(n_EXP)]
    
    # trunc_float = lambda num, dg: str(float(num))[0:int(np.ceil(abs(np.log(num + (num == 0))/np.log(10)))) + 2 + dg]

    ###############################################################################
    titlesize = 23
    axtitlesize = 20
    labelsize = 19
    
    fig = plt.figure(constrained_layout=True, dpi = 300, figsize = (14+2*(max(n_EXP - 2, 0)), 3.5 + 3.5*(n_EXP - 1)))
    if isinstance(suptitle, str):
        fig.suptitle(r"$\textbf{" + suptitle + "}$", fontsize = 24, fontweight = 'bold')
    subfigs = fig.subfigures(nrows=n_EXP, ncols=1)
    
    if n_EXP == 1:
        subfigs = (subfigs,)
    for i in range(n_EXP):
        ax_it, ax_time = subfigs[i].subplots(nrows=1,ncols=2)
        
        ax_it.scatter(list(range(nPert0,nPertF)), EXP_N_SQP_S[i])#, c = cmap[i])
        ax_it.set_ylabel('SQP iterations', size = labelsize)
        ax_it.set_ylim(bottom = 0)
        ax_it.set_xlabel('location of perturbation', size = labelsize)
        # ax_it.set_title(r"$\mu = " + trunc_float(EXP_N_SQP_mu[i], 1) + r"\ \sigma = " + trunc_float(EXP_N_SQP_sigma[i], 1) + "$", size = axtitlesize)
        ax_it.set_title(r"$\mu = " + f"{EXP_N_SQP_mu[i]:.2f}" + r"\ \sigma = " + f"{EXP_N_SQP_sigma[i]:.2f}" + "$", size = axtitlesize)
        ax_it.set_xticks(xticks)
        ax_it.tick_params(labelsize = labelsize - 1)
        
        ax_time.scatter(list(range(nPert0,nPertF)), EXP_N_secs_S[i])#, c = cmap[i])
        ax_time.set_ylabel("solution time [s]", size = labelsize)
        ax_time.set_ylim(bottom = 0)
        ax_time.set_xlabel("location of perturbation", size = labelsize)
        # ax_time.set_title(r"$\mu = " + trunc_float(EXP_N_secs_mu[i], 1) + r"\ \sigma = " + trunc_float(EXP_N_secs_sigma[i], 1) + "$", size = axtitlesize)
        ax_time.set_title(r"$\mu = " + f"{EXP_N_secs_mu[i]:.2f}" + r"\ \sigma = " + f"{EXP_N_secs_sigma[i]:.2f}" + "$", size = axtitlesize)
        
        ax_time.set_xticks(xticks)
        ax_time.tick_params(labelsize = labelsize - 1)
        
        subfigs[i].suptitle(titles[i], size = titlesize)
    if not isinstance(dirPath, str):
        plt.show()
    else:
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
        name_app = "" if suptitle is None else suptitle.replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")        
        sep = "" if dirPath[-1] == "/" else "/"
        pref = "" if savePrefix is None else savePrefix
        
        plt.savefig(dirPath + sep + pref + "_it_s_" + name_app + "_" + date_app)


def print_heading(out, EXP_names : list[str]):
    out.write(" "*27)
    for EXP_name in EXP_names:
        out.write(EXP_name[0:40].ljust(21 + 5 + 21))
    out.write("\n" + " "*27)
    for i in range(len(EXP_names)):
        out.write("mu_N".ljust(10) + "sigma_N".ljust(11) + "mu_t".ljust(10) + "sigma_t".ljust(11))
        if i < len(EXP_names) - 1:
            out.write("|".ljust(5))
    out.write("\n")
    
def print_iterations(out, name, EXP_N_SQP, EXP_N_secs, EXP_type_sol):
    n_EXP = len(EXP_N_SQP)
    EXP_N_SQP_mu = [sum(EXP_N_SQP[i])/len(EXP_N_SQP[i]) for i in range(n_EXP)]
    EXP_N_SQP_sigma = [(sum((np.array(EXP_N_SQP[i]) - EXP_N_SQP_mu[i])**2)/len(EXP_N_SQP[i]))**(0.5) for i in range(n_EXP)]
    EXP_N_secs_mu = [sum(EXP_N_secs[i])/len(EXP_N_secs[i]) for i in range(n_EXP)]
    EXP_N_secs_sigma = [(sum((np.array(EXP_N_secs[i]) - EXP_N_secs_mu[i])**2)/len(EXP_N_secs[i]))**(0.5) for i in range(n_EXP)]
    
    trunc_float = lambda num, dg: str(float(num))[0:int(np.ceil(abs(np.log(num + (num == 0))/np.log(10)))) + 2 + dg]
    out.write(name[:25].ljust(27))
    for i in range(n_EXP):
        out.write((trunc_float(EXP_N_SQP_mu[i],1) + ",").ljust(10) + (trunc_float(EXP_N_SQP_sigma[i],1) + ";").ljust(11) + (trunc_float(EXP_N_secs_mu[i],1) + "s,").ljust(10) + (trunc_float(EXP_N_secs_sigma[i],1) + "s").ljust(11))
        if i < n_EXP - 1:
            out.write("|".ljust(5))
    out.write("\n")
    

class out_dummy:
    def __init__(self):
        pass
    def write(self, Str : str):
        pass
    def close(self):
        pass

def run_blockSQP_experiments(Examples : list[type], Experiments : list[tuple[py_blockSQP_old.SQPoptions, str]], dirPath : str, nPert0 = 0, nPertF = 40, print_output = True, **kwargs):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    if print_output:
        date_app = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("'", "")
        sep = "" if dirPath[-1] == "/" else "/"
        pref = "blockSQP"
        filePath = dirPath + sep + pref + "_it_" + date_app + ".txt"
        out = open(filePath, 'w')
    else:
        out = out_dummy()
    titles = [EXP_name for _, EXP_name in Experiments]
    print_heading(out, titles)
    
    for OCclass in Examples:        
        OCprob = OCclass(**kwargs)
        itMax = 200
        titles = []
        EXP_N_SQP = []
        EXP_N_secs = []
        EXP_type_sol = []
        n_EXP = 0
        for EXP_opts, EXP_name in Experiments:
            ret_N_SQP, ret_N_secs, ret_type_sol = perturbed_starts(OCprob, EXP_opts, nPert0, nPertF, itMax = itMax)
            EXP_N_SQP.append(ret_N_SQP)
            EXP_N_secs.append(ret_N_secs)
            EXP_type_sol.append(ret_type_sol)
            titles.append(EXP_name)
            n_EXP += 1
        ###############################################################################
        plot_successful(n_EXP, nPert0, nPertF,\
            titles, EXP_N_SQP, EXP_N_secs, EXP_type_sol,\
            suptitle = OCclass.__name__, dirPath = dirPath, savePrefix = "blockSQP")
        print_iterations(out, OCclass.__name__, EXP_N_SQP, EXP_N_secs, EXP_type_sol)
    out.close()


