import numpy as np
import multiprocessing as mp
import time
import Num_SH_fast as NSH
import Emma3
import os
import sys

diagnostic = False

Num_args = len(sys.argv) - 1

if Num_args >= 1:
    if sys.argv[1][-1] == '/':
        folder = sys.argv[:-1]
    else:
        folder = sys.argv[1]
else:
    folder = "ThreeEqual"

if Num_args >= 2:
    file_header = sys.argv[2]
else:
    file_header = "three_equal"
    
mix_factor = np.ones(3)
lep_factor = np.ones(3)
run_all = False
spectra_only = False

error = False

if Num_args >= 3:
    args = sys.argv[3:]

    for arg in args:
        equal_char = arg.find("=")
        if equal_char == -1:
            error = True
            print("ERROR: usage m=#,#,# OR l=#,#,# OR r=T/F or s=T/F or d=T/F(no spaces)")
            break

        if arg[0] == 'm':
            split = arg[equal_char+1:].split(',')
            if len(split) != 3:
                error = True
                print("ERROR: usage m=#,#,# (needs three mixing angle factors)")
                break
            else:
                for i in range(3):
                    mix_factor[i] = float(split[i])

        if arg[0] == 'l':
            split = arg[equal_char+1:].split(',')
            if len(split) != 3:
                error = True
                print("ERROR: usage l=#,#,# (needs three mixing angle factors)")
                break
            else:
                for i in range(3):
                    lep_factor[i] = float(split[i])

        if arg[0] == 'r':
            if arg[equal_char+1] == 't' or arg[equal_char+1] == 'T':
                run_all = True
                
        if arg[0] == 's':
            if arg[equal_char+1] == 't' or arg[equal_char+1] == 'T':
                spectra_only = True
                
        if arg[0] == 'd':
            if arg[equal_char+1] == 't' or arg[equal_char+1] == 'T':
                diagnostic = True

if os.path.exists("{}/{}-results.npz".format(folder, file_header)):
    print("Summary file {}/{}-results.npz already exists. Do not overwrite. Abort.".format(folder, file_header))
    exit()
elif run_all and os.path.isdir("{}".format(folder)):
    print("Folder {} already exists. Do not overwrite data. Abort".format(folder))
    exit()


if error:
    exit()
                
mix_factor /= np.sum(mix_factor)


def parallel_equal(mix,lep):
    mix_e, mix_mu, mix_tau = mix_factor * mix
    lep_e, lep_mu, lep_tau = lep_factor * lep
    
    if diagnostic:
        print(Emma3.create_full_filename(folder, file_header, lep_e, lep_mu, lep_tau, mix_e, mix_mu, mix_tau))
        return 0
    
    if spectra_only:
        return Emma3.sterile_production(1000, 0.0071, mix_e, mix_mu, mix_tau, lep_e, lep_mu, lep_tau, make_plot=True, folder_name=folder, file_prefix = file_header), -1
    else:
        return NSH.solve(mix_e, mix_mu, mix_tau, lep_e, lep_mu, lep_tau, folder, file_header, make_plot=False, run_sp_again=run_all, run_pk_again=run_all)
    


if __name__ == '__main__':

    print("Data files written in {}/{}".format(folder, file_header))
    print("Mixing angles: ", mix_factor)
    print("Lepton numbers: ", lep_factor)
    if run_all:
        print("Spectra and P(k) run again.")

    mixang = np.linspace( 1e-10, 3e-9, 30)
#    mixang = np.linspace( 1e-10, 1.9e-9, 10)
    lep0 = np.linspace(0e-3, 6e-3, 31)   
#    lep0 = np.linspace(0e-3, 6e-3, 11)
    run_list = []
    new_list = []
    for i in range(len(mixang)):
        for j in range(len(lep0)):
            run_list.append((mixang[i], lep0[j]))
            new_list.append((i,j))

    p = mp.Pool(4)
    new_start_time = time.time()

    res = p.starmap(parallel_equal, run_list)
    p.close()
    p.join()

    print("Parallel, elapsed time = {} seconds".format(time.time()-new_start_time))


    if not diagnostic:
        np.savez("{}/{}-results".format(folder, file_header), results = res, mixangle = mixang, L0 = lep0, index = new_list)
    else:
        print("***Run code by turning off diagnostic with argument: d=F ***")