# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import pystan

psr = argparse.ArgumentParser()
psr.add_argument('ipt', type=str, help='input file')
args = psr.parse_args()

def main(Model_pkl):
    if not os.path.exists(Model_pkl):
        ocode = """
        data {
            int<lower=0> Npe;
            matrix[Npe, 3] pmtpos;
            vector[Npe] t;
        }
        transformed data {
            real c = 200;
        }
        parameters {
            vector[3] vortex;
            vector[3] t0;
            real t0;
            real<lower=0> sigma;
        }
        model {
            vector[Npe] trecon;
            for (n in 1:Npe) {
                trecon[n] = sqrt(dot_self(pmtpos[n]-vortex))/c + t0;
            }
            t ~ normal(trecon, sigma);
        }
        """
        sm = pystan.StanModel(model_code=ocode)
        with open(Model_pkl, 'wb') as f:
            pickle.dump(sm, f)
    return

if __name__ == '__main__':
    main(args.ipt)
