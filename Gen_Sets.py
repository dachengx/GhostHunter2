# -*- coding: utf-8 -*-

import tables
import numpy as np
import DataIO

WindowSize = [200, 400]


def main(fipt, fopt):
    PETruth = DataIO.ReadPETruth(fipt)['DataFrame']
    TimeProfile = DataIO.MakeTimeProfile(PETruth, WindowSize)
    EventID = PETruth['EventID'].to_numpy()

    TrainFile = tables.open_file(fopt, mode='w', filters=tables.Filters(complevel=4))
    TimeAtom = tables.Atom.from_dtype(TimeProfile.dtype)
    TimeArray = TrainFile.create_carray('/', 'TimeProfile', atom=TimeAtom, obj=TimeProfile)

    EventIDAtom = tables.Atom.from_dtype(EventID.dtype)
    EventIDArray = TrainFile.create_carray('/', 'EventID', atom=EventIDAtom, obj=EventIDType)

    iptfile = tables.open_file(filename, 'r')
    keys = iptfile.root.ParticleTruth.colnames
    if 'Alpha' in keys:
        ParticleType = DataIO.ReadParticleType(fipt)
        ParticleAtom = tables.Atom.from_dtype(ParticleType.dtype)
        ParticleArray = TrainFile.create_carray('/', 'ParticleType', atom=ParticleAtom, obj=ParticleType)

    TrainFile.close()
    return


if __name__ == '__main__':
    import argparse
    psr = argparse.ArgumentParser()
    psr.add_argument('ipt', help='input file', type=str)
    psr.add_argument('-o', dest='opt', help='output')
    args = psr.parse_args()

    main(args.ipt, args.opt)
