# -*- coding: utf-8 -*-

import tables
import h5py
import numpy as np
import DataIO

WindowSize = [200, 400]


def main(fipt, fopt):
#   if 'problem' in fipt:
#        PETruth = DataIO.ReadPEGuess(fipt)['DataFrame']
#    else:
#        PETruth = DataIO.ReadPETruth(fipt)['DataFrame']
#    assert TimeProfile.shape[0] == EventID.shape[0]
    Waveform = DataIO.ReadWaveform(fipt)['DataFrame']
    EventID = np.unique(Waveform['EventID'].to_numpy())
    Wave = DataIO.MakeWaveform(Waveform)
    assert Wave.shape[0] == EventID.shape[0]

    TrainFile = tables.open_file(fopt, mode='a', filters=tables.Filters(complevel=4))

#   TimeAtom = tables.Atom.from_dtype(TimeProfile.dtype)
#   TimeArray = TrainFile.create_carray('/', 'TimeProfile', atom=TimeAtom, obj=TimeProfile)
    WaveAtom = tables.Atom.from_dtype(Wave.dtype)
    WaveArray = TrainFile.create_carray('/', 'Wave', atom=WaveAtom, obj=Wave)

    EventIDAtom = tables.Atom.from_dtype(EventID.dtype)
    EventIDArray = TrainFile.create_carray('/', 'EventID', atom=EventIDAtom, obj=EventID)

    iptfile = h5py.File(fipt, 'r', libver='latest', swmr=True)
    keys = list(iptfile.keys())
    iptfile.close()
    if 'ParticleTruth' in keys:
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
