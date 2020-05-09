# -*- coding: utf-8 -*-

import tables
import numpy as np
import DataIO

WindowSize = [200, 400]

class TimeProfileData(tables.IsDescription) :
    EventID = tables.Int64Col(pos=0)
    Time = tables.Atom.from_sctype(np.uint8, shape=(200, 30))

def main(fipt, fopt):
    PETruth = DataIO.ReadPETruth(fipt)['DataFrame']
    ParticleType = DataIO.ReadParticleType(fipt)
    TimeProfile = DataIO.MakeTimeProfile(PETruth, WindowSize)
    TrainFile = tables.open_file(fopt, mode='w', title='TimeProfile', filters=tables.Filters(complevel=4))
    TrainTable = TrainFile.create_table('/', 'TimeProfile', TimeProfileData, 'TimeProfile')
    TimeTable.append(list(zip(ParticleType, TimeProfile)))
    TimeTable.flush()
    TimeFile.close()
    return


if __name__ == '__main__':
    import argparse
    psr = argparse.ArgumentParser()
    psr.add_argument('ipt', help='input file', type=str)
    psr.add_argument('-o', dest='opt', help='output')
    args = psr.parse_args()

    main(args.ipt, args.opt)
