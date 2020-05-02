import numpy as np
import tables
import pandas as pd
import numba


def ReadPETruth(filename) :
    iptfile = tables.open_file(filename, 'r')
    PETruth = iptfile.root.PETruth[:]
    keys = iptfile.root.PETruth.colnames
    PETruth_DataFrame = pd.DataFrame({key: PETruth[key] for key in keys})
    iptfile.close()
    return {"Data": PETruth, "DataFrame": PETruth_DataFrame}


@numba.jit
def MakeTimeProfile(WindowSize, nEvents, nChannels, PETruth) :
    TimeProfile = np.empty((nEvents, nChannels, WindowSize), dtype=np.uint8)
    for i in range(len(PETruth)) :
        TimeProfile[PETruth["EventID"][i] - 1][PETruth["ChannelID"][i]][PETruth["PETime"][i]] += 1
    return TimeProfile
