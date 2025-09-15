"""
Python 3 script for converting WCSim-like ROOT files to compressed .npz files (digitized hits only).

This tool exports only the digitized hit container (PMT, charge, time, trigger) per event,
and keeps minimal metadata:
- absolute path to the source ROOT file
- the event index within that ROOT file (ev)

Author: Zhihao Wang
"""

import argparse
from root_utils.root_file_utils import *
from root_utils.pos_utils import *

ROOT.gROOT.SetBatch(True)


def get_args():
    parser = argparse.ArgumentParser(description='dump WCSim data into numpy .npz file')
    parser.add_argument('input_files', type=str, nargs='+')
    parser.add_argument('-d', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def dump_file(infile, outfile):

    wcsim = WCSimFile(infile)
    nevents = wcsim.nevent

    # All data arrays are initialized here

    event_id = np.empty(nevents, dtype=np.int32)
    root_file = np.empty(nevents, dtype=object)

    digi_hit_pmt = np.empty(nevents, dtype=object)
    digi_hit_charge = np.empty(nevents, dtype=object)
    digi_hit_time = np.empty(nevents, dtype=object)
    digi_hit_trigger = np.empty(nevents, dtype=object)
    digi_hit_position = np.empty(nevents, dtype=object)

    for ev in range(wcsim.nevent):
        wcsim.get_event(ev)

        hits = wcsim.get_digitized_hits()
        digi_hit_pmt[ev]      = hits["pmt"]
        digi_hit_charge[ev]   = hits["charge"]
        digi_hit_time[ev]     = hits["time"]
        digi_hit_trigger[ev]  = hits["trigger"]
        digi_hit_position[ev] = hits["position"]
        
        event_id[ev] = ev
        root_file[ev] = infile

    np.savez_compressed(
        outfile,
        event_id=event_id,
        root_file=root_file,
        digi_hit_pmt=digi_hit_pmt,
        digi_hit_charge=digi_hit_charge,
        digi_hit_time=digi_hit_time,
        digi_hit_trigger=digi_hit_trigger,
        digi_hit_position=digi_hit_position
    )
    del wcsim


if __name__ == '__main__':

    config = get_args()
    if config.output_dir is not None:
        print("output directory: " + str(config.output_dir))
        if not os.path.exists(config.output_dir):
            print("                  (does not exist... creating new directory)")
            os.mkdir(config.output_dir)
        if not os.path.isdir(config.output_dir):
            raise argparse.ArgumentTypeError("Cannot access or create output directory" + config.output_dir)
    else:
        print("output directory not provided... output files will be in same locations as input files")

    file_count = len(config.input_files)
    current_file = 0

    for input_file in config.input_files:
        if os.path.splitext(input_file)[1].lower() != '.root':
            print("File " + input_file + " is not a .root file, skipping")
            continue
        input_file = os.path.abspath(input_file)

        if config.output_dir is None:
            output_file = os.path.splitext(input_file)[0] + '.npz'
        else:
            output_file = os.path.join(config.output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.npz')

        print("\nNow processing " + input_file)
        print("Outputting to " + output_file)

        dump_file(input_file, output_file)

        current_file += 1
        print("Finished converting file " + output_file + " (" + str(current_file) + "/" + str(file_count) + ")")

    print("\n=========== ALL FILES CONVERTED ===========\n")
