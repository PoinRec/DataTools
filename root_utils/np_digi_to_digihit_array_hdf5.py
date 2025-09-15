'''
Python 3 script for converting and merging .npz files (digitized hits only) into a single HDF5 file.

This version assumes the .npz files contain only the following arrays:
    - event_id              (shape: (N,))
    - root_file             (shape: (N,))
    - digi_hit_pmt          (shape: (N,), each element is (Nhits_event,))
    - digi_hit_charge       (shape: (N,), each element is (Nhits_event,))
    - digi_hit_time         (shape: (N,), each element is (Nhits_event,))
    - digi_hit_trigger      (shape: (N,), each element is (Nhits_event,))
    - digi_hit_position     (shape: (N,), each element is (Nhits_event, 3))

The output HDF5 file contains:
    - root_files        (N,)        : original ROOT file paths
    - event_ids         (N,)        : event indices
    - event_hits_index  (N,)        : start index of each event's hits in the flattened arrays
    - hit_time          (Nhits,)    : hit times (ns), shifted so the first hit per event is at 0
    - hit_charge        (Nhits,)    : hit charges
    - hit_pmt           (Nhits,)    : PMT ids

Author: Zhihao Wang
'''

import numpy as np
import os
import sys
import subprocess
from datetime import datetime
import argparse
import h5py
import warnings


def get_args():
    parser = argparse.ArgumentParser(description='convert and merge .npz (digi-hits only) to HDF5')
    parser.add_argument('input_files', type=str, nargs='+')
    parser.add_argument('-o', '--output_file', type=str, required=True)
    parser.add_argument('-m', '--max-hit-time', type=float, default=100.0, help='only keep hits with t < max-hit-time (ns)')
    return parser.parse_args()


if __name__ == '__main__':
    config = get_args()
    print("ouput file:", config.output_file)
    f = h5py.File(config.output_file, 'w')

    script_path = os.path.dirname(os.path.abspath(__file__))
    try:
        git_status = subprocess.check_output(
            ['git', '--no-optional-locks', '-C', script_path,
             'status', '--porcelain', '--untracked-files=no']).decode()
        if git_status:
            warnings.warn("Script directory not clean:\n" + git_status)
        git_describe = subprocess.check_output(
            ['git', '--no-optional-locks', '-C', script_path,
             'describe', '--always', '--long', '--tags', '--dirty']).decode().strip()
    except Exception as e:
        git_describe = f"(no-git) {e}"
    f.attrs['git-describe'] = git_describe
    f.attrs['command'] = " ".join(sys.argv)
    f.attrs['timestamp'] = str(datetime.now())

    total_rows = 0
    total_hits = 0
    min_hits = 1
    good_rows = 0
    good_hits = 0
    print("counting events and hits, in files")
    file_event_triggers = {}
    for input_file in config.input_files:
        print(input_file, flush=True)
        if not os.path.isfile(input_file):
            raise ValueError(input_file + " does not exist")

        npz = np.load(input_file, allow_pickle=True)
        hit_triggers = npz['digi_hit_trigger']
        hit_times = npz['digi_hit_time']

        n_rows = hit_triggers.shape[0]
        total_rows += n_rows
        event_triggers = np.full(n_rows, np.nan, dtype=float)

        for i, (trigs_i, times_i) in enumerate(zip(hit_triggers, hit_times)):
            if len(trigs_i) == 0:
                continue
            uniq = np.unique(trigs_i)
            min_t_per_trig = []
            for t in uniq:
                mask_t = (trigs_i == t)
                if np.any(mask_t):
                    min_t_per_trig.append((t, np.min(times_i[mask_t])))
            if not min_t_per_trig:
                continue
            first_trigger, _ = min(min_t_per_trig, key=lambda x: x[1])

            sel = (trigs_i == first_trigger) & (times_i < config.max_hit_time)
            nhits = int(np.count_nonzero(sel))
            total_hits += nhits
            if nhits >= min_hits:
                event_triggers[i] = first_trigger
                good_rows += 1
                good_hits += nhits

        file_event_triggers[input_file] = event_triggers

    print(len(config.input_files), "files with", total_rows, "events with", total_hits, "hits")
    print(good_rows, "events with at least", min_hits, "hits for a total of", good_hits, "hits")

    dset_PATHS = f.create_dataset("root_files",       shape=(total_rows,),   dtype=h5py.special_dtype(vlen=str))
    dset_IDX   = f.create_dataset("event_ids",        shape=(total_rows,),   dtype=np.int32)
    dset_EHI   = f.create_dataset("event_hits_index", shape=(total_rows,),   dtype=np.int64)
    dset_htime = f.create_dataset("hit_time",         shape=(good_hits,),    dtype=np.float32)
    dset_hq    = f.create_dataset("hit_charge",       shape=(good_hits,),    dtype=np.float32)
    dset_hpmt  = f.create_dataset("hit_pmt",          shape=(good_hits,),    dtype=np.int32)

    offset = 0
    hit_offset = 0

    for input_file in config.input_files:
        print(input_file, flush=True)
        if not os.path.isfile(input_file):
            raise ValueError(input_file+" does not exist")
        npz = np.load(input_file, allow_pickle=True)

        event_triggers = file_event_triggers[input_file]
        n_rows = event_triggers.shape[0]
        offset_next = offset + n_rows

        event_ids  = npz['event_id']
        root_files = npz['root_file']
        dset_IDX[offset:offset_next]   = event_ids
        dset_PATHS[offset:offset_next] = root_files

        hit_times_list   = npz['digi_hit_time']
        hit_charges_list = npz['digi_hit_charge']
        hit_pmts_list    = npz['digi_hit_pmt']
        hit_trigs_list   = npz['digi_hit_trigger']

        for i in range(n_rows):
            dset_EHI[offset + i] = hit_offset
            trig_idx = event_triggers[i]

            times_i   = hit_times_list[i]
            charges_i = hit_charges_list[i]
            pmts_i    = hit_pmts_list[i]
            trigs_i   = hit_trigs_list[i]

            if not np.isnan(trig_idx):
                sel = (trigs_i == int(trig_idx)) & (times_i < config.max_hit_time)
                idx = np.where(sel)[0]
            else:
                idx = np.array([], dtype=int)

            if idx.size > 0:
                next_hit = hit_offset + idx.size
                t0 = float(np.min(times_i[idx]))
                dset_htime[hit_offset:next_hit] = (times_i[idx] - t0).astype(np.float32)
                dset_hq[hit_offset:next_hit]    = charges_i[idx].astype(np.float32)
                dset_hpmt[hit_offset:next_hit]  = pmts_i[idx].astype(np.int32)
                hit_offset = next_hit

        offset = offset_next

    f.close()
    print("saved", hit_offset, "hits in", offset, "good events (each with at least", min_hits, "hits)")