import gzip
import numpy as np

def parse_fast(file_path):
    tot = []
    with gzip.open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            if i % 2 != 0:
                data = list(map(float, line.split()))
                if len(data) == 8: # noReg
                    tot.append(data[6])
                elif len(data) == 14: # wReg
                    tot.append(data[6])
    return np.mean(tot), len(tot)

tot_no, count_no = parse_fast('/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj_nlo_run1_OO_5.36_kap6_noReg/datafile_partial.gz')
tot_w,  count_w  = parse_fast('/mnt/workstation/bottomonia_combined_analysis/inputs/qtraj-nlo-run2-00-5.36-kap6-wReg/datafile-avg.gz')

print("noReg Raw Mean:", tot_no, "(Count:", count_no, ")")
print("wReg Raw Mean:", tot_w, "(Count:", count_w, ")")

