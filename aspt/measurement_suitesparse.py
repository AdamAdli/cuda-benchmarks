from glob import glob
from tqdm.notebook import tqdm
import subprocess

matrices = glob("ppopp19_ae/data/*/*.mtx")
# bin_path = "ppopp19_ae/ASpT_SpMM_GPU/sspmm_32"
bin_paths = {
    'sspmm32': ('/home/jim/research/paramath/ASpT-mirror/ASpT_SpMM_GPU/sspmm_32', 32),
    'sspmm128': ('/home/jim/research/paramath/ASpT-mirror/ASpT_SpMM_GPU/sspmm_128', 128),
    'dspmm32': ('/home/jim/research/paramath/ASpT-mirror/ASpT_SpMM_GPU/dspmm_32', 32),
    'dspmm128': ('/home/jim/research/paramath/ASpT-mirror/ASpT_SpMM_GPU/dspmm_128', 128)
}

reps = 5
measurements = []
for idx, filename in enumerate(tqdm(matrices)):
    for bin_variant in bin_paths:
        bin_path, batch_size = bin_paths[bin_variant]
        print(idx, bin_variant, filename)
        for _ in range(reps):
            ret = subprocess.run(
                [bin_path, filename, str(batch_size)],
                capture_output=True
            )

            stderr = ret.stderr.decode("ascii").strip().split("\n")
            preproc_ms = float(stderr[0][len("preproc: "):])
            work_ms = float(stderr[1][len("work: "):])
            errors = int(stderr[2][len("num_diff: ")])

            measurements.append((filename, bin_variant, preproc_ms, work_ms, errors))

df = pd.DataFrame.from_records(measurements, columns=['filename', 'variant', 'preproc_ms', 'work_ms', 'errors'])
df.to_csv("suitesparse_measurements.csv")

print(df)

from scipy.stats.mstats import gmean
df1 = df.groupby(['filename', 'variant']).agg({"preproc_ms": gmean, "work_ms": gmean}).reset_index()
df1.to_csv("suitesparse_measurements_compressed.csv", index=False)

