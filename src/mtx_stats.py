from typing import List, Tuple
from .mtx import *


def row_lengths(csr_pattern: CSRPattern):
  return [csr_pattern.row_ptrs[i+1] - csr_pattern.row_ptrs[i] for i in range(csr_pattern.nrows)]

def col_lengths(csr_pattern: CSRPattern):
  col_lengths = [0] * csr_pattern.ncols
  for col in csr_pattern.col_indices: col_lengths[col] += 1
  return col_lengths

def pattern_stats(csr_pattern: CSRPattern):
  _row_lengths = row_lengths(csr_pattern)
  _csr_lengths = col_lengths(csr_pattern)

  row_mean = np.mean(_row_lengths)
  row_std = np.std(_row_lengths)
  row_cov = row_std / row_mean
  col_mean = np.mean(_csr_lengths)
  col_std = np.std(_csr_lengths)
  col_cov = col_std / col_mean

  return {
      "sparsity_est": round(1 - len(csr_pattern.col_indices) / (csr_pattern.ncols * csr_pattern.nrows), 2),
      "row_mean": row_mean, "row_std": row_std, "row_cov": row_cov,
      "col_mean": col_mean, "col_std": col_std, "col_cov": col_cov,
      "cov_diff": abs(row_cov - col_cov),
      "nnz": csr_pattern.nnz
  }

def build_stats_df(csr_patterns: Tuple[str, List[CSRPattern]]):
  stats = []
  for name, pattern in csr_patterns:
    row = pattern_stats(pattern)
    row.update({"name": name, "nrows": pattern.nrows, 
                "ncols": pattern.ncols, "nnz": pattern.nnz})
    stats.append(row)
  return pd.DataFrame(stats)