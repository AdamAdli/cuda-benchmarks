import numpy as np
import scipy.sparse as ss
from cpp_lib import *
import os
from mtx import *
import pandas as pd

REPS=32
BURN_ITERS=32

def sgk_op_runtime(A, batch_size, reps=REPS, burn_iters=BURN_ITERS):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the 
	sgk code (cpp code binded in python)
	"""
	m, k = A.shape
	n = batch_size
	values, row_indices, row_offsets, column_indices = dense_to_sparse(A)
	nonzeros = values.size
	B = np.random.randn(k, batch_size)
	times = []
	for _ in range(burn_iters):
		sgkSPARSE(m, n, k, nonzeros, values, row_indices,
					row_offsets, column_indices, B.flatten())
	for _ in range(reps):
		times.append(sgkSPARSE(m, n, k, nonzeros, values, row_indices,
								row_offsets, column_indices, B.flatten()))
	return np.median(times)*1e-6

def get_batch_size(path):
	"""
	given the path to the batchsize file, return a dictionary with key as the 
	layer name and value as the corresponding batchsize
	"""
	result = {}
	with open(path, 'r') as f:
		content = f.readlines()
		content = [i.strip() for i in content]
		for i in content:
			layer_name, batch_size  = i.split(",")
			result[layer_name] = int(batch_size)
	return result

def get_all_subdirs(root_path):
	pruning_methods = os.listdir(root_path)
	all_dirs = {}
	for method in pruning_methods:
		method_dir = os.path.join(root_path, method)
		sparsities = os.listdir(method_dir)
		for sparsity in sparsities:
			sparsity_path = os.path.join(method_dir, sparsity)
			all_dirs[(method, float(sparsity))] = sparsity_path
	return all_dirs

def resnet_exp(root_path, batch_size = 32):
	"""
	Given the path to the resnet matrix directory (path to rn50), return a dictionary containing the
	pruned weight matrix layer name as well as the corresponding run time

	EG: return {botleneck_1_xxx: 3.2}
	"""
	def process_subdir(subdir_path):
		mtx_files = os.listdir(subdir_path)
		batch_size_dic = get_batch_size('/mnt/benchmark/dlmc/rn50_batchsizes.txt')
		result = {}
		for mtx_file in mtx_files:
			mtx_file_path = os.path.join(subdir_path, mtx_file)
			mtx_name = mtx_file.split(".")[0]
			A = get_mtx(mtx_file_path)
			B_batch_size = batch_size_dic[mtx_name]
			cur_runtime = sgk_op_runtime(A, B_batch_size)
			result[mtx_name] = cur_runtime
		return result

	resnet_time_dic = {}

	# get all the paths
	# pruning_methods = os.listdir(root_path)
	# all_dirs = {}
	# for method in pruning_methods:
	# 	method_dir = os.path.join(root_path, method)
	# 	sparsities = os.listdir(method_dir)
	# 	for sparsity in sparsities:
	# 		sparsity_path = os.path.join(method_dir, sparsity)
	# 		all_dirs[(method, float(sparsity))] = sparsity_path
	all_dirs = get_all_subdirs(root_path)

	for k, v in all_dirs.items():
		#k, v example:
		#('extended_magnitude_pruning', 0.96) /mnt/benchmark/dlmc/rn50/extended_magnitude_pruning/0.96
		resnet_time_dic[k] = process_subdir(v)
	return resnet_time_dic	

def transformer_exp(root_path):
	"""
	Given root path of transformer matrices, return the runtime of each group
	"""
	def cat_by_layers(subdir_path):
		files = os.listdir(subdir_path)
		result = {}
		for file in files:
			_, functionality, _, layer_number = file.split("_")[:4]
			k = (functionality, layer_number)
			if (k not in result):
				result[k] = [file]
			else:
				result[k].append(file)
		return result
	
	def layer_timing_logic():
		"""
		TODO
		"""
		return False

	all_dirs = get_all_subdirs(root_path)
	result = {}
	# for subdir in all_dirs:
			

def rn50_result_to_csv(result, fname):
  """
  Given the result (by calling resnet_exp function), save it
  into a csv, maching the table here:
  https://github.com/jimgao1/spmm-benchmarks/blob/master/aspt/measurements.ipynb
  """
  def dir_helper(method, accuracy, mtx_runtime):
    result = []
    for mtx_name, mtx_runtime in mtx_runtime.items():
      dir_name = "dlmc/rn50/{}/{}/{}.mtx".format(method, accuracy, mtx_name)
      result.append((dir_name, mtx_runtime))
    return result 

  acc = []
  for k, v in result.items():
    method, accuracy = k
    mtx_runtime = v
    tmp = dir_helper(method, accuracy, mtx_runtime)
    acc.extend(tmp)
  df = pd.DataFrame.from_records(acc, columns=["filename", "time_ms"])
  df.to_csv("{}.csv".format(fname))



# if __name__ == "__main__":
# 	result = resnet_exp("/mnt/benchmark/dlmc/rn50")
# 	for k, v in result.items():
# 		print(k, v)
# 		break