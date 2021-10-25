#!/bin/bash
PARENT_DIR=$PWD
#mkdir GPU_SpMM_result

cd data
#rm SpMM_GPU_SP.out
#rm SpMM_GPU_DP.out
#rm SpMM_GPU_SP_preprocessing.out
#rm SpMM_GPU_DP_preprocessing.out

echo "dataset, ASpT_GFLOPs(K=32), ASpT_diff_%(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=128)" >> SpMM_GPU_SP.out
echo "dataset, preprocessing_ratio" >> SpMM_GPU_SP_preprocessing.out

for i in `ls -d */`
do
cd ${i}
ii=${i%/}
cd ..
echo -n ${ii} >> SpMM_GPU_SP.out
echo -n "," >> SpMM_GPU_SP.out
echo -n ${ii} >> SpMM_GPU_SP_preprocessing.out
echo -n "," >> SpMM_GPU_SP_preprocessing.out
#${PARENT_DIR}/cuSPARSE_SpMM/cuSPARSE_SP ${ii}/${ii}.mtx
echo ${PARENT_DIR}/ASpT_SpMM_GPU/sspmm_32 ${ii}/${ii}.mtx 32
${PARENT_DIR}/ASpT_SpMM_GPU/sspmm_32 ${ii}/${ii}.mtx 32
#${PARENT_DIR}/ASpT_SpMM_GPU/sspmm_128 ${ii}/${ii}.mtx 128
#${PARENT_DIR}/merge-spmm/bin/gbspmm  --tb=128 --nt=32 --max_ncols=32 --iter=1 ${ii}/${ii}.mtx
#${PARENT_DIR}/merge-spmm/bin/gbspmm  --tb=128 --nt=32 --max_ncols=128 --iter=1 ${ii}/${ii}.mtx
echo >> SpMM_GPU_SP.out
echo >> SpMM_GPU_SP_preprocessing.out
done

echo "dataset, ASpT_GFLOPs(K=32), ASpT_diff_%(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=128)" >> SpMM_GPU_DP.out
echo "dataset, preprocessing_ratio" >> SpMM_GPU_DP_preprocessing.out

# for i in `ls -d */`
# do
# cd ${i}
# ii=${i/\//}
# cd ../
# echo -n ${ii} >> SpMM_GPU_DP.out
# echo -n "," >> SpMM_GPU_DP.out
# echo -n ${ii} >> SpMM_GPU_DP_preprocessing.out
# echo -n "," >> SpMM_GPU_DP_preprocessing.out
# #${PARENT_DIR}/cuSPARSE_SpMM/cuSPARSE_DP ${ii}/${ii}.mtx
# ${PARENT_DIR}/ASpT_SpMM_GPU/dspmm_32 ${ii}/${ii}.mtx 32
# ${PARENT_DIR}/ASpT_SpMM_GPU/dspmm_128 ${ii}/${ii}.mtx 128
# echo >> SpMM_GPU_DP.out
# echo >> SpMM_GPU_DP_preprocessing.out
# done

# rm gmon.out
# mv *.out ${PARENT_DIR}/GPU_SpMM_result
# cd ${PARENT_DIR}

#for i in `ls -d *.mtx`
#do
#${PARENT_DIR}/ASpT_SpMM/sddmm_32 ${i} 32
#${PARENT_DIR}/ASpT_SpMM/sddmm_128 ${i} 128
#done


