docker run --runtime nvidia \
	-v /home/ybgao/spmm-benchmarks:/mnt \
	--shm-size=32G \
	-it spmm-benchmarks \
	bash

	# -p 8888:8888 \

