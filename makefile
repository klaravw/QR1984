cudaQR: cudaQR.cu
	nvcc --use_fast_math -arch=sm_61 -O3 -o cudaQR cudaQR.cu -Xcompiler -fopenmp

cudaTestQR:
	nvcc --use_fast_math -arch=sm_61 -O3 -o cudaQRTest cudaQR.cu -Xcompiler -fopenmp
	for M in `seq 1 32` ; do \
		./cudaQRTest 20000000 $$M ; \
	done
	

clean:
	rm cudaQR

run:
	./cudaQR 20000000 20



