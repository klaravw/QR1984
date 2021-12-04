cudaQR: cudaQR.cu
	nvcc --use_fast_math -arch=sm_61 -O3 -o cudaQR cudaQR.cu -Xcompiler -fopenmp

cudaTestLU:
	nvcc -Xptxas -v,-O3 -arch=sm_70 --use_fast_math -Xcompiler -fopenmp,-O3 -I. -O3 -o cudaLUTest cudaLU.cu -lm ; \
	for M in `seq 1 32` ; do \
		./cudaLUTest 20000000 $$M ; \
	done

clean:
	rm cudaQR

run:
	./cudaQR 20000000 20



