main:
	g++ -O3 -std=c++11 -fopenmp prac_trofimov_ilya.cpp -o prac_2
	for N in 128 ; do \
		for omp_threads in 1 2; do \
			env OMP_NUM_THREADS=$$omp_threads ./prac_2 $$N 20 0.02 1; \
			env OMP_NUM_THREADS=$$omp_threads ./prac_2 $$N 20 0.02 pi; \
		done \
	done
