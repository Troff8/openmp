import subprocess
s ="""source /polusfs/setenv/setup.SMPI
#BSUB -J "OpenMP+MPI Job"
#BSUB -n {}
#BSUB -o "{}"
#BSUB -e "/dev/null"
#BSUB -R "affinity[core(4)]"
OMP_NUM_THREADS={}
mpiexec ./a.out {} {}"""
for l in ('1', 'pi'):
	for n in (128, 256):
		for threads in (1,2,4,8):
			for p in (2,4):
				output_file = '{}_{}_{}_{}.txt'.format(n ,threads, l,p)
				args = s.format(str(p),output_file,str(threads), str(n), str(l))
				subprocess.run(['bsub'],input=args.encode('utf-8'))


