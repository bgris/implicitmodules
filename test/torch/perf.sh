export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact,1,0
export MKL_NUM_THREADS=22

numactl -N 0 python3 perf_simple_shooting.py cpu euler 500 150 50 -it 10
numactl -N 0 python3 perf_simple_shooting.py cpu euler 1000 250 250 -it 10
numactl -N 0 python3 perf_simple_shooting.py cpu euler 2500 700 700 -it 10
numactl -N 0 python3 perf_simple_shooting.py cpu euler 5000 1500 1500 -it 10

numactl -N 0 python3 perf_simple_shooting.py cuda euler 500 150 50 -it 10
numactl -N 0 python3 perf_simple_shooting.py cuda euler 1000 250 250 -it 10
numactl -N 0 python3 perf_simple_shooting.py cuda euler 2500 700 700 -it 10
numactl -N 0 python3 perf_simple_shooting.py cuda euler 5000 1500 1500 -it 10

