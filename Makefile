sync:
	rsync -avz . ezegom@scc1.bu.edu:/usr4/ec527/ezegom/ec527/FinalProject

compile: 
	nvcc knapsack.cu -o knapsack
