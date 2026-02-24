HOST    = dtu
REMOTE  = ~/COURSES/CUDA-special

# Files to exclude from sync (mirrors .gitignore)
RSYNC_EXCLUDES = \
	--exclude='.git/' \
	--exclude='_/' \
	--exclude='AGENTS.md' \
	--exclude='*.o' \
	--exclude='*.nvvp' \
	--exclude='*.nsys-rep' \
	--exclude='*.ncu-rep' \
	--exclude='*.qdrep' \
	--exclude='*.sqlite' \
	--exclude='*.ptx' \
	--exclude='*.cubin' \
	--exclude='*.fatbin' \
	--exclude='core' \
	--exclude='core.*' \
	--exclude='.DS_Store' \
	--exclude='*.swp' \
	--exclude='*.swo' \
	--exclude='*~' \
	--exclude='week2/cpu' \
	--exclude='week2/cuda' \
	--exclude='week2/omp_offload' \
	--exclude='week2/reduction' \
	--exclude='week3/ex1' \
	--exclude='week3/ex2' \
	--exclude='week3/compare'

.PHONY: sync

sync:
	rsync -avz --progress $(RSYNC_EXCLUDES) ./ $(HOST):$(REMOTE)/
