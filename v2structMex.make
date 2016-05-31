MKLROOT = /share/apps/intel/composer_xe_2015.3.187/mkl/lib/intel64
MATLABROOT = /share/apps/matlab/R2014b
net:
	icpc v2structMex.cpp -O2 -DUSE_OMP \
		-o v2structMex.mexa64 \
		-Wl,--start-group \
		-Wl,--end-group -lpthread -lm -ldl \
		-I$(MATLABROOT)/extern/include -I$(MATLABROOT)/simulink/include -DMATLAB_MEX_FILE -ansi -D_GNU_SOURCE -fexceptions -fPIC -fno-omit-frame-pointer -pthread -shared -m64 -DNOCHECK -DMX_COMPAT_32 -DNDEBUG -Wl,--version-script,$(MATLABROOT)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined -Wl,-rpath-link,$(MATLABROOT)/bin/glnxa64 -L$(MATLABROOT)/bin/glnxa64 -lmx -ldl -lmex -lmat -lstdc++ -qopenmp
