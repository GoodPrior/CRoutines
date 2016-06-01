function compile_v2structMex
cd(fileparts(which(mfilename)));
mex -v v2structMex.cpp OPTIMFLAGS="/Z7" LINKOPTIMFLAGS="/DEBUG"; 
end