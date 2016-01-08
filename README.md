We prototype models in matlab and translate the time-consuming part to CMex for production.
Translation is sometimes painful. For a multi-dimension array, usually
we need to mannualy extract the data pointer, the dimension information,
check consistency, deal with strides, construct output etc..

MatlabMatrix.h provides several macros to automate these processes.
The basic uses can be illustrated as below.

The matlab code:
```matlab
function Test
Dim1 = 10;
Dim2 = 20;
Dim3 = 30;

rng(0823);
A = rand(Dim1,Dim2,Dim3);
B = rand(Dim1,Dim2);
C = zeros(Dim1,Dim2,Dim3); % intentionally allocate space in advance for mex call

C = DoSomething(A,B,Dim1,Dim2,Dim3);
end

function C = DoSomething(A,B,Dim1,Dim2,Dim3)
C = zeros(Dim1,Dim2,Dim3);
for i1=1:Dim1
    for i2=1:Dim2
        for i3=1:Dim3
            C(i1,i2,i3) = A(i1,i2,i3) * B(i1,i2);
        end
    end
end
end
```

With MatlabMatrix.h, the line calling DoSomething()
```matlab
C = DoSomething(A,B,Dim1,Dim2,Dim3);
```

can be replaced by the following mex file:
```C++
#include "MatlabMatrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	GET_INT(Dim1);
	GET_INT(Dim2);
	GET_INT(Dim3);
	GET_DM(A, 3);
	GET_DM(B, 2);
	GET_DM(C, 3);

	for (int i1 = 1; i1 <= Dim1 ; i1++)
	{
		for (int i2 = 1; i2 <= Dim2 ; i2++)
		{
			for (int i3 = 1; i3 <= Dim3 ; i3++)
			{
				C(i1, i2, i3) = A(i1, i2, i3)*B(i1, i2);
			}
		}
	}

	PUT(C);
}

```
