function Test
Dim1 = 10;
Dim2 = 20;
Dim3 = 30;

rng(0823);
A = rand(Dim1,Dim2,Dim3);
B = rand(Dim1,Dim2);
C = zeros(Dim1,Dim2,Dim3);

TestMex;
CMat = DoSomething(A,B,Dim1,Dim2,Dim3);
Err = max(abs(C(:)-CMat(:)))
end

function C = DoSomething(A,B,Dim1,Dim2,Dim3)
C = zeros(Dim1,Dim2,Dim3);
for i3=1:Dim3
    for i2=1:Dim2
        for i1=1:Dim1
            C(i1,i2,i3) = A(i1,i2,i3) * B(i1,i2);
        end
        C(:,i2,i3) = A(:,i2,i3);
    end
end
end