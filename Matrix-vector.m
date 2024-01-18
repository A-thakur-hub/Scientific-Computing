% CPU - single precision
A = rand(2000,2000,"single");
b = rand(2000,1,"single");
n = 20000;

tic
for i=1:n
    product = A*b;
end
toc

% CPU - double precision
A = rand(2000,2000,"double");
b = rand(2000,1,"double");
n = 20000;

tic
for i=1:n
    product = A*b;
end
toc

% GPU - single precision
A = rand(2000,2000,"single","gpuArray");
b = rand(2000,1,"single","gpuArray");
n = 20000;

tic
for i=1:n
    product = A*b;
end
toc

% GPU - double precision
A = rand(2000,2000,"double","gpuArray");
b = rand(2000,1,"double","gpuArray");
n = 20000;

tic
for i=1:n
    product = A*b;
end
toc