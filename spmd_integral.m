clear all;

if isempty(gcp())
 parpool();
end

nworkers = gcp().NumWorkers;

f = @(x, y) x.^2 + y.^2;
xlim = [0,2];
ylim = [0,3];
limitMatrix = zeros(nworkers, 4);
delx = (xlim(2) - xlim(1))/sqrt(nworkers);
dely = (ylim(2) - ylim(1))/sqrt(nworkers);

for i = 1:sqrt(nworkers)
    for j = 1:sqrt(nworkers)
    index = (i - 1) * sqrt(nworkers) + j;
    limitMatrix(index, 1) = xlim(1) + (j - 1) * delx;
    limitMatrix(index, 2) = xlim(1) + j * delx;
    limitMatrix(index, 3) = ylim(1) + (i - 1) * dely;
    limitMatrix(index, 4) = ylim(1) + i * dely;
    end
end

spmd
    workerIndex = labindex();
    ainit = limitMatrix(workerIndex, 1);
    bfin = limitMatrix(workerIndex, 2);
    cinit = limitMatrix(workerIndex, 3);
    dfin = limitMatrix(workerIndex, 4); 
    locint = integral2(f,ainit,bfin,cinit,dfin); 
    totalint = gplus(locint); 
end

totalvalue = totalint{1}
