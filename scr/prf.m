function [P,R,F] = prf(A,B)
   P = precision(A,B); 
   R = recall(A,B);
   F = fscore(P,R);
end

function P = precision(A, B)
    P = length(intersect(A,B))/length(B);
end

function R = recall(A, B)
    R = length(intersect(A,B))/length(A);
end

function F = fscore(P, R)
    F = 2 * P * R / (P + R);
end

