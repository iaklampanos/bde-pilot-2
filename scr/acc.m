function ACC = acc(C,W)
   N = 0;
   for k=1:W.Count
       N = N + length(W(int2str(k)));
   end
   MP = [];
   for j=1:C.Count
       MPwkcj = [];
       for k=1:W.Count
           intersec = ismember(W(int2str(k)),C(int2str(j)));
           n_intersec = length(find(intersec));
           P_cjwk = n_intersec / length(W(int2str(k)));
           P_wk = length(W(int2str(k)))/N;
           P_cj = length(C(int2str(j)))/N;
           P_wkcj = (P_cjwk * P_wk)/P_cj;
           MPwkcj = [MPwkcj P_wkcj];
       end
       MP = [MP max(MPwkcj)];
       disp(MP);
       ACC = sum(MP) / double(W.Count);
   end
end