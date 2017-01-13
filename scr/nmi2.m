function NMI = nmi(C,W)
   MI = 0;
   N = 0;
   for k=1:W.Count
       N = N + length(W(int2str(k)));
   end
   for k=1:W.Count
       for j=1:C.Count
           inter = ismember(W(int2str(k)),C(int2str(j)));
           n_inter = length(find(inter));
           P_inter = n_inter / N;
           P_log = (N*n_inter) / (length(W(int2str(k)))*length(C(int2str(j))));
           if P_log ~= 0
              MI = MI + P_inter*log(P_log);
           end
       end
   end
   Hc = 0;
   for k=1:W.Count
       Pc = length(W(int2str(k)))/N;
       Hc = Hc + (Pc*log(Pc));
   end
   Hc = Hc * -1;
   Hj = 0;
   for j=1:C.Count
       Pj = length(C(int2str(k)))/N;
       Hj = Hj + (Pj*log(Pj));
   end
   Hj = Hj * -1;
   NMI = MI / ((Hc+Hj)/2);
   