plot(x,u500db)
hold on
plot(x,u700db)
hold on
plot(x,u900db)
hold on
plot(x,umultdb)
legend('UV500','UV700','UV900','UV500 700 900','Location','northeast')
title('DaviesBouldin SingleAutoencoder cluster evaluation')