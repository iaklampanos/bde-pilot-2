function [uu,vv,uv,CH,DB,clust,autoencoder,autoencoder2,autoencoder3,xtrain,xval,xtest] = eval_stacked(lvls,path)
    nc_lvls = ncread(path, 'num_metgrid_levels');
    blvls = ismember(nc_lvls,lvls);
    pos_lvls = find(blvls);
    uu =  ncread(path, 'UU');
    vv =  ncread(path, 'VV');
    u_size = size(uu);
    v_size = size(vv);
    uu = uu(:,:,pos_lvls,:);
    vv = vv(:,:,pos_lvls,:);
    uu = reshape(uu,[u_size(4),u_size(1)*u_size(2)*length(lvls)]);
    vv = reshape(vv,[v_size(4),v_size(1)*v_size(2)*length(lvls)]);
    uv = cat(2,uu,vv);
    disp(size(uv));
    [xtrain,xval,xtest] = dividerand(uv',0.2,0.2,0.6);
    disp(size(xtrain));
    autoencoder = trainAutoencoder(xtrain,100,'MaxEpoch',600);
    uv_enc = encode(autoencoder,xtrain);
    disp(size(uv_enc));
    autoencoder2 = trainAutoencoder(uv_enc,100,'MaxEpoch',600);
    uv_enc2 = encode(autoencoder2,uv_enc);
    autoencoder3 = trainAutoencoder(uv_enc2,7688*length(lvls),'MaxEpoch',600);
    k_data = decode(autoencoder3,xtest);
    disp(size(k_data'));
    [CH,DB,clust] = kmeans_c(k_data');
end

function[CH, DB, clust] = kmeans_c(x)
    for i = 1:30
        clust(: , i) = kmeans(x, i, 'MaxIter', 1000);
        disp(i)
    end
    CH = evalclusters(x, clust, 'CalinskiHarabasz');
    DB = evalclusters(x, clust, 'DaviesBouldin');
end