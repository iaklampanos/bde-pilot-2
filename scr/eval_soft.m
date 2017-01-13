function [images,labels,clust,autoencoder,autoencoder2,autoencoder3,soft,out] = eval_soft()
%     nc_lvls = ncread(path, 'num_metgrid_levels');
%     blvls = ismember(nc_lvls,lvls);
%     pos_lvls = find(blvls);
%     uu =  ncread(path, 'UU');
%     vv =  ncread(path, 'VV');
%     u_size = size(uu);
%     v_size = size(vv);
%     uu = uu(:,:,pos_lvls,:);
%     vv = vv(:,:,pos_lvls,:);
%     uu = reshape(uu,[u_size(4),u_size(1)*u_size(2)*length(lvls)]);
%     vv = reshape(vv,[v_size(4),v_size(1)*v_size(2)*length(lvls)]);
%     uv = cat(2,uu,vv);
%     disp(size(uv));
%     [xtrain,xval,xtest] = dividerand(uv',0.2,0.2,0.6);
%     disp(size(xtrain));
    images = loadMNISTImages('train-images-idx3-ubyte');
    labels = loadMNISTLabels('train-labels-idx1-ubyte');
    sizeimages = size(images);
    [ixtrain,ixval,ix] = dividerand(sizeimages(2),0.6,0.2,0.2);
    xtrain = images(:,ixtrain);
    xval = images(:,ixval);
    xtest = images(:,ix);
    ltrain = labels(ixtrain);
    lval = labels(ixval);
    ltest = labels(ix);
    autoencoder = trainAutoencoder(images,100,'MaxEpoch',600);
    uv_enc = encode(autoencoder,images);
    disp(size(uv_enc));
    autoencoder2 = trainAutoencoder(uv_enc,100,'MaxEpoch',600);
    uv_enc2 = encode(autoencoder2,uv_enc);
    disp(size(uv_enc2));
    autoencoder3 = trainAutoencoder(uv_enc2,10,'MaxEpoch',600);
    uv_enc2 = encode(autoencoder3,uv_enc2);
    for i=1:30
        disp(i);
        T = kmeans(uv_enc2',i,'MaxIter',1000);
        for k=1:60000
            for j=1:i
                if j==T(k)
                   t(k,j) = 1;
                end
            end
        end
        disp(size(t));
        soft = trainSoftmaxLayer(uv_enc2,t','LossFunction','crossentropy','MaxEpoch',600);
        qnet = stack(autoencoder,autoencoder2,autoencoder3,soft);
        qnet = train(qnet,images,t');
        out = qnet(images);
        for k=1:60000
            c(k) = find(out(:,k)==max(out(:,k)));
        end
        clust(:,i) = c;
    end
end


