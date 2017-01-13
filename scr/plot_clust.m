for i=1:12
    imgs = xtest(:, find(IDX==i)); display_network(imgs(:,1:12))
    saveas(gcf,sprintf('k%d',i),'jpg')
    pause(1)
end