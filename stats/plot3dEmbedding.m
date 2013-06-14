function plot3dEmbedding( embedding, step, labels, colors, name )

[rows cols] = size(embedding);

figure;
scatter3(embedding(:,1), embedding(:,2), embedding(:,3), 10, colors, 'filled');

I = 1:step:rows;
p = findobj(gca,'Type','Patch');
legend(p(I), labels)
title(name);

end
