function testLLE(params, samples, embeddings)

[nbSamples cols] = size(params);

colors = zeros(nbSamples, 3);

for i=1:nbSamples
    colors(i,1) = params(i,2)/ (2 * pi);
    colors(i,2) = params(i,1) / 50;
    colors(i,3) = 1 - colors(i,1);
end

scatter3(samples(:,1), samples(:,2), samples(:,3), [], colors);
figure;
scatter(embeddings(:,1), embeddings(:,2), [], colors);

end

