function colors = labelColors( labels, rows, step )

colors = zeros(rows, 3);

for i=1:step:rows
    color = [rand(1) rand(1) rand(1)];
    for j=i:i+step-1
        colors(j,:) = color;
    end
end

end

