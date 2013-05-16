function plotRecognitionData( recognitionRates, colX, colY)
%PLOTRECOGNITIONDATA plots recognition rates data as computed by
% animation-character-identification.exe .

[rows cols] = size(recognitionRates);

if colX < 1 || colX > cols || colY < 1 || colY > cols
    ME = MException('colX or colY is out of bounds');
    throw(ME);
end

xValues = [];
nbXValues = 0;
nbYPerX = [];
prevValue = -1;

% first count the number of different value in colX
for i=1:rows
    current = recognitionRates(i, colX);
    if prevValue ~= current
        prevValue = current;
        nbXValues = nbXValues + 1;
        nbYPerX = [nbYPerX 1];
        xValues = [xValues current];
    else
        nbYPerX(length(nbYPerX)) = nbYPerX(length(nbYPerX)) + 1;
    end
end

means = zeros(nbXValues);
mins = zeros(nbXValues);
maxs = zeros(nbXValues);
currentOffset = 1;

for i=1:nbXValues
    samples = recognitionRates(currentOffset:currentOffset + nbYPerX(i) - 1, colY);
    means(i) = mean(samples);
    mins(i) = min(samples);
    maxs(i) = max(samples);
    currentOffset = currentOffset + nbYPerX(i);
end

plot(xValues, mins, xValues, means, xValues, maxs);

end
