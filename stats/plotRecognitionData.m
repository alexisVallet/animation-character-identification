function plotRecognitionData( csvFilename )
%PLOTRECOGNITIONDATA plots recognition rates data as computed by
% animation-character-identification.exe .

recognitionRates = csvread(csvFilename);

kValues = zeros(9);
minRates = zeros(9);
meanRates = zeros(9);
maxRates = zeros(9);

for i=1:8:72
    index = ((i - 1) / 8) + 1;
    kValues(index) = recognitionRates(i,1);
    meanRates(index) = mean(recognitionRates(i:i+7,5));
    minRates(index) = min(recognitionRates(i:i+7,5));
    maxRates(index) = max(recognitionRates(i:i+7,5));
end

plot(kValues, meanRates, kValues, minRates, kValues, maxRates);

end
