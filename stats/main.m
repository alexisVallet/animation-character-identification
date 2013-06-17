format short,  format compact; clear all;  fclose all; clc; close all;


load jaffe_single_face_affine
X = jaffe_single_face_affine;


K = 8;
d = 4;

Y = lle(X,K,d);

figure (16); hold on;
for j = 1:size(Y,2)
    plot(Y(1,j),Y(2,j),'b:.');
%     scatter(Y(1,:),Y(2,:),100,'.');
    dot = [j];
    text(Y(1,j),Y(2,j),num2str(j));
    
end
% axis([-3.5,3.5,-3.5,3.5]);
