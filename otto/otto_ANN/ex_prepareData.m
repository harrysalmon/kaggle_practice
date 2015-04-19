% data file

%% Initialization
clear; close all; clc

% path of the training data
dirPath = '/home/harry/Documents/git/kaggle/otto/otto_ANN';

% read csv file containing training labels
fid = fopen('train.csv');
C = textscan(fid, '%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%s','Delimiter', ',', 'headerLines', 1);
fclose(fid);

m = size(C{1},1);
n = 93;

X = zeros(m,n);
y = zeros(m,1);
for i=1:n
    X(:,i) = C{i+1};
end

for j=1:m
   y(j) = str2num(C{95}{j}(7));
end

indx1 = randsample(m,floor(m*0.8),false);
indx2 = setdiff(1:m,indx1)'; % data in first argument not in second

X_val = X(indx2,:);
y_val = y(indx2);

X = X(indx1,:);
y = y(indx1);

y_mat = zeros(length(indx1),9);
y_mat_val = zeros(length(indx2),9);

for i=1:9
    y_mat(y==i,i)=1;
    y_mat_val(y_val==i,i)=1;
end

% read csv file containing training labels
fid = fopen('test.csv');
D = textscan(fid, '%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%s','Delimiter', ',', 'headerLines', 1);
fclose(fid);

m = size(D{1},1);
X_sub = zeros(m,n);
for i=1:n
    X_sub(:,i) = D{i+1};
end

save('train.mat', 'X', 'y', 'X_val', 'y_val', 'y_mat', 'y_mat_val', 'X_sub') 
