%% This script demonstrate the use of the area-metric code.

% Puffy interval dataset
puffy = [[3.5, 6.4];[6.9, 8.8];[6.1, 8.4];[2.8, 6.7];[3.5, 9.7];[6.5, 9.9];[0.15, 3.8];[4.5, 4.9];[7.1, 7.9]];

% Skinny interval dataset
skinny = [[1.00, 1.52];[2.68, 2.98];[7.52, 7.67];[7.73, 8.35];[9.44, 9.99];[3.66, 4.58]];

%% Define two dataset of the same size

% Let's consider a first dataset made of the left endpoints and a second
% data set made with the right endpoints of the puffy interval dataset:

d1 = puffy(:,1);
d2 = puffy(:,2);

% We can plot the two datasets using standard Matlab library:
figure(1)
ecdf(d1)
hold on
ecdf(d2)
legend('d1','d2')
grid on

%% Compute the area metric on same-size datasets

% The area metric is a stochastic distance which value coincides with the
% area between the two CDFs displayed in Figure (1).

% We can compute the area metric by simply invoking:

AM = areaMe(d1,d2);

sprintf('The area metric on same-size datasets is: %g',AM)

% In this particular case, whereby the two datasets come from a single
% interval dataset, the area metric can be interpreted as an indicator of
% puffiness in the interval dataset.

%% Datasets of different size

% Let's consider the following datasets:
d1 = puffy(:,1);
d2 = skinny(:,1);

% These two datasets have different size. Let's see how they look like:
figure(2)
ecdf(d1)
hold on
ecdf(d2)
legend('d1','d2')
grid on

%% Compute the area metric on different-size datasets

% The algorithm that computes the area metric on different size dataset is
% slightly more elaborated, and more inefficient. The inefficiencies are
% due to two separate problems:

% (1) The code has higher complexity O(n1+n2), while in the same-size case
% it had O(n1), where n1=n2
% (2) The code does not vectorize as well as the previous one. This problem
% can be addressed using proper code optimizations.

% For the reasons herein stated it is not as efficient to compute the area
% metric on datasets of different size. So if you can work with same-size 
% datasets.

% Nevertheless, the API to call the method is exactly the same as before,
% so the end user does not notice the difference between the two cases. 

% We can compute the area metric by simply invoking:

AM_2 = areaMe(d1,d2);

sprintf('The area metric on different-size datasets is: %g',AM_2)





