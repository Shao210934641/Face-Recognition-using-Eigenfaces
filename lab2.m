%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%coursework: face recognition with eigenfaces

% need to replace with your own path
addpath software;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Loading of the images: You need to replace the directory 
Imagestrain = loadImagesInDirectory ( 'images/training-set/23x28/');
[Imagestest, Identity] = loadTestImagesInDirectory ( 'images/testing-set/23x28/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Computation of the mean, the eigenvalues, amd the eigenfaces stored in the
%facespace:
ImagestrainSizes = size(Imagestrain);
Means = floor(mean(Imagestrain));
CenteredVectors = (Imagestrain - repmat(Means, ImagestrainSizes(1), 1));

CovarianceMatrix = cov(CenteredVectors);

[U, S, V] = svd(CenteredVectors);
Space = V(: , 1 : ImagestrainSizes(1))';
Eigenvalues = diag(S);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Display of the mean image:
MeanImage = uint8 (zeros(28, 23));
for k = 0:643
   MeanImage( mod (k,28)+1, floor(k/28)+1 ) = Means (1,k+1);
end
figure;
subplot (1, 1, 1);
imshow(MeanImage);
title('Mean Image');
%close


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display of the 20 first eigenfaces : Write your code here

% The first step is normalization. Make sure Shape is in 0,255 range
Space_normalize = normalize_0_255(Space);
% The second step is to Display of the 20 first eigenfaces
figure;
x=4; 
y=5; 
for i=1:20  % Display the first 20 eigenfaces in turn
      Image = uint8 (zeros(28, 23)); % Initialise the shape of eigenface
      for k = 0:643
      Image( mod (k,28)+1, floor(k/28)+1 ) = Space_normalize(i,k+1); % Reshape
      end
      subplot (x,y,i); % Set the relative position of each subplot 
      imshow (Image);
      title(i);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Projection of the two sets of images omto the face space:
Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);

Threshold =20;

TrainSizes=size(Locationstrain);
TestSizes = size(Locationstest);
Distances=zeros(TestSizes(1),TrainSizes(1));
%Distances contains for each test image, the distance to every train image.

for i=1:TestSizes(1),
    for j=1: TrainSizes(1),
        Sum=0;
        for k=1: Threshold,
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end,
     Distances(i,j)=Sum;
    end,
end,

Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
for i=1:70,
[Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end,


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Display of first 6 recognition results, image per image:
figure;
x=6;
y=2;
for i=1:6,
      Image = uint8 (zeros(28, 23));
      for k = 0:643
     Image( mod (k,28)+1, floor(k/28)+1 ) = Imagestest (i,k+1);
      end,
   subplot (x,y,2*i-1);
    imshow (Image);
    title('Image tested');
    
    Imagerec = uint8 (zeros(28, 23));
      for k = 0:643
     Imagerec( mod (k,28)+1, floor(k/28)+1 ) = Imagestrain ((Indices(i,1)),k+1);
      end,
     subplot (x,y,2*i);
imshow (Imagerec);
title(['Image recognised with ', num2str(Threshold), ' eigenfaces:',num2str((Indices(i,1))) ]);
end,



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%recognition rate compared to the number of test images: Write your code here to compute the recognition rate using top 20 eigenfaces.

Threshold = 20; % using 20 Eigenfaces 
% Calculate the distances from the project test images to the project training images 
Distances=zeros(TestSizes(1),TrainSizes(1));
for i=1:TestSizes(1),
    for j=1: TrainSizes(1),
        Sum=0;
        for k=1: Threshold,
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end,
     Distances(i,j)=Sum;
    end,
end,
 
Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
number_of_test_images=zeros(1,40);% Number of test images of one given person
for i=1:70, % 70 test images 
number_of_test_images(1,Identity(1,i))= number_of_test_images(1,Identity(1,i))+1;
[Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end,
 
rec_rate = []; % Initialize rec_rate
for i = 1: length(Imagestest(:,1))
    if ceil(Indices(i,1)/5) == Identity(i)
        rec_rate(i) = 1; % if match, 1
    else 
        rec_rate(i) = 0; % if not match, 0
    end
end
 
recognition_rate_20 = sum(rec_rate)/70 *100; % the output


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%effect of threshold (i.e. number of eigenfaces):   
averageRR=zeros(1,20);
for t=1:40,
  Threshold =t;  
Distances=zeros(TestSizes(1),TrainSizes(1));

for i=1:TestSizes(1),
    for j=1: TrainSizes(1),
        Sum=0;
        for k=1: Threshold,
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end,
     Distances(i,j)=Sum;
    end,
end,

Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
number_of_test_images=zeros(1,40);% Number of test images of one given person.%YY I modified here
for i=1:70,
number_of_test_images(1,Identity(1,i))= number_of_test_images(1,Identity(1,i))+1;%YY I modified here
[Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end,

rec_rate = [];
for i = 1: length(Imagestest(:,1))
    % if the indices of train does not match with Identity in test then rate is 0.
    if ceil(Indices(i,1)/5) == Identity(i)
        rec_rate(i) = 1;
    else 
        rec_rate(i) = 0;
    end
end
recognition_rate = sum(rec_rate)/70 *100;
averageRR(1,t) = recognition_rate;

end,
figure;
plot(averageRR(1,:));
xlabel('the number of Eigenfaces');
ylabel('Accuracy');
title('Recognition rate against the number of eigenfaces used');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%effect of K: You need to evaluate the effect of K in KNN and plot the recognition rate against K. Use 20 eigenfaces here.
AVG = zeros(1,40); 
tem = ones(5,1);
train_labels = [tem;2*tem;3*tem;4*tem;5*tem;6*tem;7*tem;8*tem;9*tem;10*tem;11*tem;12*tem;13*tem;14*tem;15*tem;16*tem;17*tem;18*tem;19*tem;20*tem;21*tem;22*tem;23*tem;24*tem;25*tem;26*tem;27*tem;28*tem;29*tem;30*tem;31*tem;32*tem;33*tem;34*tem;35*tem;36*tem;37*tem;38*tem;39*tem;40*tem];
for m=1:40,
    k = m; % K values are taken in order from 1 to 40.
    mdl = fitcknn(Locationstrain,train_labels,'NumNeighbors',k); % using KNN classifer
    class = predict(mdl,Locationstest); % predict labels
    rec_rate = [];
    for i = 1: length(Imagestest(:,1)),
        if ceil(class(i,1)) == Identity(i) % compare class and Identity 
            rec_rate(i) = 1; % if match, 1
        else 
            rec_rate(i) = 0; % if not match, 0
        end
    end
    recognition_rate_KNN = sum(rec_rate)/70 *100; % calculate the recognition rate 
    AVG(1,m) = recognition_rate_KNN; 
end
% Display the image 
figure;
plot(AVG(1,:));
xlabel('the value of k');
ylabel('Accuracy');
title('Recognition rate against the value of k used (KNN classifier)');
