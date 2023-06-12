%% Resetting console
close all
clear
clc

%% Reading image
I = rgb2gray(imread("data/demo6.png"));

%% Segmentation

%%% Pre-processing
% Median filtering
I_segmentation = medfilt2(I,[3 3]);
% Anisotropic diffusion filtering
N_segmentation = 20;
I_segmentation = imdiffusefilt(I_segmentation,'ConductionMethod','quadratic','NumberOfIterations',N_segmentation);

%%% Otsu thresholding
I0_binary = imbinarize(I_segmentation);

%%% Highlighting perimeter
F_segmentation = bwperim(I0_binary);

%%% Cleaning perimeter
F_segmentation(:,1) = 0;
F_segmentation(:,end) = 0;
F_segmentation(1,:) = 0;
F_segmentation(end,:) = 0;

%%% Getting coordinates
[Y0,X0] = find(F_segmentation);

%% Edge detection

%%% Pre-processing
% Median filtering
I_edge_detection = medfilt2(I,[3 3]);
% Anisotropic diffusion filtering
N_edge_detection = 80;
I_edge_detection = imdiffusefilt(I_edge_detection,'ConductionMethod','quadratic','NumberOfIterations',N_edge_detection);

%%% Canny edge detection
sigma = 2;
t_low = .1;
t_high = .4;
F_edge_detection = edge(I_edge_detection,'canny',[t_low, t_high],sigma);

%%% Getting coordinates
[Y1,X1] = find(F_edge_detection);



%% Plotting result

%%% Plotting segmentation result
figure();
subplot(121);
imshow(I_segmentation,'Colormap',gray,'InitialMagnification','fit');
set(gca,'TickLabelInterpreter','latex','FontSize',20);
xlabel('x [px]','Interpreter','latex');
ylabel('y [px]','Interpreter','latex');
box on;
ax = gca;
axis on;
ax.LineWidth = 3;
title('Filtered image','Interpreter','latex');
c = colorbar();
c.TickLabelInterpreter = 'latex';

subplot(122);
imshow(I0_binary,'Colormap',gray,'InitialMagnification','fit');
hold on;
plot(X0,Y0,'.r','MarkerSize',10);
set(gca,'TickLabelInterpreter','latex','FontSize',20);
xlabel('x [px]','Interpreter','latex');
ylabel('y [px]','Interpreter','latex');
box on;
ax = gca;
axis on;
ax.LineWidth = 3;
set(gcf,'OuterPosition',[500 500 1000 500]);
title('Binary image','Interpreter','latex')
legend('Segmentation flame front','Location','north')
sgtitle('Segmentation','Interpreter','latex','FontSize',30)
c = colorbar();
c.TickLabelInterpreter = 'latex';

%%% Plotting edge detection result
figure();
subplot(121);
imshow(I_edge_detection,'Colormap',gray,'InitialMagnification','fit');
set(gca,'TickLabelInterpreter','latex','FontSize',20);
xlabel('x [px]','Interpreter','latex');
ylabel('y [px]','Interpreter','latex');
box on;
ax = gca;
axis on;
ax.LineWidth = 3;
title('Filtered image','Interpreter','latex');
c = colorbar();
c.TickLabelInterpreter = 'latex';

subplot(122);
imshow(rescale(imgradient(I_edge_detection)),'Colormap',turbo,'InitialMagnification','fit');
hold on;
plot(X1,Y1,'.k','MarkerSize',10);
set(gca,'TickLabelInterpreter','latex','FontSize',20);
xlabel('x [px]','Interpreter','latex');
ylabel('y [px]','Interpreter','latex');
box on;
ax = gca;
axis on;
ax.LineWidth = 3;
set(gcf,'OuterPosition',[500 500 1000 500]);
title('2D Gradient','Interpreter','latex')
legend('Edge detection flame front','Location','north')
c = colorbar();
c.TickLabelInterpreter = 'latex';

sgtitle('Edge detection','Interpreter','latex','FontSize',30)

%%% Final result
figure();
imshow(I,'Colormap',gray,'InitialMagnification','fit');
hold on;
plot(X0,Y0,'.r','MarkerSize',10);
hold on;
plot(X1,Y1,'.b','MarkerSize',10);
set(gca,'TickLabelInterpreter','latex','FontSize',20);
legend('Segmentation','Edge detection','Interpreter','latex','Location','northoutside');
xlabel('x [px]','Interpreter','latex');
ylabel('y [px]','Interpreter','latex');
box on;
ax = gca;
axis on;
ax.LineWidth = 3;
set(gcf,'OuterPosition',[500 500 500 500]);
title('Final result','Interpreter','latex');
c = colorbar();
c.TickLabelInterpreter = 'latex';







