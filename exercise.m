close all
clear 
clc



%%




%% Filtering an image

% %%% Creating a synthetic OH-PLIF image
% 
% % Creating an initial image
% img = zeros(100);
% img(1:100,1:50) = 1;
% 
% % Adding synthetic noise
% mu_noise = .2;
% sigma_noise = .5;
% 
% noisy_image = imnoise(img,'gaussian',mu_noise,sigma_noise);
% img2 = zeros(size(img));
% img2(1:100,1:50) = 1;
% img3 = img2.*noisy_image;
% 
% figure();
% imshow(img3,'InitialMagnification','fit');
% 
% %%
% 
% 
% clear
% close all
% clc

%% Setup

% Adding script path
addpath("scripts/");

% Computing width of morphological mask
delta_L = [.44, .44, .47];
spatial_res = 0.0703; % mm/px
w = [.5, 1.5, 1.5].*delta_L./spatial_res;

%% Reading and plotting data
figure();

t = tiledlayout(3,3);
t.Padding = 'tight';

for case_id = 1:3
    for img_id = 1:3
        % Image path
        path = "images/Chaib2023/Case"+case_id+"/img"+img_id+".im7";
        % Reading .im7 image
        v = loadvec(char(path));
        I_raw = rot90(v.w);
        I_raw = I_raw-min(I_raw(:));
        I_cropped = I_raw(250:699,150:599);
        I_cropped = uint8(rescale(I_cropped).*255);
        % Detecting flame front
        F = fdetect(I_cropped, w(case_id));
        % Plotting
        nexttile();
        BG = I_cropped;
        imshow(BG,'InitialMagnification','fit','Colormap',turbo);
        hold on;
        [Y,X] = find(F);
        plot(X,Y,'.r','MarkerSize',10);
        set(gca,'YDir','reverse','TickLabelInterpreter','latex','FontSize',30);
        axis on;
        caxis([min(BG(:)) max(BG(:))]);
        xlabel('x [px]',"Interpreter","latex","FontSize",30);
        ylabel('y [px]',"Interpreter","latex","FontSize",30);
        box on;
        ax = gca;
        axis on;
        ax.LineWidth = 4; 
        ylim([50 400])
    end

end


set(gcf,'OuterPosition',[500 500 1000 1000]);

