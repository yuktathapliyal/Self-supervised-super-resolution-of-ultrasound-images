%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Downsampling by desired factor, for e.g., 2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SCALE = 2;
% filter = gaussian
% filter = anisotropic
input_path = ['Include input path to images'];
result_gaussian = ['Include path to result folder',num2str(SCALE)];
result_dir = dir([input_path,'/*.png']); % or whatever the filename extension is
for img_idx = 1:numel(result_dir)
    FileName = result_dir(img_idx).name;
    img = imread([input_path,'/', FileName]);
    img_2 = imresize(img, (1/SCALE), 'lanczos3');
    img_2_gaussian = imgaussfilt(img_2,0.6);
    img_2_gaussian_sp = imnoise(img_2_gaussian,'speckle',0.01);
    imwrite(img_2_gaussian_sp, [result_gaussian, '/', bicFileName])
end
