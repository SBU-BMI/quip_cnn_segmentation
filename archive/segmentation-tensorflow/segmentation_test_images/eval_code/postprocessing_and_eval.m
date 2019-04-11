function segmentation_eval(ground_truth_folder, pred_folders)

fprintf('Thres\tDNC\tDice\tPixelWiseAccuracy\n');
max_dice = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setting
ground_truth_folder = '../miccai15/';
pred_folders = {'../miccai15_generative_2017-11-15_15-05-47'};
use_ostu = 0;
compute_dnc = 0;
min_size = 0;
fill_wholes = 0;
verbal = 0;
save_visual = 0;
thres_array = [0.80:0.05:1.20];
% Setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for thres = thres_array
    dice_score = eval_method(pred_folders, ground_truth_folder, ...
        thres, use_ostu, compute_dnc, verbal, min_size, save_visual, fill_wholes);
    max_dice = max([max_dice, dice_score]);
end
fprintf('\n%.4f\n', max_dice);

function dice_score = eval_method(fs, mask_folder, ...
    thres, use_ostu, compute_dnc, verbal, min_size, save_visual, fill_wholes)

files = dir([mask_folder, '/*_mask.png']);
sum_corrt = 0;
sum_pixel = 0;
sum_nnz_gt = 0;
sum_nnz_method = 0;
sum_nnz_overlap = 0;
sum_dnc = 0;
for file = files'
    mmskfn = file.name;
    predfn = [mmskfn(1:end-9), '_pred.png'];
    mmsk = imread([mask_folder, '/', mmskfn]);

    image = imread([mask_folder, '/', mmskfn(1:end-9), '.png']);
    pred = double(imread([fs{1}, '/', predfn])) / length(fs);
    for ind = 2:length(fs)
        pred = pred + double(imread([fs{ind}, '/', predfn])) / length(fs);
    end
    pred = imresize(uint8(pred), [size(mmsk, 1), size(mmsk, 2)]);

    if length(size(mmsk)) > 2
        mmsk = mmsk(:,:,1);
    end
    mask = (mmsk > 0);
    if use_ostu > 0.5
        thres_use = thres*255*graythresh(pred);
        pred = (pred > thres_use);
    else
        thres_use = thres*255*0.5;
        pred = (pred > thres_use);
    end

    bwpred = bwlabel(pred);
    for bw = 1:max(bwpred(:))
        if nnz(bwpred == bw) < min_size
            pred(bwpred == bw) = 0;
        end
    end

    if fill_wholes > 0.5
        pred = imfill(pred, 'holes');
    end

    if save_visual > 0.5
        pred_edge = edge(pred, 'canny');
        image(:,:,2) = image(:,:,2)+255*uint8(pred_edge);
        imwrite(image, sprintf('visual/%d_%.2f_%d_%d_%s.png', use_ostu, thres, min_size, fill_wholes, mmskfn(1:end-9)));
    end

    corrt = nnz(mask(:) == pred(:));
    pixel = length(mask(:));
    nnz_gt = nnz(mask);
    nnz_method = nnz(pred);
    nnz_overlap = nnz(mask & pred);

    if compute_dnc > 0.5
        dnc = diceNotCool(bwlabel(mask), bwlabel(pred, 4));
    else
        dnc = 0;
    end
    if verbal > 0.5
        fprintf('%.1f\t%.4f\t%.4f\t%.4f\t%d\t%s\n', thres_use, dnc, ...
            dice_coef(nnz_overlap, nnz_method, nnz_gt), corrt / pixel, pixel, mmskfn);
    end

    sum_corrt = sum_corrt + corrt;
    sum_pixel = sum_pixel + pixel;
    sum_nnz_gt = sum_nnz_gt + nnz_gt;
    sum_nnz_method = sum_nnz_method + nnz_method;
    sum_nnz_overlap = sum_nnz_overlap + nnz_overlap;
    sum_dnc = sum_dnc + dnc * pixel;
end

dice_score = dice_coef(sum_nnz_overlap, sum_nnz_method, sum_nnz_gt);
fprintf('%.2f\t%.4f\t%.4f\t%.4f\n', thres, sum_dnc / sum_pixel, dice_score, sum_corrt / sum_pixel);

function dc = dice_coef(nnz_overlap, nnz_method, nnz_gt)

dc = 2 * nnz_overlap / (nnz_method + nnz_gt);

