function merge_mask_contour()

for i = 0:199999
    mask = imread(sprintf('output/mask/%d.png', i));
    cont = imread(sprintf('output/contour/%d.png', i));
    merg = uint8(mask>0) + 2*uint8(cont>0);
    imwrite(merg, sprintf('output/mask_contour/%d.png', i));
    if rem(i, 1000) == 0
        fprintf('%d\n', i);
    end
end

