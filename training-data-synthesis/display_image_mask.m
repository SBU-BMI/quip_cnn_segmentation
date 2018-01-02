function display_image_mask()

for n = 1:1:300
    f = figure;
    movegui(f, 'center');
    subplot(1,3,1); imshow(sprintf('output/image/%d.png', n));
    subplot(1,3,2); im=imread(sprintf('output/mask/%d.png', n)); imshow(36*im);
    subplot(1,3,3); imshow(sprintf('output/refer/%d.png', n));
    pause()
end

