function display_image_mask(n)

f = figure;
movegui(f, 'center');
subplot(1,2,1); imshow(sprintf('new_data_400x400/image/%d.png', n));
subplot(1,2,2); im=imread(sprintf('new_data_400x400/mask/%d.png', n)); imshow(36*im);
pause()

