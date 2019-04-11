function diceNotCool = diceNotCool( labelImage1, labelImage2 )

% Value of largest label
n1 = max(labelImage1(:));
n2 = max(labelImage2(:));

intersectionArea = [];
selfAreas = [];

% Roll through each segmented object in ground truth
for i = 1:n1
    % Extract the i-th object
    l1 = (labelImage1 == i);

    for j = 1:n2
        % Extract the j-th island
        l2 = (labelImage2 == j);

        % Compute Dice
        a = numel(find(l1 == 1 & l2 == 1));

        if a > 0;
            intersectionArea = [intersectionArea; a];
            selfAreas = [selfAreas; [numel(find(l1 == 1)), numel(find(l2 == 1))]];
        end
    end % j
end %i

diceNotCool = 2*sum(intersectionArea(:))/(sum(selfAreas(:)) + eps);

return;

