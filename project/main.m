% 1. 設定資料夾路徑
inputFolder = '/Users/tingyuko/Downloads/Dog_heart_2/Train'; % 資料集所在資料夾
outputFolder = '/Users/tingyuko/Downloads/processed_test'; % 處理後影像的儲存資料夾
categories = {'Large', 'Normal', 'Small'}; % 類別資料夾名稱

% 確保輸出資料夾存在
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% 2. 遍歷每一類別資料夾並處理所有影像
for i = 1:length(categories)
    categoryFolder = fullfile(inputFolder, categories{i});
    outputCategoryFolder = fullfile(outputFolder, categories{i});
    
    % 確保類別的資料夾存在
    if ~exist(outputCategoryFolder, 'dir')
        mkdir(outputCategoryFolder);
    end

    % 取得該類別資料夾中的所有影像文件（這裡假設影像是 .jpg 格式）
    files = dir(fullfile(categoryFolder, '*.png')); 

    for j = 1:length(files)
        % 讀取影像
        imgPath = fullfile(categoryFolder, files(j).name);
        img = imread(imgPath);
        % disp('讀取影像');

        % 確保影像是灰度圖
        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        % 3. 高斯濾波去噪和平滑
        smoothedImg = imgaussfilt(img, 2);
        % disp('去噪');

        % 4. 對比度增強
        enhancedImg = imadjust(smoothedImg);
        % disp('對比');

        % 5. 使用邊緣檢測自動選取心臟區域
        edges = edge(enhancedImg, 'Sobel');
        % disp('邊緣檢測');
        
        % 使用形態學操作進行邊緣增強
        se = strel('disk', 3); % 定義結構元素大小
        dilatedEdges = imdilate(edges, se); % 擴張邊緣
        
        % 6. 自動選取心臟區域（假設心臟位於圖像的中心區域）
        [rows, cols] = size(dilatedEdges);
        centerRegion = dilatedEdges(round(rows/4):round(3*rows/4), round(cols/4):round(3*cols/4));
        % disp('選心臟');
        
        % 獲取心臟區域的邊界
        stats = regionprops(centerRegion, 'BoundingBox', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Area');

        % 初始化變數
        heartBBox = [0 0 0 0];
        heartMajorAxis = 0;
        heartMinorAxis = 0;
        heartOrientation = 0;
        enhancedHeartRegion = enhancedImg;

        if ~isempty(stats)
            % 取最大面積的區域作為心臟區域
            [~, idx] = max([stats.Area]);
            heartBBox = stats(idx).BoundingBox;

            disp(['Heart BBox coordinates: ', num2str(heartBBox)]);

            % 檢查 BoundingBox 的值是否合理
            if ~any(isnan(heartBBox)) && ~any(heartBBox < 0)
                % 根據選定的邊界區域進行裁剪
                heartRegion = imcrop(enhancedImg, heartBBox);
                enhancedHeartRegion = imadjust(heartRegion); % 增強心臟區域對比度
                
                % 取得心臟長軸與短軸
                heartMajorAxis = stats(idx).MajorAxisLength;  % 長軸
                heartMinorAxis = stats(idx).MinorAxisLength;  % 短軸
                heartOrientation = stats(idx).Orientation; % 心臟方向
            else
                disp('Warning: Invalid BoundingBox detected');
            end
        else
            disp('Warning: No heart region detected');
        end

        % 7. 計算第四胸椎股的位置並選取其區域
        vertebraStart = max(1, round(heartBBox(2) - 100)); % 確保起始位置不小於1
        vertebraEnd = min(size(enhancedImg, 1), round(heartBBox(2))); % 確保結束位置不超過圖片高度
        vertebraRegion = enhancedImg(vertebraStart:vertebraEnd, :);

        % 8. 計算心臟與第四胸椎股的長度比例
        if vertebraLength > 0
            majorAxisRatio = heartMajorAxis / vertebraLength;  % 長軸與胸椎股的比例
            minorAxisRatio = heartMinorAxis / vertebraLength;  % 短軸與胸椎股的比例
        else
            majorAxisRatio = NaN;
            minorAxisRatio = NaN;
        end

        % 顯示比例結果
        disp(['Image: ', files(j).name]);
        disp(['Heart to Vertebra (Major Axis) ratio: ', num2str(majorAxisRatio)]);
        disp(['Heart to Vertebra (Minor Axis) ratio: ', num2str(minorAxisRatio)]);

        % 9. 將增強後的心臟區域放回原圖
        % 使用 imresize 將增強的心臟區域大小調整為與原圖中心臟區域大小一致
        resizedEnhancedHeartRegion = imresize(enhancedHeartRegion, [round(heartBBox(4)), round(heartBBox(3))]);
        enhancedImg(round(heartBBox(2):(heartBBox(2) + heartBBox(4) - 1)), ...
                    round(heartBBox(1):(heartBBox(1) + heartBBox(3) - 1))) = resizedEnhancedHeartRegion;

        % 10. 使用自適應直方圖均衡（CLAHE）進一步增強對比度
        claheImg = adapthisteq(enhancedImg, 'ClipLimit', 0.02);

        % 11. 使用形態學操作增強心臟邊緣
        edges = edge(claheImg, 'Sobel');
        se = strel('disk', 2);
        dilatedEdges = imdilate(edges, se);

        % 12. 在處理後影像上顯示比例結果
        annotatedImg = insertText(claheImg, [10, 10], ...
            ['Major Axis Ratio: ', num2str(majorAxisRatio, '%.2f')], ...
            'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'black', 'BoxOpacity', 0.7);
        
        annotatedImg = insertText(annotatedImg, [10, 40], ...
            ['Minor Axis Ratio: ', num2str(minorAxisRatio, '%.2f')], ...
            'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'black', 'BoxOpacity', 0.7);

        % 13. 畫出心臟區域的邊界框
        annotatedImg = insertShape(annotatedImg, 'Rectangle', heartBBox, 'LineWidth', 3, 'Color', 'red');

        % 14. 畫出心臟的長軸與短軸
        heartCenter = [heartBBox(1) + heartBBox(3)/2, heartBBox(2) + heartBBox(4)/2]; % 心臟區域中心點
        majorAxisEnd = heartCenter + [cosd(heartOrientation) * heartMajorAxis / 2, sind(heartOrientation) * heartMajorAxis / 2];
        minorAxisEnd = heartCenter + [cosd(heartOrientation + 90) * heartMinorAxis / 2, sind(heartOrientation + 90) * heartMinorAxis / 2];
        
        annotatedImg = insertShape(annotatedImg, 'Line', [heartCenter(1), heartCenter(2), majorAxisEnd(1), majorAxisEnd(2)], 'LineWidth', 3, 'Color', 'green');
        annotatedImg = insertShape(annotatedImg, 'Line', [heartCenter(1), heartCenter(2), minorAxisEnd(1), minorAxisEnd(2)], 'LineWidth', 3, 'Color', 'blue');

        % 15. 在影像上顯示長軸與短軸的長度數字
        annotatedImg = insertText(annotatedImg, majorAxisEnd + [10, -10], ...
            ['Major Axis: ', num2str(heartMajorAxis, '%.2f')], 'FontSize', 18, 'TextColor', 'green', 'BoxOpacity', 0);
        
        annotatedImg = insertText(annotatedImg, minorAxisEnd + [10, -10], ...
            ['Minor Axis: ', num2str(heartMinorAxis, '%.2f')], 'FontSize', 18, 'TextColor', 'blue', 'BoxOpacity', 0);

        % 16. 畫出第四胸椎股的邊界框（這裡假設位置）
        vertebraBBox = [heartBBox(1), heartBBox(2) + heartBBox(4), heartBBox(3), 100]; % 假設第四胸椎股區域
        annotatedImg = insertShape(annotatedImg, 'Rectangle', vertebraBBox, 'LineWidth', 3, 'Color', 'blue');

        % 17. 保存處理後的影像
        outputImgPath = fullfile(outputCategoryFolder, files(j).name);
        imwrite(annotatedImg, outputImgPath); % 保存增強的影像

        % 18. 保存邊緣增強影像
        edgeOutputPath = fullfile(outputCategoryFolder, ['edges_' files(j).name]);
        imwrite(dilatedEdges, edgeOutputPath); % 保存邊緣增強影像
    end
end

disp('所有影像處理完成');
