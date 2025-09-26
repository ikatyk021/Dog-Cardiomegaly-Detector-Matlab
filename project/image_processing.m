% 設定資料夾路徑
inputFolder = '/Users/tingyuko/Downloads/Dog_heart_2/Train'; 
outputFolder = '/Users/tingyuko/Downloads/processed_test';
categories = {'Large', 'Normal', 'Small'};

% 確保輸出資料夾存在
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% 遍歷每一類別資料夾並處理所有影像
for i = 1:length(categories)
    categoryFolder = fullfile(inputFolder, categories{i});
    outputCategoryFolder = fullfile(outputFolder, categories{i});
    
    % 確保類別的資料夾存在
    if ~exist(outputCategoryFolder, 'dir')
        mkdir(outputCategoryFolder);
    end

    % 取得該類別資料夾中的所有影像文件
    files = dir(fullfile(categoryFolder, '*.png')); 

    for j = 1:length(files)
        % 讀取影像
        imgPath = fullfile(categoryFolder, files(j).name);
        img = imread(imgPath);

        % 確保影像為灰階
        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        % 檢測函數
        [heartCenter, vertebraCenter] = detectDogChestXray(img);
        
        % 如果檢測成功，進行標註
        if ~isempty(heartCenter) && ~isempty(vertebraCenter)
            % 創建註釋影像
            annotatedImg = repmat(img, [1 1 3]); % 將灰階圖像轉換為RGB
            
            % 在心臟中心點畫十字標記
            crossSize = 20;
            % 心臟標記（紅色）
            annotatedImg = insertShape(annotatedImg, 'Line', ...
                [heartCenter(1)-crossSize, heartCenter(2), heartCenter(1)+crossSize, heartCenter(2)], ...
                'LineWidth', 2, 'Color', 'red');
            annotatedImg = insertShape(annotatedImg, 'Line', ...
                [heartCenter(1), heartCenter(2)-crossSize, heartCenter(1), heartCenter(2)+crossSize], ...
                'LineWidth', 2, 'Color', 'red');
            
            % % 第四胸椎標記（藍色）
            % annotatedImg = insertShape(annotatedImg, 'Line', ...
            %     [vertebraCenter(1)-crossSize, vertebraCenter(2), vertebraCenter(1)+crossSize, vertebraCenter(2)], ...
            %     'LineWidth', 2, 'Color', 'blue');
            % annotatedImg = insertShape(annotatedImg, 'Line', ...
            %     [vertebraCenter(1), vertebraCenter(2)-crossSize, vertebraCenter(1), vertebraCenter(2)+crossSize], ...
            %     'LineWidth', 2, 'Color', 'blue');
            
            % 只保存標註後的影像
            outputPath = fullfile(outputCategoryFolder, [files(j).name(1:end-4) '_annotated.png']);
            imwrite(annotatedImg, outputPath);
            
            % 顯示處理資訊
            disp(['Processed: ' files(j).name]);
            disp(['已保存處理結果到: ' outputPath]);
        else
            warning(['Failed to detect heart or vertebra in: ' files(j).name]);
        end
    end
end

disp('所有影像處理完成');

% 檢測函數
function [heartCenter, vertebraCenter] = detectDogChestXray(img)
    % 1. 前處理
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % 2. 增強對比度
    enhancedImg = adapthisteq(img, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
    
    % 3. 找心臟中心點
    % 使用形態學操作增強心臟輪廓
    se_heart = strel('disk', 15);
    heartEnhanced = imclose(enhancedImg, se_heart);
    heartEnhanced = imopen(heartEnhanced, se_heart);
    
    % 使用區域生長法檢測心臟
    [rows, cols] = size(heartEnhanced);
    seedPoint = [round(rows*0.6), round(cols*0.5)]; % 根據側面X光片心臟通常位置設定種子點
    
    % 區域生長參數
    threshold = graythresh(heartEnhanced);
    heartMask = regiongrowing(heartEnhanced, seedPoint(1), seedPoint(2), threshold);
    
    % 計算心臟中心點
    stats = regionprops(heartMask, 'Centroid');
    if ~isempty(stats)
        heartCenter = stats(1).Centroid;
    else
        heartCenter = [];
    end
    
    % 4. 找第四胸椎中心點
    % 使用Top-hat變換突出胸椎
    se_vertebra = strel('line', 30, 0);
    vertebraEnhanced = imtophat(enhancedImg, se_vertebra);
    
    % 找出胸椎位置
    [rows, cols] = size(vertebraEnhanced);
    upperRegion = vertebraEnhanced(1:round(rows/2), :);
    [~, vertebraRow] = max(sum(upperRegion, 2));
    
    % 在胸椎行找最強的點作為中心
    if ~isempty(vertebraRow)
        rowProfile = vertebraEnhanced(vertebraRow, :);
        [~, vertebraCol] = max(rowProfile);
        vertebraCenter = [vertebraCol, vertebraRow];
    else
        vertebraCenter = [];
    end
end

% 區域生長函數
function mask = regiongrowing(img, x, y, threshold)
    [rows, cols] = size(img);
    mask = false(rows, cols);
    seed_value = img(x, y);
    
    % 初始化待處理點隊列
    queue = [x, y];
    mask(x, y) = true;
    
    while ~isempty(queue)
        current = queue(1, :);
        queue(1, :) = [];
        
        % 檢查8個相鄰像素
        for i = -1:1
            for j = -1:1
                if i == 0 && j == 0
                    continue;
                end
                
                newX = current(1) + i;
                newY = current(2) + j;
                
                % 檢查邊界
                if newX < 1 || newX > rows || newY < 1 || newY > cols
                    continue;
                end
                
                % 如果像素未被訪問且滿足閾值條件
                if ~mask(newX, newY) && abs(double(img(newX, newY)) - double(seed_value)) < threshold
                    mask(newX, newY) = true;
                    queue = [queue; newX, newY];
                end
            end
        end
    end
end