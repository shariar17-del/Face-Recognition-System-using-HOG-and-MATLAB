function ultimate_face_matcher_fixed_folder
    % Get the directory of the current script
    scriptPath = mfilename('fullpath');
    scriptDir = fileparts(scriptPath);
    fixedFolderPath = fullfile(scriptDir, 'GroupPhotos');

    % --- GUI Figure ---
    fig = uifigure('Name', 'Facial Recognition', 'Position', [100 100 1100 750]);

    % --- UI Component Positions and Sizes ---
    buttonWidth = 220;
    buttonHeight = 50;
    startX = 30;
    startYTop = 680;
    spacingY = 15;
    axesWidthRef = 350;
    axesHeight = 450;
    axesStartXResult = startX + axesWidthRef + 40; % Add spacing between axes
    axesStartY = 200;
    statusLabelHeight = 30;
    statusLabelStartY = axesStartY - spacingY - statusLabelHeight;
    folderLabelHeight = 20;
    folderLabelStartY = statusLabelStartY - spacingY - folderLabelHeight;
    infoPanelWidth = 650;
    infoPanelHeight = 120;
    infoPanelStartY = 20;
    infoPanelStartX = axesStartXResult;
    infoLabelWidth = 120;
    infoLabelHeight = 30;
    infoLabelSpacingX = 130;
    infoLabelStartYPanel = 40;

    setappdata(fig, 'groupPhotosFolder', fixedFolderPath); % Set fixed folder

    % --- UI Components ---
    loadRefButton = uibutton(fig, 'Text', '1. Load Reference Face', ...
        'Position', [startX startYTop buttonWidth buttonHeight], ...
        'ButtonPushedFcn', @(btn,event) loadReferenceFace(fig,btn), ...
        'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0.8 0.9 1]);

    % Display areas
    refAxes = uiaxes(fig, 'Tag', 'refAxes', 'Position', [startX axesStartY axesWidthRef axesHeight]);
    resultAxes = uiaxes(fig, 'Tag', 'resultAxes', 'Position', [axesStartXResult axesStartY infoPanelWidth axesHeight]); % Use infoPanelWidth for consistent width
    statusLabel = uilabel(fig, 'Position', [startX statusLabelStartY 1000 statusLabelHeight], 'Tag', 'statusLabel', ...
        'FontSize', 12, 'FontWeight', 'bold');
    folderLabel = uilabel(fig, 'Position', [startX folderLabelStartY 370 folderLabelHeight], 'Tag', 'folderLabel', ...
        'Text', sprintf('Searching in folder: %s', getappdata(fig, 'groupPhotosFolder')), ...
        'FontSize', 10);

    % Panel for personal info
    infoPanel = uipanel(fig, 'Title', 'Matched Person Info', ...
        'FontSize', 12, 'Position', [infoPanelStartX infoPanelStartY infoPanelWidth infoPanelHeight]);
    labels = {'Name', 'Student_ID', 'Department', 'Batch', 'Contact'};
    for k = 1:length(labels)
        uilabel(infoPanel, 'Text', [labels{k} ':'], ...
            'Position', [10 + (k-1)*infoLabelSpacingX, infoLabelStartYPanel, infoLabelWidth, infoLabelHeight], ...
            'FontSize', 11, 'Tag', ['info' labels{k}]);
    end
    % Load face detectors
    try
        detector1 = vision.CascadeObjectDetector('MergeThreshold', 8, 'MinSize', [60 60]);
        detector2 = vision.CascadeObjectDetector('ClassificationModel', 'ProfileFace', 'MergeThreshold', 10);
        setappdata(fig, 'detector1', detector1);
        setappdata(fig, 'detector2', detector2);
        setappdata(fig, 'matchThreshold', 0.4);
    catch ME
        errordlg(sprintf('Initialization error: %s', ME.message));
        return;
    end
end

function loadReferenceFace(fig, btn)
    btn.Enable = 'off';
    drawnow;
    try
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.jpeg'}, 'Select Reference Face');
        if isequal(file, 0)
            btn.Enable = 'on'; return;
        end
        refImg = imread(fullfile(path, file));
        detector1 = getappdata(fig, 'detector1');
        detector2 = getappdata(fig, 'detector2');
        bboxes = step(detector1, refImg);
        if isempty(bboxes), bboxes = step(detector2, refImg); end
        if isempty(bboxes)
            enhancedImg = imadjust(rgb2gray(refImg));
            bboxes = step(detector1, enhancedImg);
            if isempty(bboxes), bboxes = step(detector2, enhancedImg); end
        end
        if isempty(bboxes)
            uialert(fig, 'No face detected in reference image', 'Detection Failed');
            btn.Enable = 'on'; return;
        end
        [~, idx] = max(bboxes(:,3).*bboxes(:,4));
        mainFace = bboxes(idx,:);
        processedFace = preprocessFaceUltimate(refImg, mainFace);
        hogFeatures = extractHOGFeatures(processedFace, 'CellSize', [6 6]);
        lbpFeatures = extractLBPFeatures(processedFace, 'NumNeighbors', 8, 'Radius', 2);
        setappdata(fig, 'refHOG', hogFeatures);
        setappdata(fig, 'refLBP', lbpFeatures);
        setappdata(fig, 'refImg', refImg);
        setappdata(fig, 'refBox', mainFace);
        ax = findobj(fig, 'Tag', 'refAxes');
        imshow(refImg, 'Parent', ax);
        rectangle('Position', mainFace, 'EdgeColor', 'g', 'LineWidth', 3, 'Parent', ax);
        title(ax, 'Reference Face', 'FontSize', 10);
        status = findobj(fig, 'Tag', 'statusLabel');
        status.Text = '✅ Reference face loaded successfully. Starting search...';
        drawnow; % Update status immediately
        findMatchInFixedFolder(fig); % Automatically start matching
    catch ME
        uialert(fig, sprintf('Error: %s', ME.message), 'Loading Failed');
    end
    btn.Enable = 'on';
end

function findMatchInFixedFolder(fig)
    try
        if ~isappdata(fig, 'refHOG')
            uialert(fig, 'Load a reference face first', 'Error');
            return;
        end
        folderPath = getappdata(fig, 'groupPhotosFolder');
        extensions = {'*.jpg','*.jpeg','*.png','*.bmp'};
        imageFiles = []; for ext = extensions
            imageFiles = [imageFiles; dir(fullfile(folderPath, ext{1}))];
        end
        if isempty(imageFiles)
            uialert(fig, sprintf('No images found in the folder: %s', folderPath), 'Error');
            return;
        end
        bestMatch = struct('score', Inf, 'file', '', 'img', [], 'box', []);
        detector1 = getappdata(fig, 'detector1');
        detector2 = getappdata(fig, 'detector2');
        matchThreshold = getappdata(fig, 'matchThreshold');
        wb = uiprogressdlg(fig, 'Title', 'Searching...', 'Message', 'Initializing', 'Indeterminate', 'on');
        for i = 1:length(imageFiles)
            wb.Value = i/length(imageFiles);
            wb.Message = sprintf('Processing %s', imageFiles(i).name);
            try
                img = imread(fullfile(folderPath, imageFiles(i).name));
                bboxes = step(detector1, img);
                if isempty(bboxes), bboxes = step(detector2, img); end
                if isempty(bboxes)
                    enhancedImg = imadjust(rgb2gray(img));
                    bboxes = step(detector1, enhancedImg);
                    if isempty(bboxes), bboxes = step(detector2, enhancedImg); end
                end
                for j = 1:size(bboxes,1)
                    face = preprocessFaceUltimate(img, bboxes(j,:));
                    hogFeatures = extractHOGFeatures(face, 'CellSize', [6 6]);
                    lbpFeatures = extractLBPFeatures(face, 'NumNeighbors', 8, 'Radius', 2);
                    refHOG = getappdata(fig, 'refHOG');
                    refLBP = getappdata(fig, 'refLBP');
                    combinedScore = 0.6 * norm(refHOG - hogFeatures) + 0.4 * norm(refLBP - lbpFeatures);
                    if combinedScore < bestMatch.score
                        bestMatch.score = combinedScore;
                        bestMatch.file = imageFiles(i).name;
                        bestMatch.img = img;
                        bestMatch.box = bboxes(j,:);
                    end
                end
            catch; continue; end
        end
        close(wb);
        ax = findobj(fig, 'Tag', 'resultAxes');
        status = findobj(fig, 'Tag', 'statusLabel');
        cla(ax);
        if isinf(bestMatch.score)
            imshow(zeros(300,300,3,'uint8'), 'Parent', ax);
            status.Text = sprintf('❌ No faces detected in any images in: %s', folderPath);
        elseif bestMatch.score < matchThreshold
            imshow(bestMatch.img, 'Parent', ax);
            rectangle('Position', bestMatch.box, 'EdgeColor', 'g', 'LineWidth', 3, 'Parent', ax);
            status.Text = sprintf('✅ Strong match found! (Score: %.3f in %s)', bestMatch.score, bestMatch.file);
            showInfo(fig, bestMatch.file);
        else
            imshow(bestMatch.img, 'Parent', ax);
            rectangle('Position', bestMatch.box, 'EdgeColor', 'y', 'LineWidth', 3, 'Parent', ax);
            status.Text = sprintf('⚠️ Potential match (Score: %.3f in %s)', bestMatch.score, bestMatch.file);
            showInfo(fig, bestMatch.file);
        end
    catch ME
        uialert(fig, sprintf('Error: %s', ME.message), 'Matching Failed');
    end
end

function showInfo(fig, filename)
    try
        [~, idNo, ~] = fileparts(filename);
        opts = detectImportOptions('Book1.xlsx');
        opts.SelectedVariableNames = {'ID','Name','Student_ID','Department','Batch','Contact'};
        data = readtable('Book1.xlsx', opts);
        matchRow = find(strcmp(string(data.ID), idNo));
        if isempty(matchRow), return; end
        info = data(matchRow, :);
        fields = {'Name', 'Student_ID', 'Department', 'Batch', 'Contact'};
        for i = 1:length(fields)
            label = findobj(fig, 'Tag', ['info' fields{i}]);
            label.Text = sprintf('%s: %s', fields{i}, string(info.(fields{i})));
        end
    catch
        % silently fail if Excel not found or ID not matched
    end
end

function processed = preprocessFaceUltimate(img, bbox)
    try
        margin = 0.2;
        x = max(1, floor(bbox(1)-margin*bbox(3)));
        y = max(1, floor(bbox(2)-margin*bbox(4)));
        w = min(size(img,2)-x, floor(bbox(3)*(1+2*margin)));
        h = min(size(img,1)-y, floor(bbox(4)*(1+2*margin)));
        face = imcrop(img, [x y w h]);
        if size(face,3) == 3
            face = rgb2gray(face);
        end
        scale = min([200 200]./size(face));
        face = imresize(face, scale, 'Antialiasing', true);
        padSize = [200 200] - size(face);
        face = padarray(face, floor(padSize/2), 'symmetric', 'pre');
        face = padarray(face, ceil(padSize/2), 'symmetric', 'post');
        face = adapthisteq(face, 'ClipLimit', 0.02);
        face = imgaussfilt(face, 1.2);
        face = imadjust(face);
        processed = face;
    catch
        processed = [];
    end
end