
function leaf_base_point_searching

%% Read point cloud data from h5py file
file_path = '../../dataset/inner_point_removal/Pos_Norm_Color_Sem Ins_Test DS (45 pcs)/training_set.h5';
pc_file_list = importdata('../../dataset/train_file_list.txt');

% - Load point cloud data from *.h5 file
% pc_file_name = cell2mat(pc_file_list(1));
pc_file_name = 'Harvest_03_PotNr_407';
display(pc_file_name)
data = h5read(file_path, ['/', pc_file_name]);
data = data';
display(['Leaf number: ', num2str(max(data(:, end-1)) - 5)])

%% - Data format -
% N-by-50: [x, y, z, RGB channels for 15 cameras (3*15), instance labels,
% semantic labels]
%% ---------------

% - To remove the leaf close to the bottom of the plant that may have bad
% influence towards the TreeQSM performance
leaf_removal = [6];
if ~isempty(leaf_removal)
    for i = 1 : size(leaf_removal, 2)
        data(data(:, end-1) == leaf_removal(i), :) = [];
    end
end

% - Select the points belonging to the stemwork
pts_stemwork = data((data(:, end) == 2), 1:3);
% plot3(pts_stemwork(:, 1), pts_stemwork(:, 2), pts_stemwork(:, 3), '.b')
% axis equal

% - Filtering
pass = filtering(pts_stemwork, 0.05, 50, 0.1, 0.1, 15, true, true);  % Pass = filtering(P0, r1, n1, d2, r2, n2, Scaling, AllPoints)
pc_filtered = pts_stemwork(pass,:);

% - Apply TreeQSM to the stemwork point cloud
inputs = get_inputs;
qsm = treeqsm(pc_filtered, inputs);
pl_model = get_plant_model(qsm);

save pl_model pl_model

% visualisation(pl_model);

% column = 'AO';
% load('pl_model.mat')
% info_save_path = './training_set_leaf_info.xlsx';
% for i = 1 : pl_model.nStem_Internodes
%     display(eval(['pl_model.segments.segment', num2str(i), '.annotation']))
%     x = eval(['pl_model.segments.segment', num2str(i), '.terminal_pt(1)']);
%     y = eval(['pl_model.segments.segment', num2str(i), '.terminal_pt(2)']);
%     z = eval(['pl_model.segments.segment', num2str(i), '.terminal_pt(3)']);
%     display([x, y, z])
%     writematrix(x, info_save_path, 'Range', [column, num2str(i * 6 + 2)])
%     writematrix(y, info_save_path, 'Range', [column, num2str(i * 6 + 3)])
%     writematrix(z, info_save_path, 'Range', [column, num2str(i * 6 + 4)])
% end

% pts_stemwork = data((data(:, end - 1) == 6), 1:3);
% figure();
% plot3(pts_stemwork(:, 1), pts_stemwork(:, 2), pts_stemwork(:, 3), '.b')
% axis equal

% figure();
% hold on;
% plot3(data(:, 1), data(:, 2), data(:, 3), '.b')
% pts_stemwork = data((data(:, end-1) == 6), 1:3);
% plot3(pts_stemwork(:, 1), pts_stemwork(:, 2), pts_stemwork(:, 3), '.r')
% pts_stemwork = data((data(:, end-1) == 7), 1:3);
% plot3(pts_stemwork(:, 1), pts_stemwork(:, 2), pts_stemwork(:, 3), '.g')
% axis equal

return;


% - TreeQSM Inputs
function inputs = get_inputs

%% QSM reconstruction input parameters
% THE FIVE INPUT PARAMETERS TO BE OPTIMIZED.
% These CAN BE VARIED AND SHOULD BE OPTIMIZED 
% (they can have multiple values given as vectors, e.g. [0.01 0.02]).
% Patch size of the first uniform-size cover:
% Function for the first cover set: 1) To remove the points that don’t belong to the tree; 2) 
% To make initial segmentation that is used as a priori information for the branch connections 
% and size of the cover sets in the second cover set generation
inputs.PatchDiam1 = [0.02];
% Minimum patch size of the cover sets in the second cover:
inputs.PatchDiam2Min = [0.002];  % The minimum size is at the tips of the branches or the stem
% Maximum cover set size in the stem's base in the second cover:
inputs.PatchDiam2Max = [0.05];  % The maximum size is at the base of the stem

% ADDITIONAL PATCH GENERATION PARAMETERS.
% The following parameters CAN BE VARIED BUT CAN BE USUALLY KEPT AS SHOWN 
% (i.e. little bigger than PatchDiam parameters).
% Ball radius in the first uniform-size cover generation:
inputs.BallRad1 = inputs.PatchDiam1 + 0.005; 
% Maximum ball radius in the second cover generation:
inputs.BallRad2 = inputs.PatchDiam2Max + 0.005; 

% The following parameters CAN BE USUALLY KEPT FIXED as shown.
% Minimum number of points in BallRad1-balls, generally good value is 3:
inputs.nmin1 = 3; 
% Minimum number of points in BallRad2-balls, generally good value is 1:
inputs.nmin2 = 1; 
% Does the point cloud contain points only from the tree (if 1, then yes):
inputs.OnlyTree = 1; 
% Produce a triangulation of the stem's bottom part up to the first main
% branch (if 1, then yes):
inputs.Tria = 0; 
% Compute the point-model distances (if 1, then yes):
inputs.Dist = 1; 

% RADIUS CORRECTION OPTIONS FOR MODIFYING TOO LARGE AND TOO SMALL CYLINDERS.
% These parameters CAN BE USUALLY KEPT FIXED as shown.
% Traditional TreeQSM choices:
% Minimum cylinder radius, used particularly in the taper corrections:
inputs.MinCylRad = 0.001; 
% Radius correction based on radius of the parent. If 1, radii in a branch 
% are always smaller than the radius of the parent in the parent branch:
inputs.ParentCor = 1; 
% Taper correction of radii inside branches. If 1, use partially linear 
% (stem) and parabola (branches) taper corrections:seg_start_rad1ius
inputs.TaperCor = 0; 

% Growth volume correction approach introduced by Jan Hackenberg, 
% allometry: GrowthVol = a*Radius^b+c
inputs.GrowthVolCor = 0; % If 1, use growth volume (GV) correction 
% fac-parameter of the GV-approach, defines upper and lower bound. When 
% using GV-approach, consider setting TaperCorr = 0, ParentCorr = 0, 
% MinCylinderRadius = 0.
inputs.GrowthVolFac = 2.5; 

%% Other inputs
% These parameters don't affect the QSM-reconstruction but define what is
% saved, plotted, and displayed and how the models are named/indexed
% Name string for saving output files and naming models:
inputs.name = 'tree'; 
% Tree index. If modelling multiple trees, then they can be indexed uniquely:
inputs.tree = 1;
% Model index, can separate models if multiple models with the same inputs:
inputs.model = 1; 
% Save the output struct QSM as a matlab-file into \result folder. 
% If name = 'pine', tree = 2, model = 5, the name of the saved file is 
% 'QSM_pine_t2_m5.mat':
inputs.savemat = 1; 
% Save the models in .txt-files (check "save_model_text.m"):
inputs.savetxt = 0; 
% What are plotted during reconstruction process: 
% 2 = plots the QSM, the segmentated point cloud and distributions, 
% 1 = plots the QSM and the segmentated point cloud
% 0 = plots nothing
inputs.plot = 1; 
% What are displayed during the reconstruction: 2 = display all; 
% 1 = display name, parameters and distances; 0 = display only the name:
inputs.disp = 1;

return;


% - Get Digital Plant Model from TreeQSM Result
function pl_model = get_plant_model(qsm)

shoot_id = 1;  % Shoot id is set to 1 at current stage.

pl_model = struct;

nSegment = 0;

% Stem segments
Stem_Length = 0;
branch_order = 0;  % Branch order of zero refers to the stem.
stem_id = 0;
cy_id_of_nodes_on_stem_record = [];
id_cylinders_in_chain = find(qsm.cylinder.BranchOrder == branch_order);
start_pt = qsm.cylinder.start(id_cylinders_in_chain(1), :);
seg_start_radius = qsm.cylinder.radius(id_cylinders_in_chain(1));
for id_cylinder = min(id_cylinders_in_chain) : max(id_cylinders_in_chain)
    id_1 = find(qsm.cylinder.parent == id_cylinder);
    if numel(id_1) > 1
        cy_id_of_nodes_on_stem_record = [cy_id_of_nodes_on_stem_record, id_cylinder];
        id_2 = find(id_1 ~= id_cylinder + 1);
        terminal_pt = qsm.cylinder.start(id_1(id_2), :);
        nSegment = nSegment + 1;
        % Attributes of plant segment
        segment_type = 'stem';
        stem_id = stem_id + 1;
        annotation = annotation_sys(shoot_id, segment_type, stem_id);
        seg_terminal_radius = qsm.cylinder.radius(id_cylinder);
        radius = [seg_start_radius, seg_terminal_radius];
        length = sqrt(sum((terminal_pt - start_pt) .^ 2));
        orientation = (terminal_pt - start_pt) / length;
        eval(['pl_model.segments.segment', num2str(nSegment), '.annotation = annotation;']);
        eval(['pl_model.segments.segment', num2str(nSegment), '.start_pt = start_pt;']);
        eval(['pl_model.segments.segment', num2str(nSegment), '.terminal_pt = terminal_pt;']);
        eval(['pl_model.segments.segment', num2str(nSegment), '.radius = radius;']);
        eval(['pl_model.segments.segment', num2str(nSegment), '.length = length;']);
        eval(['pl_model.segments.segment', num2str(nSegment), '.orientation = orientation;']);
        eval(['pl_model.segments.segment', num2str(nSegment), '.is_branch_end = false;']);
        start_pt = terminal_pt;  % Update the start position for the next segment.
        seg_start_radius = seg_terminal_radius;  % Update the start radius of the next segment.
        Stem_Length = Stem_Length + length;
    end
end

nStem_Internodes = nSegment;

% Petiole and rachis segments
branch_order = 1;  % Branch order 0 refers to trunk, branch order 1 refers to first order branches (petioles and rachis).
id_branch_selected = find(qsm.branch.order == branch_order);
cy_id_of_nodes_on_secondary_record = [];
petiole_orientations = [];
branching_angle = [];
azimuth_angle = [];
nDetected_Leaves = numel(id_branch_selected);
for i = 1 : nDetected_Leaves
    id_branch = id_branch_selected(i);
    id_cylinders_in_chain = find(qsm.cylinder.branch == id_branch);  % Select all cylinders within selected secondary branch
    isPetiole_Exist = 0;
    rachis_id = 0;
    start_pt = qsm.cylinder.start(id_cylinders_in_chain(1), :);
    seg_start_radius = qsm.cylinder.radius(id_cylinders_in_chain(1));
    for j = 1 : numel(id_cylinders_in_chain)
        id_cylinder = id_cylinders_in_chain(j);
        if j ~= numel(id_cylinders_in_chain)
            id_1 = find(qsm.cylinder.parent == id_cylinder);
            if numel(id_1) > 1
                id_2 = find(id_1 ~= id_cylinder + 1);
                for k = 1 : numel(id_2)  % Sometimes, a fitted cylinder segment may contain start points of two or more petiolules, which means that the cylinder segment might be part of two connected rachis/petiole.
                    cy_id_of_nodes_on_secondary_record = [cy_id_of_nodes_on_secondary_record, id_cylinder];
                    terminal_pt = qsm.cylinder.start(id_1(id_2(k)), :);
                    nSegment = nSegment + 1;
                    if ~isPetiole_Exist
                        segment_type = 'petiole';
                        parent_stem_cy = qsm.cylinder.parent(id_cylinders_in_chain(1));
                        stem_id = find(cy_id_of_nodes_on_stem_record == parent_stem_cy);
                        annotation = annotation_sys(shoot_id, segment_type, stem_id);
                        branching_angle = [branching_angle, qsm.branch.angle(i + 1)];  % The first branch is considered as the stem (trunk), and there is usually one stem only in a plant. Leaves start with the second branch.
                        azimuth_angle = [azimuth_angle, qsm.branch.azimuth(i + 1)];
                        isPetiole_Exist = 1;
                    else
                        segment_type = 'rachis';
                        parent_stem_cy = qsm.cylinder.parent(id_cylinders_in_chain(1));
                        stem_id = find(cy_id_of_nodes_on_stem_record == parent_stem_cy);
                        rachis_id = rachis_id + 1;
                        annotation = annotation_sys(shoot_id, segment_type, stem_id, rachis_id);
                    end
                    seg_terminal_radius = qsm.cylinder.radius(id_cylinder);
                    radius = [seg_start_radius, seg_terminal_radius];
                    length = sqrt(sum((terminal_pt - start_pt) .^ 2));
                    orientation = (terminal_pt - start_pt) / length;
                    if strcmp(annotation(1:4), 'PTL*')
                        petiole_orientations = [petiole_orientations; orientation];
                    end
                    eval(['pl_model.segments.segment', num2str(nSegment), '.annotation = annotation;']);
                    eval(['pl_model.segments.segment', num2str(nSegment), '.start_pt = start_pt;']);
                    eval(['pl_model.segments.segment', num2str(nSegment), '.terminal_pt = terminal_pt;']);
                    eval(['pl_model.segments.segment', num2str(nSegment), '.radius = radius;']);
                    eval(['pl_model.segments.segment', num2str(nSegment), '.length = length;']);
                    eval(['pl_model.segments.segment', num2str(nSegment), '.orientation = orientation;']);
                    eval(['pl_model.segments.segment', num2str(nSegment), '.is_branch_end = false;']);
                    start_pt = terminal_pt;  % Update the start position for the next segment.
                    seg_start_radius = seg_terminal_radius;  % Update the start radius of the next segment.
                end
            end
        else
            terminal_pt = qsm.cylinder.start(id_cylinders_in_chain(j), :) + ...
                qsm.cylinder.length(id_cylinders_in_chain(j)) * qsm.cylinder.axis(id_cylinders_in_chain(j), :);
            nSegment = nSegment + 1;
            if ~isPetiole_Exist
                segment_type = 'petiole';
                parent_stem_cy = qsm.cylinder.parent(id_cylinders_in_chain(1));
                stem_id = find(cy_id_of_nodes_on_stem_record == parent_stem_cy);
                annotation = annotation_sys(shoot_id, segment_type, stem_id);
                branching_angle = [branching_angle, qsm.branch.angle(i + 1)];  % The first branch is considered as the stem (trunk), and there is usually one stem only in a plant. Leaves start with the second branch.
                azimuth_angle = [azimuth_angle, qsm.branch.azimuth(i + 1)];
                isPetiole_Exist = 1;
            else
                segment_type = 'rachis';
                parent_stem_cy = qsm.cylinder.parent(id_cylinders_in_chain(1));
                stem_id = find(cy_id_of_nodes_on_stem_record == parent_stem_cy);
                rachis_id = rachis_id + 1;
                annotation = annotation_sys(shoot_id, segment_type, stem_id, rachis_id);
            end
            seg_terminal_radius = qsm.cylinder.radius(id_cylinder);
            radius = [seg_start_radius, seg_terminal_radius];
            length = sqrt(sum((terminal_pt - start_pt) .^ 2));
            orientation = (terminal_pt - start_pt) / length;
            if strcmp(annotation(1:4), 'PTL*')
                petiole_orientations = [petiole_orientations; orientation];
            end
            eval(['pl_model.segments.segment', num2str(nSegment), '.annotation = annotation;']);
            eval(['pl_model.segments.segment', num2str(nSegment), '.start_pt = start_pt;']);
            eval(['pl_model.segments.segment', num2str(nSegment), '.terminal_pt = terminal_pt;']);
            eval(['pl_model.segments.segment', num2str(nSegment), '.radius = radius;']);
            eval(['pl_model.segments.segment', num2str(nSegment), '.length = length;']);
            eval(['pl_model.segments.segment', num2str(nSegment), '.orientation = orientation;']);
            eval(['pl_model.segments.segment', num2str(nSegment), '.is_branch_end = false;']);
            start_pt = terminal_pt;  % Update the start position for the next segment.
            seg_start_radius = seg_terminal_radius;  % Update the start radius of the next segment.
        end
    end
end

% Phyllotactic angle
phyllotactic_angle = [];
for i = 1 : numel(azimuth_angle) - 1
    last_leaf_azimuth_angle = azimuth_angle(i);
    current_leaf_azimuth_angle = azimuth_angle(i + 1);
    if last_leaf_azimuth_angle < 0 & current_leaf_azimuth_angle > 0
        current_phyllotactic_angle = abs(last_leaf_azimuth_angle) + current_leaf_azimuth_angle;
    elseif last_leaf_azimuth_angle < 0 & current_leaf_azimuth_angle < 0
        if abs(last_leaf_azimuth_angle) < abs(current_leaf_azimuth_angle)
            current_phyllotactic_angle = 360 - abs(last_leaf_azimuth_angle - current_leaf_azimuth_angle);
        else
            current_phyllotactic_angle = abs(last_leaf_azimuth_angle - current_leaf_azimuth_angle);
        end
    elseif last_leaf_azimuth_angle > 0 & current_leaf_azimuth_angle < 0
        current_phyllotactic_angle = 360 - (last_leaf_azimuth_angle + abs(current_leaf_azimuth_angle));
    else
        if last_leaf_azimuth_angle < current_leaf_azimuth_angle
            current_phyllotactic_angle = current_leaf_azimuth_angle - last_leaf_azimuth_angle;
        else
            current_phyllotactic_angle = 360 - abs(last_leaf_azimuth_angle - current_leaf_azimuth_angle);
        end
    end
    phyllotactic_angle = [phyllotactic_angle, current_phyllotactic_angle];
end

pl_model.nStem_Internodes = nStem_Internodes;
pl_model.Stem_Length = Stem_Length;
pl_model.nDetected_Leaves = nDetected_Leaves;
pl_model.Branching_Angle = branching_angle;
pl_model.Azimuth_Angle = phyllotactic_angle;

return;


% - Set Annotations for Individual Plant Segments
function annotation = annotation_sys(shoot_id, segment_type, varargin)

if strcmp(segment_type, 'stem')
    stem_id = varargin{1};
    annotation = ['ST', num2str(stem_id), '*S', num2str(shoot_id)];
elseif strcmp(segment_type, 'petiole')
    stem_id = varargin{1};
    annotation = ['PTL*ST', num2str(stem_id), '*S', num2str(shoot_id)];
elseif strcmp(segment_type, 'rachis')
    stem_id = varargin{1};
    rachis_id = varargin{2};  % Start from 1 after petiole.
    annotation = ['R', num2str(rachis_id), '*ST', num2str(stem_id), '*S', num2str(shoot_id)];
elseif strcmp(segment_type, 'petiolule')
%     We temporarily ignore petiolule at current stage.
else
    stem_id = varargin{1};
    petiole_exist = varargin{2};
    if petiole_exist
        annotation = ['L*PTL*ST', num2str(stem_id), '*S', num2str(shoot_id)];
    else
        rachis_id = varargin{3};
        annotation = ['L*R', num2str(rachis_id), '*ST', num2str(stem_id), '*S', num2str(shoot_id)];
    end
end

return;


% - Visualisation of the Digital Plant Model
function visualisation(pl_model)

set(0, 'defaultfigurecolor', 'w');

figure;
hold on;
axis equal;
axis off;
grid off;

% Color maps
colors = [5 79 250; 0 128 255; 1 172 254; 17 216 238; 2 253 247] / 256;
cn = 256;
cm = size(colors, 1);
ct0 = linspace(0, 1, cm)';
ct = linspace(0, 1, cn)';
cr = interp1(ct0, colors(:, 1), ct);
cg = interp1(ct0, colors(:, 2), ct);
cb = interp1(ct0, colors(:, 3), ct);
cmap_stem = [cr,cg,cb];

colors = [254 134 1; 248 146 7; 243 163 12; 236 187 19; 233 217 22] / 256;
cn = 256;
cm = size(colors, 1);
ct0 = linspace(0, 1, cm)';
ct = linspace(0, 1, cn)';
cr = interp1(ct0, colors(:, 1), ct);
cg = interp1(ct0, colors(:, 2), ct);
cb = interp1(ct0, colors(:, 3), ct);
cmap_petiole = [cr,cg,cb];

colors = [48 207 72; 62 222 33; 99 242 13; 152 248 7; 198 255 0] / 256;
cn = 256;
cm = size(colors, 1);
ct0 = linspace(0, 1, cm)';
ct = linspace(0, 1, cn)';
cr = interp1(ct0, colors(:, 1), ct);
cg = interp1(ct0, colors(:, 2), ct);
cb = interp1(ct0, colors(:, 3), ct);
cmap_rachis = [cr,cg,cb];

[cy_x, cy_y, cy_z] = cylinder;
[Sp_x, Sp_y, Sp_z] = sphere;

field_names = fieldnames(pl_model.segments);
cylinder_num = numel(field_names);
Node_Num = 0; 
for cylinder_id = 1 : cylinder_num
    radius = eval(['pl_model.segments.segment', num2str(cylinder_id), '.radius;']);
    start_pt = eval(['pl_model.segments.segment', num2str(cylinder_id), '.start_pt;']);
    terminal_pt = eval(['pl_model.segments.segment', num2str(cylinder_id), '.terminal_pt;']);

    x = [cy_x(1, :) * radius(1) + start_pt(1); cy_x(2, :) * radius(2) + terminal_pt(1)];
    y = [cy_y(1, :) * radius(1) + start_pt(2); cy_y(2, :) * radius(2) + terminal_pt(2)];
    z = [repmat(start_pt(3), 1, size(cy_x, 2)); repmat(terminal_pt(3), 1, size(cy_x, 2))];

    Myplot_Rachis(cylinder_id) = surf(x, y, z, cy_x);

    annotation = eval(['pl_model.segments.segment', num2str(cylinder_id), '.annotation;']);

    if annotation(1) == 'S'
        colormap(cmap_stem)
    elseif strcmp(annotation(1:4), 'PTLU')
%             We temporarily ignore petiolule at current stage.
    elseif strcmp(annotation(1:3), 'PTL')
        colormap(cmap_petiole)
    elseif annotation(1) == 'R'
        colormap(cmap_rachis)
    end
    freezeColors

%     % Print all nodes
%     is_branch_end = eval(['input_structure.segments.segment', num2str(cylinder_id), '.is_branch_end;']);
%     if ~is_branch_end
%         Node_Num = Node_Num + 1;
%         Radius = 0.015;
%         Myplot_Nodes(Node_Num) = mesh(Radius * Sp_x + x(2), Radius * Sp_y + y(2), Radius * Sp_z + z(2), Sp_x);
%         colormap([0, 0, 0])
% %             camlight
%         freezeColors
%     end

end

% Print stem nodes only
for i = 1 : pl_model.nStem_Internodes
    Radius = 0.015;
    x_stem_node = eval(['pl_model.segments.segment', num2str(i), '.terminal_pt(1)']);
    y_stem_node = eval(['pl_model.segments.segment', num2str(i), '.terminal_pt(2)']);
    z_stem_node = eval(['pl_model.segments.segment', num2str(i), '.terminal_pt(3)']);
    mesh(Radius * Sp_x + x_stem_node, Radius * Sp_y + y_stem_node, Radius * Sp_z + z_stem_node, Sp_x);
    colormap([0, 0, 0])
    freezeColors
    display([x_stem_node, y_stem_node, z_stem_node]);
end

caxis auto
shading interp

return;