
input_folder = 'results/RRDB_VGG128_DIV2K_400k/visualization';
dataset = 'Set14'
source_folder = strcat(input_folder, '/', dataset);
folder = dir(strcat(source_folder, '/', '*.png'));
piqe_all = 0;
niqe_all = 0;
brisque_all = 0;
len = length(folder);
for i = 1:(len)
    filename = folder(i).name;
    if isequal(filename, '.')
    elseif isequal(filename, '..')
    else
        new_filename = strcat(source_folder, '/', filename);
        file = imread(new_filename);
        piqe_score = piqe(file);
        piqe_all = piqe_all + piqe_score;
        brisque_score = brisque(file);
        brisque_all = brisque_all + brisque_score;
        niqe_score = niqe(file);
        niqe_all = niqe_all + niqe_score;
    end
end

piqe_avg = piqe_all / len
niqe_avg = niqe_all / len
brisque_avg = brisque_all / len
        