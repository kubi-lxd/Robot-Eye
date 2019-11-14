function result=MarkAllFigures(folder,datapath)
%Mark all .bmp figures in the folder
%Input: str  figure folder path
%Input：str  data folder path
%This fuction will create a txt file which records mark point information of all figures in the same folder
FileFolder=fullfile(folder);
dirOutput=dir(fullfile(FileFolder,'image*.bmp'));
filenames={dirOutput.name}';
infos=[];
init = ' ';
for i=1:length(filenames)
    disp([i,length(filenames)]);
    [info,init]=MarkOneFigure(fullfile(folder,filenames{i}),init);
    infos=[infos;info];
end
DataFloder = fullfile(datapath);
txtname = fullfile(DataFloder,'mark.txt');
dlmwrite(txtname, infos, 'delimiter','\t','newline','pc');
result = 1;
end
