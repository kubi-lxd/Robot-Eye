function [info,initresulut]=MarkOneFigure(filename,initinput)
% Continuously pop-up pictures for point marking
% Input: filename   Figure path to be marked
%        initinput  Last mark point information for quick marking

s = size(initinput);
if s(2) == 1 % initinput is empty, initial the initinput
    initinput1 = {'1200','3300','1'};
elseif s(1) == 1 && s(2)==3 % The last time was a single mark
    initinput1 = {num2str(initinput(1)),num2str(initinput(2)),num2str(initinput(3))};
elseif s(1) == 2 && s(2)==3 % The last time was a mark of two points
    initinput2 ={num2str(initinput(2,1)),num2str(initinput(2,2)),'2'};
    initinput1 = {num2str(initinput(1,1)),num2str(initinput(1,2)),num2str(initinput(1,3))};
else
    initinput1 = 0; % Error
end
% Calculate the robot's control variable information based on the image name
sp_name = strsplit(filename,'_');
alpha = str2double(sp_name{2});
beta = str2double(sp_name{3});
gama_sp = strsplit(sp_name{4},'.');
gama = str2double(gama_sp{1});
% Start marking
im = imread(filename);
f = figure(1); imshow(im);
hold on;
% Ask user input real world information
prompt = {'Enter RealX:','Enter RealY:','Point Num'};
[x,y] = ginput(1);
s = size(initinput);
answer = inputdlg(prompt, 'Input',1,initinput1);

if str2double(answer{3}) == 1
    % Mark only one point
	hold off;
    close(f);
    info = [str2double(answer{1}),str2double(answer{2}),x,y,alpha,beta,gama,str2double(answer{3})];
    initresulut = [str2double(answer{1}),str2double(answer{2}),str2double(answer{3})];
else % Mark two points
    [a,b] = ginput(1);
    if s(1)==1
        answer2 = inputdlg(prompt, 'Input',1,initinput1); 
    else
        answer2 = inputdlg(prompt, 'Input',1,initinput2); 
    end
    info = [str2double(answer{1}),str2double(answer{2}),x,y,alpha,beta,gama,str2double(answer{3});
        str2double(answer2{1}),str2double(answer2{2}),a,b,alpha,beta,gama,str2double(answer2{3})];
    initresulut = [str2double(answer{1}),str2double(answer{2}),str2double(answer{3});
        str2double(answer2{1}),str2double(answer2{2}),str2double(answer2{3})];
    hold off;
    close(f);
end


