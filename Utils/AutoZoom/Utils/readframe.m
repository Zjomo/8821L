clc;
clear all;
close all;
VideoAd = VideoReader('E:\USST\博士\实验数据\平面纳米片驱动\平面旋转实验\石英衬底\飞秒加工\Au\应用实验\金属旋转测量二次谐波\260513\20260514_161952.mp4');%输入视频位置
numFrames = VideoAd.NumberOfFrames;% 帧的总数
videoF=VideoAd.FrameRate;%FrameRate 视频采集速率
videoD=VideoAd.Duration;  %Duration  时间
numname=6;%the length of image name
nz = strcat('%0',num2str(numname),'d');
T=20*videoF;%提取帧数间隔，这里设定每3秒提取1帧 
i=439;
 for k = 1 :T: numFrames   %     
     numframe = read(VideoAd,k);%读取第几帧
     num=sprintf(nz,i);   %i为保存图片的序号
     i=i+1;
     imwrite(numframe,strcat('E:\USST\博士\实验数据\平面纳米片驱动\平面旋转实验\石英衬底\飞秒加工\Au\应用实验\金属旋转测量二次谐波\260513\素材库\',num,'.png'),'png');  
     % 保存帧,
     %位置：F:\USST\博士\论文\旋转马达\实验数据\图1数据\第一周视频\测试取帧\
 end
