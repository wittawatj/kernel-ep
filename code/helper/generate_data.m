function [x,ytrue,n,d,c]=generate_data(dataset,trial)

%`case' = [34 40 130 150 80 113].

%1-6: artificial
%10-29: IDA
%30: USPS
%40: Olivetti faces
%50: Zelnik_data
%60: TechTC300
%70: Kato Chemo
%80: Hachiya ALKAN
%90: Yamada VOC
%100: Yamada freesound
%110: Yamada voice conversion
%120: 20 Newsgroups (100 words)
%130: 20 Newsgroups (10000 words)
%140: Kimura NLP-line
%150: Kimura NLP-interest

rand('state',trial);
randn('state',trial);

switch dataset
 case 1 % Two Gaussians
  n=200;
  d=2;
  c=2;
  flag=sort(rand(1,n)>0.5);
  x=randn(2,n)+5*[(flag)*2-1; zeros(1,n)];
  ytrue=flag+1;
 case 2 % Four Gaussians
  n=200;
  d=2;
  c=4;
  flag1=(rand(1,n)>0.5);
  flag2=(rand(1,n)>0.5);
  x=0.5*randn(2,n)+[flag1*4-2; flag2*4-2];
  ytrue=flag1+flag2*2+1;
 case 3 % Spiral
  n=200;
  d=2;
  c=2;
  x=zeros(2,n);
  for i=1:n/2
    r=1+4*(i-1)/n;
    t=(i-1)*3/n*pi;
    x(1,i)=r*cos(t);
    x(2,i)=r*sin(t);
    x(1,i+n/2)=r*cos(t+pi);
    x(2,i+n/2)=r*sin(t+pi);
  end
  x=x+0.1*randn(2,n);
  ytrue=[ones(1,n/2) 2*ones(1,n/2)];
 case 4 % High and low densities
  n=200;
  d=2;
  c=2;
  x=[0.1*randn(2,n/2) randn(2,n/2)];
  ytrue=[ones(1,n/2) 2*ones(1,n/2)];
 case 5 % Circle and Gaussian
  n=200;
  c=2;
  x(1,:)=[5*cos(linspace(0,2*pi,n/2)) randn(1,n/2)];
  x(2,:)=[5*sin(linspace(0,2*pi,n/2)) randn(1,n/2)];
  x=x+0.1*randn(2,n);
  ytrue=[ones(1,n/2) 2*ones(1,n/2)];
 case 6 % Concentric circles
  n=300;
  d=2;
  c=3;
  x(1,:)=[1*cos(linspace(0,2*pi,n/3)) 3*cos(linspace(0,2*pi,n/3)) 5*cos(linspace(0,2*pi,n/3))];
  x(2,:)=[1*sin(linspace(0,2*pi,n/3)) 3*sin(linspace(0,2*pi,n/3)) 5*sin(linspace(0,2*pi,n/3))];
  x=x+0.1*randn(2,n);
  ytrue=[ones(1,n/3) 2*ones(1,n/3) 3*ones(1,n/3)];

 case {10,11,12,13,14,15,16,17,18,19,20,21,22,23} %IDA
  data_name={'banana','breast-cancer','diabetes','flare-solar','german',...
                 'heart','image','ringnorm','splice','thyroid',...
                 'titanic','twonorm','waveform'};
  data_name_disp={'Banana','Cancer','Diabetes','Solar','German',...
                      'Heart','Image','Ringnorm','Splice','Thyroid',...
                      'Titanic','Twonorm','Waveform'};
  load(['/home/local/data/IDA/' data_name{dataset-9}],'x_train','y_train');
  x=x_train(:,:,trial)';
  ytrue=y_train(:,trial)'+1;
  c=2;
  
 case {30,31,32,33,34} %USPS
  load('/home/local/data/letter/USPS')
switch dataset
case 30
   nc=100;
case 31
   nc=50;
case 32
   nc=200;
case 33
   nc=20;
case 34
   nc=500;
end

  tmp=randperm(708);
  z=tmp(1:nc);
  
  x=[x1(z,:)' x2(z,:)' x3(z,:)' x4(z,:)' x5(z,:)' ...
     x6(z,:)' x7(z,:)' x8(z,:)' x9(z,:)' x0(z,:)'];
  ytrue=[1*ones(1,nc) 2*ones(1,nc) 3*ones(1,nc) 4*ones(1,nc) 5*ones(1,nc) ...
         6*ones(1,nc) 7*ones(1,nc) 8*ones(1,nc) 9*ones(1,nc) 10*ones(1,nc)];
  c=10;
  
 case {40,41} %Olivetti faces
  load('/home/local/data/face/olivettifaces')
  
  tmp=randperm(40);
  
switch dataset
case 40
  c=10;
case 41
  c=40;
end

  x=[];
  ytrue=[];
  for i=1:c
    a=(tmp(i)-1)*10+1;
    x=[x faces(:,a:a+9)];
    ytrue=[ytrue i*ones(1,10)];
  end
  [d,n]=size(x);
  x=x./repmat(std(x,1,2),[1 n]);
  disp(sprintf('d=%g, n=%g, c=%g',d,n,c));
 case {50,51,52,53,54,55}
  load('Zelnik_data')
  x=XX{dataset-49}';
  c=group_num(dataset-49);
  ytrue=[ones(1,size(x,2)-c+1) 2:c]; %dummy
 
 case {60,61,62,63,64,65,66,67,68,69} %TechTC300

file_name={
'Exp_100241_17900',
'Exp_10341_10755',
'Exp_10341_14271',
'Exp_10341_14525',
'Exp_10341_186330',
'Exp_10341_194927',
'Exp_10341_61792',
'Exp_10350_10539',
'Exp_10350_13928',
'Exp_10350_194915',
'Exp_10385_14525',
'Exp_10385_25326',
'Exp_10385_269078',
'Exp_10385_299104',
'Exp_10385_312035',
'Exp_10539_10567',
'Exp_10539_11346',
'Exp_10539_186330',
'Exp_10539_194915',
'Exp_10539_20673',
'Exp_10539_300332',
'Exp_10539_61792',
'Exp_10539_85489',
'Exp_10567_11346',
'Exp_10567_12121',
'Exp_10567_46076',
'Exp_106614_114202',
'Exp_10762_325847',
'Exp_10762_524208',
'Exp_10762_5861635',
'Exp_108204_2631',
'Exp_108204_42345',
'Exp_1092_1110',
'Exp_1092_135724',
'Exp_1092_789236',
'Exp_1110_47793',
'Exp_11342_9639',
'Exp_11346_17360',
'Exp_11346_22294',
'Exp_114202_190888',
'Exp_114202_58074',
'Exp_114857_312807',
'Exp_114857_40392',
'Exp_114857_535947',
'Exp_114857_789236',
'Exp_11498_14517',
'Exp_1155181_138526',
'Exp_1155181_2597',
'Exp_1155181_29965',
'Exp_1155181_40392',
'Exp_1155181_5560',
'Exp_1155181_789236',
'Exp_123412_17899',
'Exp_123412_233389',
'Exp_123412_325847',
'Exp_123906_2592',
'Exp_123906_463854',
'Exp_124388_23112',
'Exp_124388_7393',
'Exp_127007_17900',
'Exp_127749_72031',
'Exp_135724_2631',
'Exp_137433_449165',
'Exp_138526_2597',
'Exp_138526_2631',
'Exp_138526_789236',
'Exp_139208_23038',
'Exp_13928_18479',
'Exp_13928_186330',
'Exp_13928_300332',
'Exp_13928_312035',
'Exp_13928_71892',
'Exp_14271_194927',
'Exp_14271_20186',
'Exp_14271_312035',
'Exp_14271_46076',
'Exp_14517_186330',
'Exp_14517_20673',
'Exp_14518_472203',
'Exp_14518_96104',
'Exp_14525_194927',
'Exp_14525_61792',
'Exp_14630_18479',
'Exp_14630_20186',
'Exp_14630_300332',
'Exp_14630_312035',
'Exp_14630_814096',
'Exp_14630_94142',
'Exp_14653_5861635',
'Exp_1622_42350',
'Exp_17088_312651',
'Exp_17088_421943',
'Exp_173089_40398',
'Exp_173089_524208',
'Exp_17360_20186',
'Exp_17360_46875',
'Exp_17899_240218',
'Exp_17899_278949',
'Exp_17899_48446',
'Exp_17900_61765',
'Exp_17900_704167',
'Exp_181232_215009',
'Exp_181232_257734',
'Exp_181232_789236',
'Exp_18479_186330',
'Exp_18479_20186',
'Exp_18479_20673',
'Exp_18479_46076',
'Exp_186330_195558',
'Exp_186330_300332',
'Exp_186330_314499',
'Exp_186330_46076',
'Exp_186330_94142',
'Exp_190005_287061',
'Exp_190005_454516',
'Exp_190005_58074',
'Exp_190005_5861635',
'Exp_190005_72031',
'Exp_190005_849002',
'Exp_194915_194927',
'Exp_194915_324745',
'Exp_194915_67777',
'Exp_194927_20186',
'Exp_194927_299104',
'Exp_194927_312035',
'Exp_194927_46875',
'Exp_194927_61792',
'Exp_1996_261990',
'Exp_20186_22294',
'Exp_20186_61792',
'Exp_203793_204402',
'Exp_203793_28718',
'Exp_203793_7393',
'Exp_203793_81066',
'Exp_203793_86383',
'Exp_204402_287061',
'Exp_204402_29041',
'Exp_204402_72031',
'Exp_204402_7393',
'Exp_205242_463854',
'Exp_20546_215009',
'Exp_20546_65374',
'Exp_20546_96104',
'Exp_20673_269078',
'Exp_20673_312035',
'Exp_20673_46076',
'Exp_20826_29965',
'Exp_210192_520393',
'Exp_210192_8564',
'Exp_21119_96104',
'Exp_211244_224533',
'Exp_21433_418948',
'Exp_21433_5823851',
'Exp_215009_418948',
'Exp_215009_61765',
'Exp_217155_3093',
'Exp_222417_472203',
'Exp_22294_25575',
'Exp_22294_46076',
'Exp_224533_25321',
'Exp_224533_83261',
'Exp_224533_88266',
'Exp_23038_47793',
'Exp_23038_68416',
'Exp_23038_83261',
'Exp_23222_430894',
'Exp_23222_849002',
'Exp_233389_458776',
'Exp_233389_849002',
'Exp_233389_86383',
'Exp_234662_2597',
'Exp_234662_52622',
'Exp_234662_5823851',
'Exp_238688_56994',
'Exp_238688_57037',
'Exp_240218_271300',
'Exp_240218_325847',
'Exp_240218_474717',
'Exp_240790_47793',
'Exp_25575_275169',
'Exp_25575_47456',
'Exp_2592_3431',
'Exp_2592_42357',
'Exp_25936_94142',
'Exp_2597_56702',
'Exp_261259_60532',
'Exp_261259_81066',
'Exp_261259_8564',
'Exp_261990_8564',
'Exp_2631_449165',
'Exp_2631_789236',
'Exp_2631_83261',
'Exp_263248_5861635',
'Exp_266541_278949',
'Exp_266541_301161',
'Exp_266541_5861635',
'Exp_266541_60741',
'Exp_268608_49870',
'Exp_269078_324745',
'Exp_269078_46076',
'Exp_271300_49870',
'Exp_271300_5861635',
'Exp_271300_849002',
'Exp_275733_58074',
'Exp_278949_40348',
'Exp_278949_849002',
'Exp_280052_325847',
'Exp_280052_83450',
'Exp_28718_849002',
'Exp_28718_8564',
'Exp_29041_7393',
'Exp_299104_312035',
'Exp_299104_46076',
'Exp_299104_58108',
'Exp_29965_68416',
'Exp_300332_85489',
'Exp_301161_849002',
'Exp_303829_789236',
'Exp_3093_421943',
'Exp_312651_49870',
'Exp_312651_5861635',
'Exp_312807_449927',
'Exp_316970_85489',
'Exp_319115_472203',
'Exp_324745_61792',
'Exp_324745_85489',
'Exp_325847_5861635',
'Exp_325847_8564',
'Exp_332386_61792',
'Exp_332386_85489',
'Exp_3431_48472',
'Exp_344007_47793',
'Exp_344007_789236',
'Exp_364836_71892',
'Exp_378028_5841153',
'Exp_40378_849002',
'Exp_40392_61765',
'Exp_40392_789236',
'Exp_40398_421943',
'Exp_40398_849002',
'Exp_40622_69440',
'Exp_40622_8292',
'Exp_406522_85489',
'Exp_415500_454516',
'Exp_415500_5861635',
'Exp_418948_71432',
'Exp_418948_789236',
'Exp_421943_789236',
'Exp_42345_56702',
'Exp_43404_47186',
'Exp_43404_5861635',
'Exp_43404_849002',
'Exp_449927_789236',
'Exp_45502_5838985',
'Exp_458776_5861635',
'Exp_458776_81066',
'Exp_458776_849002',
'Exp_458776_8564',
'Exp_46076_61792',
'Exp_463854_58074',
'Exp_46875_61792',
'Exp_472203_57037',
'Exp_472203_71432',
'Exp_472203_789236',
'Exp_47418_814096',
'Exp_47456_497201',
'Exp_474717_849002',
'Exp_48446_69440',
'Exp_49502_56994',
'Exp_520393_849002',
'Exp_52622_60974',
'Exp_5310_9639',
'Exp_535947_57037',
'Exp_5560_592118',
'Exp_5560_704167',
'Exp_56994_96104',
'Exp_58108_85489',
'Exp_5823851_789236',
'Exp_5838985_789236',
'Exp_5861635_60741',
'Exp_5861635_72031',
'Exp_5861635_849002',
'Exp_592118_68416',
'Exp_60532_8567',
'Exp_60741_849002',
'Exp_60974_789236',
'Exp_61792_814096',
'Exp_6920_8366',
'Exp_69753_85489',
'Exp_72031_849002',
'Exp_8308_8366',
'Exp_83261_88266',
'Exp_85489_90753',
'Exp_8564_8567',
'Exp_8564_8767'};

input=[];
output=[];
  load(sprintf('/home/local/data/Okanohara-Document/DATA/%s/vectors.mat',...
               file_name{dataset-59}),'input','output')

%  x0=full(input);
%  [d,n]=size(x0);
%  xc=x0-repmat(mean(x0,2),[1 n]);
%  K=xc'*xc;
%  [eigvec,eigval]=eigs(K,10);
%  x=eigvec';
  x=full(input);
  ytrue=((output==1)+1)';
  c=2;  

 case {70} %Kato Chemo
input=[];
output=[];
  load('/home/local/data/Kato-Chemo/data.mat','input','output')
  x=full(input);
  ytrue=output';
  c=2;  

   case {80} %Hachiya ALKAN
% $$$             case 1 % 1: walk, 2: run
% $$$                 users = [243 278 131 167 141 124 242 77 178];
% $$$             case 2 % 1: walk, 3: stair up
% $$$                 users = [3 77 117 124 125 133 141 155 167 178];
% $$$             case 3 % 1: walk, 4: stair down
% $$$                 users = [3 249 116 81 124 141 77 218 225 122];
% $$$             case 4 % 1: walk, 5: stand up
% $$$                 users = [141 137 243 155 178 271 3 258 193 134];
% $$$             case 5 % 1: walk, 6: sit down
% $$$                 users = [243 124 137 125 279 134 229 193 117 114];
% $$$             case 6 % 1: walk, 8: eating
% $$$                 users = [77 141 229 104 243 124 176 163 216 125 137];    
x=[];
ytrue=[];
nn=100;
c=3;
for i=1:c
NAME={'fet243_1','fet278_1','fet131_1','fet141_1',...
      'fet124_1','fet242_1','fet178_1'};
load(sprintf('/home/local/data/Hachiya-ALKAN/%s',NAME{i}))
xtmp=eval(sprintf('%s(%s(:,7)==1,[1 2 3 5 6])',NAME{i},NAME{i}));
tmp=randperm(size(xtmp,1));
x=[x xtmp(tmp(1:nn),:)'];
ytrue=[ytrue i*ones(1,nn)];
end
   case {85} %Hachiya ALKAN
x=[];
ytrue=[];
nn=100;
c=3;
for i=1:c
  %walk, run, stand up, sit down, sitting, bycicle
NAME={'fet243_1','fet243_2','fet243_5','fet243_6','fet243_8','fet243_13'};
load(sprintf('/home/local/data/Hachiya-ALKAN/%s',NAME{i}))
xtmp=eval(sprintf('%s(%s(:,7)==1,[1 2 3 5 6])',NAME{i},NAME{i}));
tmp=randperm(size(xtmp,1));
x=[x xtmp(tmp(1:nn),:)'];
ytrue=[ytrue i*ones(1,nn)];
end

                   case {90} %Yamada VOC
                load('/home/local/data/Yamada/DataVOC_distribute.mat');
                nmax=size(X,2);
                nn=100;
                dd=50;
                tmp=randperm(nmax);
                xtmp=princomp(X(:,tmp(1:nn))');
                x=xtmp(:,1:dd)'*X(:,tmp(1:nn));
                ytrue=Y(1,tmp(1:nn))+1;
                c=2;
                
                   case {100} %Yamada freesound
                load('/home/local/data/Yamada/data_freesound_BOF.mat');
                nmax=size(X,2);
                nn=100;
                dd=50;
                tmp=randperm(nmax);
                xtmp=princomp(X(:,tmp(1:nn))');
                x=xtmp(:,1:dd)'*X(:,tmp(1:nn));
                ytrue=Y(1,tmp(1:nn))+1;
                c=2;
 case {110,111,112,113} % Yamada voice conversion
  load('/home/local/data/Yamada/testDataYamaHiya.mat')
tmp1=randperm(size(data_src,1));
tmp2=randperm(size(data_tar,1));
switch dataset
case 110
nn=100;
case 111
nn=50;
case 112
nn=500;
case 113
nn=200;
end

x=[data_src(tmp1(1:nn),:)' data_tar(tmp1(1:nn),:)'];
ytrue=[ones(1,nn) 2*ones(1,nn)];
c=2;  


 case {120} % 20 Newsgroups (100 words)
  load('/home/local/data/text/20news_w100.mat')
c=4;
x=[];
ytrue=[];
nn=100;
for y=1:c
  ny=sum(newsgroups==y);
  tmp=randperm(ny);
  xtmp=documents(:,newsgroups==y);
  x=full([x xtmp(:,tmp(1:nn))]);
  ytrue=[ytrue y*ones(1,nn)];
end

 case {130} % 20 Newsgroups (10000 words)
  load('/home/local/data/text/20news_w10000_counts.mat','labels')
  load('/home/local/data/text/20news_w10000_tfidf.mat','tfidf')

% $$$   [nSmp,nFea] = size(counts');
% $$$   [idx,jdx,vv] = find(counts');
% $$$ df = full(sum(sparse(idx,jdx,1),1));
% $$$ idf = log(nSmp./df);
% $$$ tfidf = sparse(idx,jdx,log(vv)+1)';
% $$$ idf = idf';
% $$$ for i = 1:nSmp
% $$$      tfidf(:,i) = tfidf(:,i) .* idf;
% $$$ end
% $$$   save('/home/local/data/text/20news_w10000_tfidf.mat','tfidf')

c=7;
x=[];
ytrue=[];
nn=100;
for y=1:c
  ny=sum(labels==y);
  tmp=randperm(ny);
  xtmp=tfidf(:,labels==y);
  x=full([x xtmp(:,tmp(1:nn))]);
  ytrue=[ytrue y*ones(1,nn)];
end

%x=x(sum(x,2)~=0,:);

dd=50;
xtmp=princomp(x');
x=xtmp(:,1:dd)'*x;

 case {140} %Kimura NLP-line
  load('/home/local/data/Kimura-NLP/Senseval2-WordSenseDisambiguation/line.mat')
c=6;
x=[];
ytrue=[];
nn=100;
for y=1:c
  ny=sum(Y==y);
  tmp=randperm(ny);
  xy=X(Y==y,:)';
  x=full([x xy(:,tmp(1:nn))]);
  ytrue=[ytrue y*ones(1,nn)];
end

%x=x(sum(x,2)~=0,:);

dd=50;
xtmp=princomp(x');
x=xtmp(:,1:dd)'*x;

 case {150,151} %Kimura NLP-intesrest
  load('/home/local/data/Kimura-NLP/Senseval2-WordSenseDisambiguation/interest.mat')
c=3;
x=[];
ytrue=[];
switch dataset
  case 150
nn=100;
y_list=[4 5 6];
end
for y_index=1:length(y_list)
  y=y_list(y_index);
  ny=sum(Y==y);
  tmp=randperm(ny);
  xy=X(Y==y,:)';
  x=full([x xy(:,tmp(1:nn))]);
  ytrue=[ytrue y_index*ones(1,nn)];
end

%x=x(sum(x,2)~=0,:);

dd=50;
xtmp=princomp(x');
x=xtmp(:,1:dd)'*x;

end




[d,n]=size(x);
x=x-repmat(mean(x,2),[1 n]);
x=x./repmat(std(x,1,2),[1 n]);
