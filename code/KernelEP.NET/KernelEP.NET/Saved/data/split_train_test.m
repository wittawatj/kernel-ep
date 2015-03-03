function split_train_test()
% load data, split into train and test sets, write back with different
% names
%

%   X         4x1372            43904  double               
%   Y         1x1372            10976  double            
% 
%   Value    Count   Percent
%       0      762     55.54%
%       1      610     44.46%
[Xtr, Ytr, Xte, Yte] = split_save(200, 'banknote_norm');
write_xy(Xtr, Ytr, 'banknote_norm_tr');
write_xy(Xte, Yte, 'banknote_norm_te');

%   Name      Size             Bytes  Class     Attributes
% 
%   X         4x748            23936  double              
%   Y         1x748             5984  double              
% 
%   Value    Count   Percent
%       0      570     76.20%
%       1      178     23.80%
[Xtr, Ytr, Xte, Yte] = split_save(200, 'blood_transfusion_norm');
write_xy(Xtr, Ytr, 'blood_transfusion_norm_tr');
write_xy(Xte, Yte, 'blood_transfusion_norm_te');


%   X         9x100             7200  double              
%   Y         1x100              800  double              
% 
%   Value    Count   Percent
%       0       88     88.00%
%       1       12     12.00%
[Xtr, Ytr, Xte, Yte] = split_save(50, 'fertility_norm');
write_xy(Xtr, Ytr, 'fertility_norm_tr');
write_xy(Xte, Yte, 'fertility_norm_te');


%   X         33x351            92664  double              
%   Y          1x351             2808  double              
% 
%   Value    Count   Percent
%       0      225     64.10%
%       1      126     35.90%
[Xtr, Ytr, Xte, Yte] = split_save(200, 'ionosphere_norm');
write_xy(Xtr, Ytr, 'ionosphere_norm_tr');
write_xy(Xte, Yte, 'ionosphere_norm_te');

end

function write_xy(X, Y, fname)
    save(fname, 'X', 'Y');
end

function [Xtr, Ytr, Xte, Yte] = split_save(ntr, fname)
    s = load(sprintf('%s', fname));
    tr_portion = ntr/size(s.X, 2);
    Itr = straportion(s.Y, tr_portion, 1);    
    Xtr = s.X(:, Itr);
    Ytr = s.Y(:, Itr);
    Xte = s.X(:, ~Itr);
    Yte = s.Y(:, ~Itr);
end