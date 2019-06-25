% Script to download the mat files and write into the form [A,b] for later
% python usage in the sketching scripts.

urls = {'https://sparse.tamu.edu/mat/Brogan/specular.mat', 
        'https://sparse.tamu.edu/mat/JGD_Taha/abtaha2.mat'} ; 
    
datasets = {'abtaha2','specular' } ; 

for ii = 1:2
    %url = urls{ii} ; 
    fname = strcat(datasets{ii},'.mat' ) ; 
    prob_struct = load(fname) ; 
    data = prob_struct.Problem.A ; 
    
    try
        A = data.A ; 
        b = data.b ; 
        if size(b,2) > 1
            b = b(:,1) ; 
        end
    catch
        A = data(:,1:end-1) ; 
        b = data(:,end) ; 
    end
    
    if ii == 2
        rng(100)
        b = randn(size(A,1),1) ; 
        cols = randperm(size(A,2)) ;
        A = A(:,cols(1:50)) ; 
        
        disp(100*nnz(A)/(size(A,1)*size(A,2)))
    end
    disp(size(A)) ; 
    disp(size(b)) ; 
    
    % save the files
    out_file = strcat('sp_',fname) ; 
    save(out_file,'A','b') ; 
    
    
end
