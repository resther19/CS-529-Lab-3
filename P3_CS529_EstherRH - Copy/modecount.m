function modecnt = modecount(ClassCounts,mxcount)
% This functions returns the mode of the class labels predicted over a number of feature vectors. 
        modecnt = Inf(size(ClassCounts,2),1);
        for nc = 1:size(ClassCounts,2)
            modecnt(nc) = histc(ClassCounts(:,nc),mxcount(nc));
        end
    end