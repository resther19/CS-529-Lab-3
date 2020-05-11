function features = helperscatfeatures(x,sf)
% This function is in support of wavelet scattering examples only.
% This function uses the wavelet transform to calculate wavelet
% coefficients which are used as features.

features = featureMatrix(sf,x(1:2^19),'Transform','log');
features = features(:,1:8:end)';
end

