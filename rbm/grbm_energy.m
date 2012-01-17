
function [E, Emin, Emax, Es] = grbm_energy (x, W, vbias, hbias, sigmas)

n_samples = size(x,1);

n_samples = size(x,1);

% in order to avoid overflow
sigm_lb = -50.;
sigm_ub = 50.;

Wh = ones(n_samples,1)*hbias' + bsxfun(@rdivide, x, sigmas.^2) * W;
Wh = max(min(Wh, sigm_ub), sigm_lb);

Es = bsxfun(@minus,x,vbias');
Es = sum(bsxfun(@rdivide, Es.^2, sigmas.^2),2)/2;

Es = Es - sum(-log(1./(1+exp(Wh))),2);
Es = Es';
Emin = min(Es);
Emax = max(Es);
E = mean(Es);

end
