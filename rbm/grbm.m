% grbm - training GB-RBM using Gibbs sampling
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [R] = rbm(R, patches);

if R.enhanced_grad.bit_flipping == 1
    error('NOT SUPPORTED');
end

actual_lrate = R.learning.lrate;

if R.adaptive_lrate.use == 1
    initial_lrate = R.learning.lrate;
    actual_lrate = initial_lrate;
end

if R.adaptive_lrate.alrate_use_current_batch == 1
    warning('Adaptive learning rate might diverge');
end

n_samples = size(patches, 1);
if R.structure.n_visible ~= size(patches, 2)
    error('Data is not properly aligned');
end
n_visible = R.structure.n_visible;

minibatch_sz = R.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

n_epochs = R.iteration.n_epochs;
adaptive_lrate = R.adaptive_lrate.use;
enhanced_grad = R.enhanced_grad.use;
bit_flipping = R.enhanced_grad.bit_flipping;

cd_k = R.learning.cd_k;
persistent_cd = R.learning.persistent_cd;
momentum = R.learning.momentum;
weight_decay = R.learning.weight_decay;
mf_update = R.learning.mf_update;

n_hidden = R.structure.n_hidden;
n_visible = R.structure.n_visible;

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

do_normalize = R.grbm.do_normalize;
do_normalize_std = R.grbm.do_normalize_std;
update_sigmas = R.grbm.learn_sigmas;
do_vsample = R.grbm.do_vsample;

lrate_lb = R.adaptive_lrate.lrate_lb;
lrate_ub = R.adaptive_lrate.lrate_ub;

if do_normalize == 1
    % make it zero-mean
    patches_mean = mean(patches, 1);
    patches = bsxfun(@minus, patches, patches_mean);
end

if do_normalize_std ==1
    % make it unit-variance
    patches_std = std(patches, [], 1);
    patches = bsxfun(@rdivide, patches, patches_std);
end

n_samples = size(patches, 1);

logsigmas = log(R.sigmas.^2);

% data statistics
mu_d = mean(patches, 1)';
sigmas_d = std(patches, [], 1);
logsigmas_d = log(sigmas_d.^2);

% upper-bound.. but is there any need for it?
sigmas_ub = R.grbm.sigmas_ub;
logsigmas_ub = log(R.grbm.sigmas_ub);

vbias_grad_old = zeros(size(R.vbias))';
hbias_grad_old = zeros(size(R.hbias))';
W_grad_old = zeros(size(R.W));
sigma_grad_old = zeros(size(R.vbias))';

n_minibatches = ceil(n_samples / minibatch_sz);
n_updates = 0;

epsilon_sigma = 1e-8;
epsilon_logsigma = log(epsilon_sigma^2);

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

if R.debug.do_display == 1
    figure(R.debug.display_fid);
end

for step=1:n_epochs
    if R.verbose
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
    end
    for mb=1:n_minibatches
        R.iteration.n_updates = R.iteration.n_updates + 1;

        % p_0
        v0 = patches((mb-1) * minibatch_sz + 1:min(mb * minibatch_sz, n_samples), :);
        mb_sz = size(v0,1);
        
        % just for a bit of speed-up
        fmb_sigma2s = repmat(R.sigmas, [mb_sz 1]);
        
        if adaptive_lrate == 1
            if mb == n_minibatches
                mb_next = 1;
            else
                mb_next = mb + 1;
            end
            
            v0_next = patches((mb_next-1) * minibatch_sz + 1:min(mb_next * minibatch_sz, n_samples), :);
            next_mb_sz = size(v0_next,1);
            
            if persistent_cd == 0
                nmb_sigma2s = repmat(R.sigmas, [next_mb_sz 1]);
            else
                if next_mb_sz ~= minibatch_sz
                    nmb_sigma2s = repmat(R.sigmas, [next_mb_sz 1]);
                else
                    nmb_sigma2s = fmb_sigma2s;
                end
            end
        end
        
        h0 = 1./(1 + exp(bsxfun(@minus, -bsxfun(@rdivide, v0, R.sigmas.^2) * R.W, R.hbias')));
        
        W0 = bsxfun(@rdivide, (v0' * h0) / mb_sz, R.sigmas.^2');
        vbias0 = bsxfun(@rdivide, mean(v0, 1), R.sigmas.^2);
        hbias0 = mean(h0, 1);
        sigma0 = bsxfun(@rdivide, mean((bsxfun(@minus, v0, R.vbias').^2) - v0 .* (h0 * R.W'), 1), R.sigmas.^2);

        % compute reconstruction error
        hr = binornd(1, h0, size(h0,1), size(h0,2));
        vr = sigmoid(bsxfun(@plus,hr * R.W',R.vbias'));

        R.signals.recon_errors = [R.signals.recon_errors mean(sum((v0 - vr).^2,2))];

        % p_1
        if (persistent_cd ~= 0 && exist('h1') == 0)
            h1 = h0;
        end
        
        if (persistent_cd == 0)
            h1 = h0;
        end
        
        for k=1:cd_k
            if mf_update == 0
                h1 = binornd(1, h1, size(h1,1), size(h1,2));
            end
            v1_mean = bsxfun(@plus, h1 * R.W', R.vbias');
            if (do_vsample)
                v1 = normrnd(v1_mean, fmb_sigma2s);
            else
                v1 = v1_mean;
            end
            if (sum(sum(isnan(v1))) > 0)
                error('NaN found in the visual fantasy particles.\n It is advisable to adjust learning parameters.');
            end
            h1 = sigmoid(bsxfun(@plus, bsxfun(@rdivide, v1, R.sigmas.^2) * R.W, R.hbias'));
            if (sum(sum(isnan(h1))) > 0)
                error('NaN found in the hidden fantasy particles.\n It is advisable to adjust learning parameters.');
            end
        end
        
        W1 = bsxfun(@rdivide, (v1' * h1) / mb_sz, R.sigmas.^2');
        vbias1 = bsxfun(@rdivide, mean(v1, 1), R.sigmas.^2);
        hbias1 = mean(h1, 1);
        sigma1 = bsxfun(@rdivide, mean((bsxfun(@minus, v1, R.vbias').^2) - v1 .* (h1 * R.W'), 1), R.sigmas.^2);
        
        % get gradient
        vbias_grad = vbias0 - vbias1;
        W_grad = W0 - W1;
        sigma_grad = (sigma0 - sigma1);
        if R.grbm.use_single_sigma == 1
            mean_sigma_grad = mean(sigma_grad);
            sigma_grad = mean_sigma_grad * ones(size(sigma_grad));
        end
        hbias_grad = hbias0 - hbias1;
        
        if enhanced_grad == 1
            vacts = (mean(v1, 1) + mean(v0, 1))/2;
            hacts = (mean(h1, 1) + mean(h0, 1))/2;

            vacts = bsxfun(@rdivide, vacts, R.sigmas.^2);

            R.signals.norms.gradients0 = [R.signals.norms.gradients0; ...
                norm(W_grad(:), 2), norm(vbias_grad(:), 2), norm(hbias_grad(:), 2)];

            W_adj = vbias_grad' * hacts + vacts' * hbias_grad;
            W_adj_grad = W_grad - W_adj;

            vbias_adj = hacts * W_adj_grad';
            hbias_adj = vacts * W_adj_grad;

            W_grad = W_adj_grad;
            vbias_grad = vbias_grad - vbias_adj;
            hbias_grad = hbias_grad - hbias_adj;

            R.signals.norms.adjustments = [R.signals.norms.adjustments; ...
                norm(W_adj(:), 2), norm(vbias_adj(:), 2), norm(hbias_adj(:), 2)];
        end
        
        R.signals.norms.gradients = [R.signals.norms.gradients; ...
            norm(W_grad(:), 2), norm(vbias_grad(:), 2), norm(hbias_grad(:), 2)];
        
        % now we find the optimal step size
        if adaptive_lrate == 1
            base_lrate = actual_lrate;

            vf = v1;

            if (R.adaptive_lrate.alrate_use_current_batch)
                vd = v0;
            else
                vd = v0_next;
            end
            
            candidate_lrates;

            [cE, cEmin, cEmax, cEs] = grbm_energy(vf, R.W, R.vbias, R.hbias, R.sigmas);
            [dcE, dcEmin, dcEmax, dcEs] = grbm_energy(vd, R.W, R.vbias, R.hbias, R.sigmas);
            % current
            curr_cost = sum(-dcEs);
            if R.verbose == 1
                fprintf(2, '%f: current cost %f\n', actual_lrate, curr_cost);
            end
            
            % search
            for s=1:length(cand_lrates)
                cand_lrate = cand_lrates(s);
                
                vbias_test = R.vbias + (1-momentum) * cand_lrate * vbias_grad' + momentum * vbias_grad_old' - weight_decay * R.vbias;
                hbias_test = R.hbias + (1-momentum) * cand_lrate * hbias_grad' + momentum * hbias_grad_old' - weight_decay * R.hbias;
                W_test = R.W + (1-momentum) * cand_lrate * W_grad + momentum * W_grad_old - weight_decay * R.W;
                if update_sigmas == 1
                    logsigmas_test = logsigmas + (1-momentum) * cand_lrate * sigma_grad + momentum * sigma_grad_old - weight_decay * logsigmas;
                    logsigmas_test = max(epsilon_logsigma, min(logsigmas_ub, logsigmas_test));
                    sigmas_test = sqrt(exp(logsigmas_test));
                else
                    sigmas_test = R.sigmas;
                end
                
                [dE, dEmin, dEmax, dEs] = grbm_energy(vd, W_test, vbias_test, hbias_test, sigmas_test);
                [fE, fEmin, fEmax, fEs] = grbm_energy(vf, W_test, vbias_test, hbias_test, sigmas_test);

                now_cost = sum(-dEs - logsum(-fEs + cEs) + log(size(vf,1)));
                
                costs(s) = now_cost;
            end
            
            [chosen_cost chosen_index] = max(costs);
            actual_lrate = max(lrate_lb, min(lrate_ub, cand_lrates(chosen_index)));
        else
            actual_lrate = R.learning.lrate;
        end

        R.signals.lrates = [R.signals.lrates actual_lrate];
        
        if R.debug.do_display == 1 && mod(R.iteration.n_updates, R.debug.display_interval) == 0
            R.debug.display_function (R.debug.display_fid, R, v0, v1, W_grad, vbias_grad, hbias_grad, sigma_grad);
            drawnow;
        end
        
        vbias_grad = actual_lrate * vbias_grad;
        hbias_grad = actual_lrate * hbias_grad;
        W_grad = actual_lrate * W_grad;
        sigma_grad = actual_lrate * sigma_grad;

        % update
        R.vbias = R.vbias + (1-momentum) * vbias_grad' + momentum * vbias_grad_old' - weight_decay * R.vbias;
        R.hbias = R.hbias + (1-momentum) * hbias_grad' + momentum * hbias_grad_old' - weight_decay * R.hbias;
        R.W = R.W + (1-momentum) * W_grad + momentum * W_grad_old - weight_decay * R.W;
    
        if update_sigmas == 1
            logsigmas = logsigmas + (1-momentum) * sigma_grad + momentum * sigma_grad_old - weight_decay * logsigmas;
            logsigmas = max(epsilon_logsigma, min(logsigmas_ub, logsigmas));
            R.sigmas = sqrt(exp(logsigmas));
        end
        
        vbias_grad_old = vbias_grad;
        hbias_grad_old = hbias_grad;
        W_grad = W_grad_old;
        sigma_grad_old = sigma_grad;

        if R.verbose == 1
            fprintf(2, '.');
        end

        if R.stop.criterion > 0
            if R.stop.criterion == 1
                if min_recon_error > R.signals.recon_errors(end)
                    min_recon_error = R.signals.recon_errors(end);
                    min_recon_error_update_idx = R.iteration.n_updates;
                else
                    if R.iteration.n_updates > min_recon_error_update_idx + R.stop.recon_error.tolerate_count 
                        fprintf(2, '\nStopping criterion reached (recon error) %f > %f\n', ...
                            R.signals.recon_errors(end), min_recon_error);
                        stopping = 1;
                        break;
                    end
                end
            elseif R.stop.criterion == 2
                if actual_lrate < R.stop.lrate.lowrate_threshold
                    fprintf(2, '\nStopping criterion reached (lrate) %f > %f\n', ...
                        R.stop.lrate.lowlrate_threshold, actual_lrate);
                    stopping = 1;
                    break;
                end
            else
                error ('Unknown stopping criterion %d', R.stop.criterion);
            end
        end

        if length(R.hook.per_update) > 1
            err = R.hook.per_update{1}(R, R.hook.per_update{2});

            if err == -1
                stopping = 1;
                break;
            end
        end
    end

    if length(R.hook.per_epoch) > 1
        err = R.hook.per_epoch{1}(R, R.hook.per_epoch{2});

        if err == -1
            stopping = 1;
        end
    end

    if stopping == 1
        break;
    end
    
    if R.verbose == 1
        fprintf(2, '\n');
    end
        
    fprintf(2, 'Epoch %d/%d - recon_error: %f norms: %f/%f/%f\n', step, n_epochs, ...
        R.signals.recon_errors(end), ...
        R.W(:)' * R.W(:) / length(R.W(:)), ...
        R.vbias' * R.vbias / length(R.vbias), ...
        R.hbias' * R.hbias / length(R.hbias));
end

