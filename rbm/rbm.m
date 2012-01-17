% rbm - training restricted Boltzmann machine using Gibbs sampling
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

if R.learning.persistent_cd == 1 && R.enhanced_grad.use == 1
    warning('PCD tends to diverge easily when used together with enhanced gradient');
end

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

vbias_grad_old = zeros(size(R.vbias'));
hbias_grad_old = zeros(size(R.hbias'));
W_grad_old = zeros(size(R.W));

hbias0_old = zeros(1, R.structure.n_hidden);

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

        if adaptive_lrate
            if mb == n_minibatches
                mb_next = 1;
            else
                mb_next = mb + 1;
            end

            v0_next = patches((mb_next-1) * minibatch_sz + 1:min(mb_next * minibatch_sz, n_samples), :);
        end

        h0 = sigmoid(bsxfun(@plus, v0 * R.W, R.hbias'));

        vbias0 = mean(v0, 1);
        hbias0 = mean(h0, 1);
        W0 = (v0' * h0) / size(v0,1);

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
            if (mf_update ~= 0)
                h1 = binornd(1, h1);
            end
            v1 = sigmoid(bsxfun(@plus,h1 * R.W', R.vbias'));
            if (mf_update ~= 0)
                v1 = binornd(1, v1, size(v1,1), size(v1,2));
            end
            h1 = sigmoid(bsxfun(@plus,v1 * R.W,R.hbias'));
        end

        vbias1 = mean(v1, 1);
        hbias1 = mean(h1, 1);
        W1 = (v1' * h1) / size(v1, 1);

        % get gradient
        vbias_grad = vbias0 - vbias1;
        hbias_grad = hbias0 - hbias1;
        W_grad = W0 - W1;

        if enhanced_grad == 1
            vacts = (mean(v1, 1) + mean(v0, 1))/2;
            hacts = (mean(h1, 1) + mean(h0, 1))/2;

            if bit_flipping == 1
                vacts = round(vacts);
                hacts = round(hacts);
            end

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

        % now we find the optimal(?) step size
        if adaptive_lrate == 1
            base_lrate = actual_lrate;

            % we assume that the fantasy particles are truly from the model.
            vf = v1;
            if (R.adaptive_lrate.alrate_use_current_batch)
                vd = v0;
            else
                vd = v0_next;
            end

            candidate_lrates;

            [cE, cEmin, cEmax, cEs] = rbm_energy(vf, R.W, R.vbias, R.hbias);
            [dcE, dcEmin, dcEmax, dcEs] = rbm_energy(vd, R.W, R.vbias, R.hbias);
            % current
            curr_cost = sum(-dcEs);
            if R.verbose == 1
                fprintf(2, '%f: current cost %f\n', actual_lrate, curr_cost);
            end

            % search
            for s=1:length(cand_lrates)
                cand_lrate = cand_lrates(s);

                W_test = R.W + cand_lrate * (momentum * W_grad_old + (1-momentum) * W_grad);
                vbias_test = R.vbias + cand_lrate * (momentum * vbias_grad_old' + (1-momentum) * vbias_grad');
                hbias_test = R.hbias + cand_lrate * (momentum * hbias_grad_old' + (1-momentum) * hbias_grad');

                [dE, dEmin, dEmax, dEs] = rbm_energy(vd, W_test, vbias_test, hbias_test);
                [fE, fEmin, fEmax, fEs] = rbm_energy(vf, W_test, vbias_test, hbias_test);

                now_cost = sum(-dEs - logsum(-fEs + cEs) + log(size(vf,1)));

                costs(s) = now_cost;
            end

            [chosen_cost chosen_index] = max(costs);
            actual_lrate = cand_lrates(chosen_index);

            if R.verbose == 1
                fprintf(2, 'lrate %f (cost %f) selected\n', actual_lrate, chosen_cost);
            end
        else
            actual_lrate = R.learning.lrate;
        end

        R.signals.lrates = [R.signals.lrates actual_lrate];

        % update
	vbias_grad_old = (1-momentum) * vbias_grad + momentum * vbias_grad_old;
	hbias_grad_old = (1-momentum) * hbias_grad + momentum * hbias_grad_old;
	W_grad_old = (1-momentum) * W_grad + momentum * W_grad_old;

        R.vbias = R.vbias + actual_lrate * vbias_grad_old' - weight_decay * R.vbias;
        R.hbias = R.hbias + actual_lrate * hbias_grad_old' - weight_decay * R.hbias;
        R.W = R.W + actual_lrate * W_grad_old - weight_decay * R.W;

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
        
        if R.debug.do_display == 1 && mod(R.iteration.n_updates, R.debug.display_interval) == 0
            R.debug.display_function (R.debug.display_fid, R, v0, v1, W_grad, vbias_grad, hbias_grad);
            drawnow;
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


