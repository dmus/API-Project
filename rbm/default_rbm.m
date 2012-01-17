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
function [R] = default_rbm (n_visible, n_hidden);
    % data type
    R.data.binary = 0;
    %R.data.binary = 0; % for GBRBM

    % learning parameters
    R.learning.cd_k = 1;
    R.learning.persistent_cd = 0;
    R.learning.mf_update = 0;
    R.learning.lrate = 1e-2;
    R.learning.momentum = 0;
    R.learning.weight_decay = 0;
    R.learning.weight_scale = 1/sqrt(n_visible + n_hidden);
    R.learning.minibatch_sz = 100;

    % Gaussian-Bernoulli RBM
    R.grbm.do_vsample = 1;
    R.grbm.do_normalize = 1;
    R.grbm.do_normalize_std = 1;
    R.grbm.learn_sigmas = 1;
    R.grbm.sigmas_ub = Inf;
    R.grbm.use_single_sigma = 1;

    % stopping criterion
    % if you happen to know some other criteria, please, do add them.
    % if the criterion is zero, it won't stop unless the whole training epochs were consumed.
    R.stop.criterion = 2;
    % criterion == 1
    R.stop.recon_error.tolerate_count = 1000;
    % criterion == 2
    R.stop.lrate.lowlrate_threshold = 1e-8;

    % adaptive learning rate
    R.adaptive_lrate.use = 1;
    R.adaptive_lrate.alrate_use_current_batch = 0;
    R.adaptive_lrate.max_iter_up = 1;
    R.adaptive_lrate.max_iter_down = 1;
    R.adaptive_lrate.exp_up = 1.01;
    R.adaptive_lrate.exp_down = 0.99;
    R.adaptive_lrate.lrate_ub = Inf;
    R.adaptive_lrate.lrate_lb = -Inf;

    % enhanced gradient
    R.enhanced_grad.use = 1;
    R.enhanced_grad.bit_flipping = 0; % not even supported

    % adaptive momentum
    R.adaptive_momentum.use = 0;
    R.adaptive_momentum.max_iter_up = 2;
    R.adaptive_momentum.max_iter_down = 2;
    R.adaptive_momentum.exp_up = 1.01;
    R.adaptive_momentum.exp_down = 0.99;

    % structure
    R.structure.n_visible = n_visible;
    R.structure.n_hidden = n_hidden;

    % initializations
    R.W_init = R.learning.weight_scale * 2 * (rand(n_visible, n_hidden) - 0.5);
    R.vbias_init = zeros(n_visible, 1);
    R.hbias_init = zeros(n_hidden, 1);
    R.sigmas_init = ones(1, n_visible); % only for GBRBM
    R.W = R.W_init;
    R.vbias = R.vbias_init;
    R.hbias = R.hbias_init;
    R.sigmas = R.sigmas_init; % only for GBRBM

    % iteration
    R.iteration.n_epochs = 100;
    R.iteration.n_updates = 0;

    % parallel tempering
    R.parallel_tempering.use = 1;
    R.parallel_tempering.n_chains = 11;
    R.parallel_tempering.swap_interval = 1;

    % learning signals
    R.signals.recon_errors = [];
    R.signals.lrates = [];
    R.signals.momentums = [];
    R.signals.norms.gradients = [];
    R.signals.norms.adjustments = [];
    R.signals.norms.gradients0 = [];
    R.signals.n_epochs = 0;

    % debug
    R.verbose = 0;
    R.debug.do_display = 0;
    R.debug.display_interval = 10;
    R.debug.display_fid = 1;
    R.debug.display_function = @visualize_rbm;

    % hook
    R.hook.per_epoch = {@save_intermediate, {'rbm.mat'}};
    R.hook.per_update = {@print_n_updates, {}};

