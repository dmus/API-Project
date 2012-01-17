% add the path of RBM code
addpath('..');

% load MNIST
mnist

% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);
X_labels = X_labels(perm_idx);

hidden_layers = [100 100 50];
n_hidden_layers = length(hidden_layers);

% construct RBM and use default configurations
for l=1:n_hidden_layers
    if l == 1
        X_layer = X;
    else
        X_layer = rbm_get_hidden(X_layer, R{l-1});
    end

    R{l} = default_rbm (size(X_layer, 2), hidden_layers(l));

    if l < n_hidden_layers
        % use CD learning
        R{l}.parallel_tempering.use = 0;

        % set the stopping criterion
        R{l}.stop.criterion = 1;
        R{l}.stop.recon_error.tolerate_count = 1000;
    else
        % use PT learning
        R{l}.parallel_tempering.use = 1;
        R{l}.parallel_tempering.n_chains = 21;

        % set the stopping criterion
        R{l}.stop.criterion = 1;
        R{l}.stop.recon_error.tolerate_count = 1000;
    end

    % max. 100 epochs
    R{l}.iteration.n_epochs = 100;

    % save the intermediate data after every epoch
    R{l}.hook.per_epoch = {@save_intermediate, {sprintf('rbm_mnist_layer%d.mat',l)}};

    % print learining process
    R{l}.verbose = 1;

    % display the progress
    R{l}.debug.do_display = 1;
    R{l}.debug.display_interval = 5;
    R{l}.debug.display_fid = 1;
    R{l}.debug.display_function = @visualize_rbm;

    % train RBM
    fprintf(1, 'Training a layer %d\n', l);
    tic;
    R{l} = train_rbm (R{l}, X_layer);
    fprintf(1, 'Training is done after %f seconds\n', toc);
end

% grab some samples from RBM
S = rbm_sample(binornd(1, 0.5, 10, R{end}.structure.n_visible), R{end}, 30, 1000);

S_down = cell(1, 10);
for t=1:10
    S_down{t} = squeeze(S(t,:,:))';
end

for l=(length(R) - 1):-1:1
    for t=1:10
        S_down{t} = rbm_get_visible(S_down{t},R{l});
    end
end

figure;
for i=1:10
    subplot(10, 1, i);
    visualize_adv(S_down{i}', 1, 1, 30, 0, 0);
end





