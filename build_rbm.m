function rbm = build_rbm(X, num_features)
%BUILDMODEL Builds and trains a RBM
%   Detailed explanation goes here

    % shuffle the training data
    perm_idx = randperm (size(X,1));
    X = X(perm_idx, :);

    % construct RBM and use default configurations
    R = default_rbm (size(X, 2), num_features);

    % use continuous values
    R.data.binary = 0;

    % set grbm parameters
    R.grbm.do_vsample = 1;
    R.grbm.do_normalize = 1;
    R.grbm.do_normalize_std = 1;
    R.grbm.learn_sigmas = 1;

    % max. 100 epochs
    R.iteration.n_epochs = 100;

    % set the stopping criterion
    R.stop.criterion = 1;
    R.stop.recon_error.tolerate_count = 1000;

    % save the intermediate data after every epoch
    R.hook.per_epoch = {@save_intermediate, {'grbm_faces.mat'}};

    % print learining process
    R.verbose = 1;

    % display the progress
    R.debug.do_display = 0;
    R.debug.display_interval = 10;
    R.debug.display_fid = 1;
    R.debug.display_function = @visualize_grbm;

    % train RBM
    fprintf(1, 'Training GB-RBM\n');
    tic;
    rbm = train_rbm (R, X);
    fprintf(1, 'Training is done after %f seconds\n', toc);
end

