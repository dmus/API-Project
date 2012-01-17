costs = [];
cand_lrates = [];
cand_lrate = base_lrate;
for s=1:(R.adaptive_lrate.max_iter_up + 1)
    cand_lrates = [cand_lrates cand_lrate];
    cand_lrate = cand_lrate * R.adaptive_lrate.exp_up;
end
cand_lrate = base_lrate * R.adaptive_lrate.exp_down;
for s=1:(R.adaptive_lrate.max_iter_down)
    cand_lrates = [cand_lrates cand_lrate];
    cand_lrate = cand_lrate * R.adaptive_lrate.exp_down;
end


