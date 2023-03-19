classdef net_params
    properties
        use_bn          %%%% if you use Batch Normalization of not
        W               %%%% weight
        b               %%%% bias
        Grad_W          %%%% gradients of weight
        Grad_b          %%%% gradients of bias
        gammas          %%%% gammas of all layers
        betas           %%%% betas of all layers
        Grad_gm         %%%% gradients of gammas
        Grad_bt         %%%% gradients of betas
        n_v             %%%% normalized variance  
        n_mu            %%%% normalized mean
        un_v            %%%% unnormalized variance
        un_mu           %%%% unnormalized mean
        ave_v           %%%% moving average variance
        ave_mu          %%%% moving average mean
        alpha           %%%% moving average parameters alpha
    end
end