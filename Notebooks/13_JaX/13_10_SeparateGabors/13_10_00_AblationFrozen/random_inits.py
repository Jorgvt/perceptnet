from jax import random, numpy as jnp

def randomly_initialize_gabors(params, fs, key):
    key_f_a, key_f_t, key_f_d, key_t_a, key_t_t, key_t_d = random.split(key, 6)
    params_ = params.copy()
    ## 1. Random initializacions
    ### Freqs
    params_["freq_a"] = random.uniform(key=key_f_a, shape=params_["freq_a"].shape, minval=0, maxval=fs/2)
    params_["freq_t"] = random.uniform(key=key_f_t, shape=params_["freq_t"].shape, minval=0, maxval=fs/2)
    params_["freq_d"] = random.uniform(key=key_f_d, shape=params_["freq_d"].shape, minval=0, maxval=fs/2)
    ### Orients
    params_["theta_a"] = random.uniform(key=key_t_a, shape=params_["theta_a"].shape, minval=0, maxval=jnp.pi)
    params_["theta_t"] = random.uniform(key=key_t_t, shape=params_["theta_t"].shape, minval=0, maxval=jnp.pi)
    params_["theta_d"] = random.uniform(key=key_t_d, shape=params_["theta_d"].shape, minval=0, maxval=jnp.pi)

    ## 2. Conditional initializations
    ### Widths
    params_["gammax_a"] = 0.4*params_["freq_a"]**0.8
    params_["gammay_a"] = params_["gammax_a"]*0.8
    params_["gammax_t"] = 0.4*params_["freq_t"]**0.8
    params_["gammay_t"] = params_["gammax_t"]*0.8
    params_["gammax_d"] = 0.4*params_["freq_d"]**0.8
    params_["gammay_d"] = params_["gammax_d"]*0.8
    ### Gaussian orientation
    params_["sigma_theta_a"] = params_["theta_a"]
    params_["sigma_theta_t"] = params_["theta_t"]
    params_["sigma_theta_d"] = params_["theta_d"]
    
    return params_

def randomly_initialize_gdnfinal(params, fs, key):
    params_ = params.copy()
    key_f_a, key_f_t, key_f_d, key_t_a, key_t_t, key_t_d = random.split(key, 6)
    params_["ChromaFreqOrientGaussianGamma_0"]["gamma_f_a"] = random.uniform(key=key_f_a, shape=params_["ChromaFreqOrientGaussianGamma_0"]["gamma_f_a"].shape, maxval=1., minval=1/(fs/4))
    params_["ChromaFreqOrientGaussianGamma_0"]["gamma_f_t"] = random.uniform(key=key_f_t, shape=params_["ChromaFreqOrientGaussianGamma_0"]["gamma_f_t"].shape, maxval=1., minval=1/(fs/4))
    params_["ChromaFreqOrientGaussianGamma_0"]["gamma_f_d"] = random.uniform(key=key_f_d, shape=params_["ChromaFreqOrientGaussianGamma_0"]["gamma_f_d"].shape, maxval=1., minval=1/(fs/4))

    params_["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_a"] = random.uniform(key=key_t_a, shape=params_["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_a"].shape, maxval=1., minval=1/20)
    params_["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_t"] = random.uniform(key=key_t_t, shape=params_["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_t"].shape, maxval=1., minval=1/20)
    params_["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_d"] = random.uniform(key=key_t_d, shape=params_["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_d"].shape, maxval=1., minval=1/20)

    return params_ 
