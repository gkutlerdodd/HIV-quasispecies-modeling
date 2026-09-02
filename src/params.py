"""
Default model parameters for simulation runs.
"""
def default_params():
    """Default parameters used in the model.
        **Target cell dynamics:**
        - lambda_T: production rate of target cells (cells/day)
        - d: target cell loss rate (1/day)

        **Infection dynamics:**
        - beta_wt: infection rate of the wild type virus (1/(virions*day))
        - gamma: progression rate through eclipse phase (1/day)
        - delta1: death rate of I1 cells (1/day)
        - delta2: death rate of I2 cells (1/day)
        - n: burst size (virions)
        - c: clearance rate of virus (1/day)

        **Binding affinity:**
        - RT: Thermal energy (kcal/mol)
        - KD_WT: Wild-type antibody dissociation constant (M)
        - h: saturation location parameter
        - K: saturation shape parameter

        **Antibody dynamics (optional):**
        - k: Fraction of fast-decaying component in biphasic decay
        - a1: Fast decay rate (1/day)
        - a2: Slow decay rate (1/day)
    """
    return {
        #Target cell dynamics
        "lambda_T": 1e7,
        "d": 0.2,

        #Infection dynamics
        "beta_wt": 2e-9,
        "gamma": 1.0,
        "delta1": 0.5,
        "delta2": 1.0,
        "n": 1000.0,
        "c": 20.0,

        #Binding affinity
        "RT": 0.616,
        "KD_WT": 1e-8,
        "h": 0.1,
        "K": 5.0,

        #Antibody dynamics
        "k": 0.664,
        "a1": 0.288,
        "a2": 0.070
    }