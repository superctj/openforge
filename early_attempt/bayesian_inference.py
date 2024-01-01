import argparse
import pickle
import random

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
# import pymc.sampling.jax as pmjax

# from sklearn.model_selection import train_test_split

# from prepare_bayesian_prior_model import RidgeClassifierwithProba # this is needed for unpickling the model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--arts_data", type=str, default="./data/arts_num-head-concepts-100.pkl", help="Path to the training and test data synthesized from ARTS.")

    # parser.add_argument("--arts_test_size", type=float, default=0.25, help="Test size for the ARTS data.")

    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed.")

    parser.add_argument("--evidence_filepath", type=str, default="./data/arts_top-100-concepts_evidence.pkl", help="Path to the observed evidence.")

    parser.add_argument("--prior_model_path", type=str, default="./models/rc_arts_num-head-concepts-100.pkl", help="Path to a machine learning model that gives prior probabilities.")

    parser.add_argument("--sample_size", type=int, default=10000, help="Number of samples to draw from each chain of the posterior.")

    parser.add_argument("--num_chains", type=int, default=4, help="Number of chains in the sampler.")

    args = parser.parse_args()
    random.seed(args.random_seed)

    # with open(args.arts_data, "rb") as f:
    #     arts_data = pickle.load(f)
    #     x_arts = np.array([x[0] for x in arts_data])
    #     y_arts = np.array([x[1] for x in arts_data])
    #     print(x_arts.shape)
    #     print(y_arts.shape)
    
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x_arts, y_arts, test_size=args.arts_test_size, random_state=args.random_seed, stratify=y_arts)
    # assert(x_arts.shape[1] == x_train.shape[1] == x_test.shape[1])
    # domain_dim = x_arts.shape[1]
    # n_train = x_train.shape[0]
    # n_test = x_test.shape[0]
    # idx_train = range(n_train)
    # idx_test = range(n_train, n_train + n_test)

    with open(args.evidence_filepath, "rb") as f:
        evidence = pickle.load(f)
        print("Number of relation variables:", len(evidence))
    
    e1_values = []
    e2_values = []
    for e in evidence:
        e1_values.append(e[1][0])
        e2_values.append(e[1][1])
    e1_values = np.array(e1_values)
    e2_values = np.array(e2_values)
    # x_values = np.column_stack((e1_values, e2_values))

    with open(args.prior_model_path, "rb") as f:
        prior_model = pickle.load(f)

    # Fit a GP model as the prior
    # gp_model = pm.Model(coords={"domain_dim": range(domain_dim)})
    # with gp_model:
    #     gp_model.add_coord(name="idx", values=idx_train, mutable=True)

    #     ls = pm.Gamma(name="ls", alpha=2, beta=1, dims="domain_dim") # length scale
    #     amplitude = pm.Gamma(name="amplitude", alpha=2, beta=1)
    #     cov = amplitude ** 2 * pm.gp.cov.ExpQuad(input_dim=2, ls=ls)

    #     gp= pm.gp.Latent(cov_func=cov)
    #     f = gp.prior(name="f", X=x_train, dims="idx")
    #     p = pm.Deterministic(name="p", var=pm.math.invlogit(f), dims="idx")
    #     # likelihood
    #     likelihood = pm.Bernoulli(name="likelihood", p=p, dims="idx", observed=y_train)

    #     gp_idata = pmjax.sample_numpyro_nuts(draws=100, chains=1)
    #     gp_posterior_predictive = pm.sample_posterior_predictive(trace=gp_idata)

    # with gp_model:
    #     f_pred = gp.conditional(name="f_pred", Xnew=x_test)
    #     p_pred = pm.Deterministic(name="p_pred", var=pm.math.invlogit(f_pred))
    #     likelihood_pred = pm.Bernoulli(name="likelihood_pred", p=p_pred)
    #     gp_posterior_predictive_test = pm.sample_posterior_predictive(
    #         trace=gp_idata, var_names=["f_pred", "p_pred", "likelihood_pred"]
    #     )

    # gp_p_pred_samples = gp_posterior_predictive_test.posterior_predictive["p_pred"].stack(sample=("chain", "draw"))
    # gp_pred_test = np.random.binomial(n=1, p=gp_p_pred_samples)
    # print(gp_pred_test)

    with pm.Model() as model:
        e1 = pm.Normal("e1", mu=0, sigma=1, shape=len(evidence), observed=e1_values)
        e2 = pm.Normal("e2", mu=0, sigma=1, shape=len(evidence),observed=e2_values)
        # print(type(e1))
        # print(type(e1.eval()))

        ml_prior = prior_model.predict_proba(np.column_stack((e1.eval(), e2.eval())))[:, 1]

        r = pm.Bernoulli("r", p=ml_prior, shape=len(evidence))

        # Use the Metropolis-Hastings sampler
        # step = pm.Metropolis()
        step = pm.BinaryMetropolis([r])
        # step = pm.BinaryGibbsMetropolis([r])

        # Run the sampler
        trace = pm.sample(args.sample_size, step=step, chains=args.num_chains, random_seed=args.random_seed)

    save_filename = f"bM_trace_num-samples-{args.sample_size}_num-chains-{args.num_chains}"
    # pm.summary(trace, round_to=2, to_file=f"./pymc_results/{save_filename}.csv")
    axes = az.plot_trace(trace, combined=True)
    fig = axes.ravel()[0].figure
    fig.savefig(f"./pymc_results/{save_filename}.png")

    sample_counts = {}
    for i in range(args.num_chains):
        for sample in trace.posterior.r.values[i]:
            sample_str = "".join([str(int(s)) for s in sample])
            if sample_str not in sample_counts:
                sample_counts[sample_str] = 1
            else:
                sample_counts[sample_str] += 1
    
    print("Number of distinct samples", len(sample_counts))
    print(sample_counts)
