#!/bin/env python3
"""
A simple script to provide updated UVEX ToO trigger/success rate estimates under a changed BNS rate.

"""

import numpy as np
import scipy.stats as st
from scipy import special, integrate, optimize
import argparse



## support functions, sourced from the code for the LK Observing Guide
@np.vectorize
def poisson_lognormal_rate_cdf(k, mu, sigma,duration=1):
    lognorm_pdf = st.lognorm(s=sigma, scale=np.exp(mu)).pdf

    def func(lam):
        prior = lognorm_pdf(lam)
        ## lam is rate * 1 yr; lam_adj is rate * duration
        lam_adj = lam * duration
        poisson_pdf = np.exp(special.xlogy(k, lam_adj) - special.gammaln(k + 1) - lam_adj)
        poisson_cdf = special.gammaincc(k + 1, lam_adj)
        return poisson_cdf * prior

    # Marginalize over lambda.
    #
    # Note that we use scipy.integrate.odeint instead
    # of scipy.integrate.quad because it is important for the stability of
    # root_scalar below that we calculate the pdf and the cdf at the same time,
    # using the same exact quadrature rule.
    cdf, _ = integrate.quad(func, 0, np.inf, epsabs=0)
    return cdf


@np.vectorize
def poisson_lognormal_rate_quantiles(p, mu, sigma, duration=1):
    """Find the quantiles of a Poisson distribution with
    a log-normal prior on its rate.

    Parameters
    ----------
    p : float
        The quantiles at which to find the number of counts.
    mu : float
        The mean of the log of the rate.
    sigma : float
        The standard deviation of the log of the rate.

    Returns
    -------
    k : float
        The number of events.

    Notes
    -----
    This algorithm treats the Poisson count k as a continuous
    real variable so that it can use the scipy.optimize.root_scalar
    root finding/polishing algorithms.
    """
    def func(k):
        return poisson_lognormal_rate_cdf(k, mu, sigma, duration) - p

    if func(0) >= 0:
        return 0

    result = optimize.root_scalar(func, bracket=[0, 1e6])
    return result.root

def f2s(val):
    '''
    Simple function to convert a float to a string with 1 decimal place.
    '''
    return "{:0.1f}".format(val)


## go from triggers to successful observations, given a success rate
    
def triggers2observations(triggers,success_rate,duration,
                          bns_rate=[210,90,450],fmt_in='list',fmt_out='list'):
    '''
    Simple function to compute an estimated number of successful ToO observations, 
    given a ToO trigger count (with error bars), a success rate, the duration of the simulated run,
    and a BNS merger rate estimate.
    
    Arguments
    ----------------------
    triggers (str or list; see fmt_in) : Estimated trigger count, with 90 percent C.I..
    success_rate (float)               : Overall success rate of the observing strategy.
    duration (float)                   : Duration of the observing run in years.
    bns_rate (list of floats)          : [Median, lower 90 percent C.I., upper 90 percent C.I.] 
                                          of the BNS rate used in the original ToO count estimate
    fmt_in (str)                       : 'latex' or 'list'. If latex, the function will try to parse a latex string
                                          containing the trigger count. If 'list', the function will instead
                                          try to parse a list of floats as [median, lower90, upper90].
    fmt_out (str)                      : Output format. If 'latex', will output a latex string; if 'list',
                                          will return a list of floats.
                                          
    Returns
    ---------------
    successful_observations (str or list; see fmt_out) : Estimate (with 90 percent C.I.) of the 
                                                          number of successful ToO observations.
    '''
    ## parse input
    if fmt_in=='latex':
        print("Warning: Usinng the LaTeX string may induce a small amount of rounding error. Use fmt_in='list' with full precision for production applications.")
        trigger_med = float(triggers.split('^')[0])
#         trigger_lower90 = float(triggers.split('^')[1].split('_')[0].repace('{+','').replace('}',''))
#         trigger_upper90 = float(triggers.split('^')[1].split('_')[1].repace('{-','').replace('}',''))
    elif fmt_in=='list':
        trigger_med, trigger_lower90, trigger_upper90 = triggers
    else:
        raise ValueError("fmt_in must be 'latex' or 'list'.")
    
    ## now recompute lognormal + Poisson errors with the updated median rate
    mu = np.log(bns_rate[0])
    sigma = (np.log(bns_rate[2]) - np.log(bns_rate[1]))/np.diff(st.norm.interval(0.9))
    ## effective duration is modified by the efficiency
    dur_eff = duration*success_rate
    observed_med = bns_rate[0]*dur_eff
    observed_low, observed_high = poisson_lognormal_rate_quantiles([0.05,0.95],mu,sigma, duration=dur_eff)
    
    ## normalize by trigger count from simulations
    sim_norm = trigger_med/(duration*bns_rate[0])
    
    result_med, result_low, result_high = sim_norm*observed_med, sim_norm*observed_low, sim_norm*observed_high
    
    error_high = f2s(result_high-result_med)
    error_low = f2s(result_med-result_low)
    
    if fmt_out == 'latex':
        return f2s(result_med)+"^{+"+error_high+"}_{-"+error_low+"}"
    elif fmt_out == 'list':
        return [result_med, result_low, result_high]
    else:
        raise ValueError("fmt_in must be 'latex' or 'list'.")
    

## workhorse function
def update_trigger_estimates(triggers,bns_rate_new,duration=1.5,bns_rate_old=[210,90,450],get_obs=False,success_rate=None,fmt_in='list',fmt_out='list'):
    '''
    Simple function to reweight a UVEX trigger count estimate to account for an updated BNS rate.
    
    Arguments
    ------------------------
    triggers (str or list; see fmt_in) : Estimated trigger count, with 90 percent C.I..
    bns_rate_new (list of floats)      : Updated BNS rate, given as [median, lower 90 percent CI, upper 90 percent CI]
    
    duration (float)                   : Duration of the simulated run in years. Defaults to 1.5 years (value considered in Criswell+24).
    bns_rate_old (list of floats)      : Original BNS rate used for the estimates. Same format as bns_rate_new. Defaults to the GWTC-3 rate.
    get_obs (bool)                     : If True, also compute an estimate of successful KN counterpart observations, under the assumptions stated in Criswell+24. Requires success_rate if True. Default False.
    success_rate (float)               : Overall success rate of the observing strategy. Should be between 0 and 1. Required if get_obs==True.
    fmt_in (str)                       : 'latex' or 'list'. If latex, the function will try to parse a latex string
                                          containing the trigger count. If 'list', the function will instead
                                          try to parse a list of floats as [median, lower90, upper90].
    fmt_out (str)                      : Output format. If 'latex', will output a latex string; if 'list',
                                          will return a list of floats.
    
    Returns
    -------------------------
    new_trigger_estimate (str or list; see fmt_out)    : Updated estimate (with 90 percent C.I.) of the number of ToO triggers.
    
    (if get_obs==True) 
    successful_observations (str or list; see fmt_out) : Updated estimate (with 90 percent C.I.) of the number of successful ToO observations.
    
    '''
    
    
    ## parse input
    if fmt_in=='latex':
        print("Warning: Usinng the LaTeX string may induce a small amount of rounding error. Use fmt_in='list' with full precision for production applications.")
        trigger_med = float(triggers.split('^')[0])

    elif fmt_in=='list':
        trigger_med, trigger_lower90, trigger_upper90 = triggers
    else:
        raise ValueError("fmt_in must be 'latex' or 'list'.")
    
    
    ## normalize trigger count
    trigger_med_norm = trigger_med/(duration*bns_rate_old[0])
    
    
    ## now recompute lognormal + Poisson errors with the updated median rate
    mu = np.log(bns_rate_new[0])
    sigma = (np.log(bns_rate_new[2]) - np.log(bns_rate_new[1]))/np.diff(st.norm.interval(0.9))

    update_med = bns_rate_new[0]*duration
    update_low, update_high = poisson_lognormal_rate_quantiles([0.05,0.95],mu,sigma, duration=duration)
    
    updated_trigger_med, updated_trigger_low, updated_trigger_high = trigger_med_norm*update_med, trigger_med_norm*update_low, trigger_med_norm*update_high
    updates = [updated_trigger_med, updated_trigger_low, updated_trigger_high]
    
    error_high = f2s(updated_trigger_high-updated_trigger_med)
    error_low = f2s(updated_trigger_med-updated_trigger_low)
    
    if fmt_out == 'latex':
        trigger_result = f2s(updated_trigger_med)+"^{+"+error_high+"}_{-"+error_low+"}"
    elif fmt_out == 'list':
        trigger_result = updates
    else:
        raise ValueError("fmt_in must be 'latex' or 'list'.")
    
    if get_obs:
        if success_rate is None:
            raise TypeError("If get_obs is set to True, you must specify success_rate.")
        
        observation_result = triggers2observations(updates,success_rate,duration,bns_rate=bns_rate_new,fmt_in='list',fmt_out=fmt_out)
        
        return trigger_result, observation_result
    else:
        return trigger_result
    







if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='uvex_rate_updater', usage='%(prog)s [options] trigger_estimate new_bns_rate', description='Recompute UVEX trigger rate estimates based on an updated BNS rate.')

    # Add arguments
    parser.add_argument('trigger_estimate', metavar='trigger_estimate', type=float, help='The original median trigger estimate. You do not need the original error bars.')
    parser.add_argument('new_bns_rate', metavar='new_bns_rate', type=str, help='The desired updated BNS rate, in Gpc^-3 yr^-1, given as [median, lower 90 percent CI, upper 90 percent CI]. (Also accepted: "gwtc-4", which will use the GWTC-4.0 FullPop-4.0 rate.)')

    parser.add_argument('--get_obs', action='store_true', help="Whether to also provided an updated estimate of successful KN counterpart observations. Requires --success_rate. Default False.")

    
    parser.add_argument('--old_bns_rate', type=str, default="[210,90,450]", help="The original BNS rate, in Gpc^-3 yr^-1, given as [median, lower 90 percent CI, upper 90 percent CI]. Defaults to the GWTC-3 rate ([210,90,450]).")
    parser.add_argument('--duration', type=float, default=1.5, help="Duration of the observing run in years. Default 1.5 years (as used in Criswell+24).")
    parser.add_argument('--success_rate', type=float, default=None, help="ToO strategy success rate (see Criswell+24). Required if get_obs is set.")
    parser.add_argument('--fmt_out', type=str, default='latex', help="Output format ('latex' or 'list'). 'latex' returns a LaTeX string, 'list' returns [median, lower 90 percent CI, upper 90 percent CI]. Default 'latex'.")
    
    # execute parser
    args = parser.parse_args()
    
    ## GWTC-4.0 BNS rate
    if args.new_bns_rate == 'gwtc-4':
        args.new_bns_rate = "[89, 22, 248]"
    
    ## run workhorse function
    result = update_trigger_estimates([args.trigger_estimate,None,None],eval(args.new_bns_rate),duration=args.duration,bns_rate_old=eval(args.old_bns_rate),get_obs=args.get_obs,success_rate=args.success_rate,fmt_in='list',fmt_out=args.fmt_out)

    ## display results
    if args.get_obs:
        if args.fmt_out=='list':
            print("The estimated number of ToO triggers in {:0.1f} year(s) is {} ([median, lower 90 percent CI, upper 90 percent CI]).".format(args.duration,result[0]))
            print("With a strategy success rate of {}%, this corresponds to an estimated {} ([median, lower 90 percent CI, upper 90 percent CI]) successful KN observations.".format(args.success_rate,result[1]))
        elif args.fmt_out=='latex':
            print("The estimated number of ToO triggers in {:0.1f} year(s) is ".format(args.duration)+result[0])
            print("With a strategy success rate of {}%, this corresponds to an estimated ".format(args.success_rate)+result[1]+" successful KN observations.")
        else:
            raise ValueError("fmt_out must be 'list' or 'latex'.")
    else:
        if args.fmt_out=='list':
            print("The estimated number of ToO triggers in {:0.1f} year(s) is {} ([median, lower 90 percent CI, upper 90 percent CI]).".format(args.duration,result))
        elif args.fmt_out=='latex':
            print("The estimated number of ToO triggers in {:0.1f} year(s) is ".format(args.duration)+result)


















