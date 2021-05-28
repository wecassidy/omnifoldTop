"""
Reweight the events prior to unfolding. Use the `@reweighter`
decorator to register a new reweighter to the `rw` dict.

Example:

    import reweight
    @reweight.reweighter("th_pt", "mtt")
    def reweighter_name(th_pt, mtt):
        return th_pt * mtt

Then:

    import reweight
    reweight.rw["reweighter_name](datahandler, vars_dict)
"""

import functools

rw = dict()

def reweighter(*observables):
    def rw_decorator(rw_func):
        @functools.wraps(rw_func)
        def do_rw(handler, vars_dict):
            assert handler.truth_known
            data = []
            for v in observables:
                assert v in vars_dict
                varname = vars_dict[v]["branch_mc"]
                data.append(handler.get_variable_arr(varname))
            return rw_func(*data)
        rw[rw_func.__name__] = do_rw
        return do_rw

    return rw_decorator

@reweighter("th_pt")
def linear_th_pt(th_pt):
    return 1 + th_pt / 800

@reweighter("mtt")
def gaussian_bump(mtt):
    k = 0.5
    m0 = 800
    sigma = 100
    return 1 + k*np.exp( -( (mtt-m0)/sigma )**2 )

@reweighter("mtt")
def gaussian_tail(mtt):
    k = 0.5
    m0 = 2000
    sigma = 1000
    return 1 + k*np.exp( -( (mtt-m0)/sigma )**2 )

@reweighter("mtt", "ytt")
def linear_two_dim(mtt, ytt):
    return 2 + mtt / 300 + 2 * ytt

@reweighter("mtt", "ytt")
def linear_2d_small(mtt, ytt):
    return 2 +  mtt / 300 + ytt / 700
