import uproot
import numpy as np
import numpy.lib.recfunctions as rfn

import datahandler as dh
from datahandler import DataHandler

def MeVtoGeV(array):
    """
    Convert unit from MeV to GeV

    Parameters
    ----------
    array : numpy structured array

    """
    for fname in list(array.dtype.names):
        # jet_pt, jet_e, met_met, mwt, lep_pt, lep_m
        isObjectVar = fname in ['jet_pt', 'jet_e', 'met_met', 'mwt', 'lep_pt', 'lep_m']
        # MC_*_afterFSR_[pt,m,E, Ht, pout]
        isPartonVar = fname.startswith('MC_') and (
            fname.endswith('_pt') or fname.endswith('_m') or
            fname.endswith('_E') or fname.endswith('_Ht') or
            fname.endswith('_pout')
            )

        if isObjectVar or isPartonVar:
            array[fname] /= 1000.

def setDummyValue(array, masks, dummy_value):
    """
    Set dummy value of entries in array that are masked by masks

    Parameters
    ----------
    array : numpy structured array
    masks : numpy ndarray
    dummy_value : float
    """

    for vname in list(array.dtype.names):
        if vname in ['isDummy', 'isMatched']:
            continue

        array[vname][masks] = dummy_value

def load_dataset_root(
        file_names,
        tree_name,
        variable_names = [],
        weight_name = None,
        dummy_value = None
    ):
    """
    Load data from a list of ROOT files
    Return a structured numpy array of data and a numpy ndarray as masks

    Parameters
    ----------
    file_names : str or file-like object; or sequence of str or file-like objects
        List of root files to load
    tree_name : str
        Name of the tree in root files
    variable_names : list of str, optional
        List of variables to read. If not provided, read all available variables
    weight_name : str, default None
        Name of the event weights. If not None, add it to the list of variables
    dummy_value : float, default None
        Dummy value for setting events that are flagged as dummy. If None, only
        include events that are not dummy in the array
    """
    if isinstance(file_names, str):
        file_names = [file_names]

    intrees = [fname + ':' + tree_name for fname in file_names]

    if variable_names:
        branches = list(variable_names)

        if weight_name:
            branches.append(weight_name)

        # flags for identifying events
        branches += ['isMatched', 'isDummy']

        # in case of KLFitter
        branches.append('klfitter_logLikelihood')

        # for checking invalid value in truth trees
        branches.append('MC_thad_afterFSR_y')
    else:
        # load everything
        branches = None

    data_array = uproot.lazy(intrees, filter_name=branches)
    # variables in branches but not in intrees are ignored

    # convert awkward array to numpy array for now
    # can probably use awkward array directly once it is more stable
    data_array = data_array.to_numpy()

    # convert units
    MeVtoGeV(data_array)

    #####
    # event selection flag
    pass_sel = (data_array['isDummy'] == 0) & (data_array['isMatched'] == 1)

    # In case of reco variables with KLFitter, cut on KLFitter logLikelihood
    if 'klfitter_logLikelihood' in data_array.dtype.names:
        pass_sel &= data_array['klfitter_logLikelihood'] > -52.

    # A special case where some matched events in truth trees contain invalid
    # (nan or inf) values
    if 'MC_thad_afterFSR_y' in data_array.dtype.names:
        invalid = np.isnan(data_array['MC_thad_afterFSR_y'])
        pass_sel &= (~invalid)
        #print("number of events with invalid value", np.sum(invalid))

    # TODO: return a masked array?
    return data_array, pass_sel

class DataHandlerROOT(DataHandler):
    """
    Load data from root files

    Parameters
    ----------
    filepaths :  str or sequence of str
        List of root file names to load
    treename_reco : str, default 'reco'
        Name of the reconstruction-level tree
    treename_truth : str, default 'parton'
        Name of the truth-level tree. If empty or None, skip loading the 
        truth-level tree
    variable_names : list of str, optional
        List of reco level variable names to read. If not provided, read all
    variable_names_mc : list of str, optional
        List of truth level variable names to read. If not provided, read all
    dummy_value : float, default None
        Dummy value for setting events that are flagged as dummy. If None, only
        include events that are not dummy in the array
    """
    def __init__(
        self,
        filepaths,
        treename_reco='reco',
        treename_truth='parton',
        variable_names=[],
        variable_names_mc=[],
        weights_name=None, #"normedWeight",
        weights_name_mc=None, #"weight_mc",
        dummy_value=None
    ):
        # load data from root files
        variable_names = dh._filter_variable_names(variable_names)
        self.data_reco, self.pass_reco = load_dataset_root(
            filepaths, treename_reco, variable_names, weights_name, dummy_value
            )

        # event weights
        if weights_name:
            self.weights = self.data_reco[weights_name]
        else:
            self.weights = np.ones(len(self.data_reco))

        # truth variables if available
        if treename_truth:
            variable_names_mc = dh._filter_variable_names(variable_names_mc)
            self.data_truth, self.pass_truth = load_dataset_root(
                filepaths, treename_truth, variable_names_mc, weights_name_mc,
                dummy_value
            )

            # rename fields of the truth array if the name is already in the reco
            # array
            prefix = 'truth_'
            newnames = {}
            for fname in self.data_truth.dtype.names:
                if fname in self.data_reco.dtype.names:
                    newnames[fname] = prefix+fname
            self.data_truth = rfn.rename_fields(self.data_truth, newnames)

            # mc weights
            if weights_name_mc:
                self.weights_mc = self.data_truth[weights_name_mc]
            else:
                self.weights_mc = np.ones(len(self.data_truth))
        else:
            self.data_truth = None
            self.weights_mc = None

        # deal with events that fail selections
        if dummy_value is None:
            # include only events that pass all selections
            if self.data_truth is not None:
                pass_all = self.pass_reco & self.pass_truth
                self.data_reco = self.data_reco[pass_all]
                self.data_truth = self.data_truth[pass_all]
                self.weights = self.weights[pass_all]
                self.weights_mc = self.weights_mc[pass_all]
            else:
                self.data_reco = self.data_reco[self.pass_reco]
                self.weights = self.weights[self.pass_reco]
        else:
            # set dummy value
            dummy_value = float(dummy_value)
            setDummyValue(self.data_reco, ~self.pass_reco, dummy_value)
            if self.data_truth is not None:
                setDummyValue(self.data_truth, ~self.pass_truth, dummy_value)

        # sanity check
        assert(len(self.data_reco)==len(self.weights))
        if self.data_truth is not None:
            assert(len(self.data_reco)==len(self.data_truth))
            assert(len(self.data_truth)==len(self.weights_mc))

    def __len__(self):
        """
        Get the number of events in the dataset.

        Returns
        -------
        non-negative int
        """
        return len(self.data_reco)

    def __contains__(self, variable):
        """
        Check if a variable is in the dataset.

        Parameters
        ----------
        variable : str

        Returns
        -------
        bool
        """
        inReco = variable in self.data_reco.dtype.names

        if self.data_truth is None:
            inTruth = False
        else:
            inTruth = variable in self.data_truth.dtype.names

        return inReco or inTruth

    def __iter__(self):
        """
        Create an iterator over the variable names in the dataset.

        Returns
        -------
        iterator of strings
        """
        if self.data_truth is None:
            return iter(self.data_reco.dtype.names)
        else:
            return iter(
                list(self.data_reco.dtype.names) +
                list(self.data_truth.dtype.names)
                )

    def _get_array(self, variable):
        """
        Return a 1D numpy array of the variable

        Parameters
        ----------
        variable : str
            Name of the variable

        Returns
        -------
        np.ndarray of shape (n_events,)
        """
        if variable in self.data_reco.dtype.names:
            return self.data_reco[str(variable)]

        if self.data_truth is not None:
            if variable in self.data_truth.dtype.names:
                return self.data_truth[str(variable)]

        # no 'variable' in the data arrays
        return None
