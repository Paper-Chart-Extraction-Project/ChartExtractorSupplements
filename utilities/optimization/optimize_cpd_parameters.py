"""A command line program to optimize the parameters of the coherent point drift algorithm.

The code that performs CPD here is from the pycpd package by Siavash Khallaghi. I copy pasted
it here instead of importing to reduce dependencies and also to prepare for using a Rust
implementation in the future.

The optimization is handled by the Optuna package.
"""

from __future__ import division
import argparse
from builtins import super
from collections import namedtuple
from functools import partial
import json
import numbers
import numpy as np
import optuna
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional
from warnings import warn


# pycpd code


def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError(
            "Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}".format(R)
        )
    return np.all(np.linalg.eigvals(R) > 0)


def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :, :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))


def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


def initialize_sigma2(X, Y):
    """
    Initialize the variance (sigma2).

    Attributes
    ----------
    X: numpy array
        NxD array of points for target.

    Y: numpy array
        MxD array of points for source.

    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff**2
    return np.sum(err) / (D * M * N)


def lowrankQS(G, beta, num_eig, eig_fgt=False):
    """
    Calculate eigenvectors and eigenvalues of gaussian matrix G.

    !!!
    This function is a placeholder for implementing the fast
    gauss transform. It is not yet implemented.
    !!!

    Attributes
    ----------
    G: numpy array
        Gaussian kernel matrix.

    beta: float
        Width of the Gaussian kernel.

    num_eig: int
        Number of eigenvectors to use in lowrank calculation of G

    eig_fgt: bool
        If True, use fast gauss transform method to speed up.
    """

    # if we do not use FGT we construct affinity matrix G and find the
    # first eigenvectors/values directly

    if eig_fgt is False:
        S, Q = np.linalg.eigh(G)
        eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
        Q = Q[:, eig_indices]  # eigenvectors
        S = S[eig_indices]  # eigenvalues.

        return Q, S

    elif eig_fgt is True:
        raise Exception("Fast Gauss Transform Not Implemented!")


class EMRegistration(object):
    """
    Expectation maximization point cloud registration.

    Attributes
    ----------
    X: numpy array
        NxD array of target points.

    Y: numpy array
        MxD array of source points.

    TY: numpy array
        MxD array of transformed source points.

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    N: int
        Number of target points.

    M: int
        Number of source points.

    D: int
        Dimensionality of source and target points

    iteration: int
        The current iteration throughout registration.

    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.

    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    q: float
        The objective function value that represents the misalignment between source
        and target point clouds.

    diff: float (positive)
        The absolute difference between the current and previous objective function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    """

    def __init__(
        self,
        X,
        Y,
        sigma2=None,
        max_iterations=None,
        tolerance=None,
        w=None,
        *args,
        **kwargs,
    ):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("The target point cloud (X) must be at a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions."
            )

        if sigma2 is not None and (
            not isinstance(sigma2, numbers.Number) or sigma2 <= 0
        ):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2)
            )

        if max_iterations is not None and (
            not isinstance(max_iterations, numbers.Number) or max_iterations < 0
        ):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(
                    max_iterations
                )
            )
        elif isinstance(max_iterations, numbers.Number) and not isinstance(
            max_iterations, int
        ):
            warn(
                "Received a non-integer value for max_iterations: {}. Casting to integer.".format(
                    max_iterations
                )
            )
            max_iterations = int(max_iterations)

        if tolerance is not None and (
            not isinstance(tolerance, numbers.Number) or tolerance < 0
        ):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(
                    tolerance
                )
            )

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(
                    w
                )
            )

        self.X = X
        self.Y = Y
        self.TY = Y
        self.sigma2 = initialize_sigma2(X, Y) if sigma2 is None else sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N,))
        self.P1 = np.zeros((self.M,))
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0

    def register(self, callback=lambda **kwargs: None):
        """
        Perform the EM registration.

        Attributes
        ----------
        callback: function
            A function that will be called after each iteration.
            Can be used to visualize the registration process.

        Returns
        -------
        self.TY: numpy array
            MxD array of transformed source points.

        registration_parameters:
            Returned params dependent on registration method used.
        """
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {
                    "iteration": self.iteration,
                    "error": self.q,
                    "X": self.X,
                    "Y": self.TY,
                }
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes."
        )

    def update_transform(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes."
        )

    def transform_point_cloud(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes."
        )

    def update_variance(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes."
        )

    def iterate(self):
        """
        Perform one iteration of the EM algorithm.
        """
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        P = np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)  # (M, N)
        P = np.exp(-P / (2 * self.sigma2))
        c = (
            (2 * np.pi * self.sigma2) ** (self.D / 2)
            * self.w
            / (1.0 - self.w)
            * self.M
            / self.N
        )

        den = np.sum(P, axis=0, keepdims=True)  # (1, N)
        den = np.clip(den, np.finfo(self.X.dtype).eps, None) + c

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()


class DeformableRegistration(EMRegistration):
    """
    Deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.

    low_rank: bool
        Whether to use low rank approximation.

    num_eig: int
        Number of eigenvectors to use in lowrank calculation.
    """

    def __init__(
        self, alpha=None, beta=None, low_rank=False, num_eig=100, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(
                    alpha
                )
            )

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(
                    beta
                )
            )

        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1.0 / self.S)
            self.S = np.diag(self.S)
            self.E = 0.0

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        if self.low_rank is False:
            A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(
                self.M
            )
            B = self.PX - np.dot(np.diag(self.P1), self.Y)
            self.W = np.linalg.solve(A, B)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.matmul(dP, self.Y)

            self.W = (
                1
                / (self.alpha * self.sigma2)
                * (
                    F
                    - np.matmul(
                        dPQ,
                        (
                            np.linalg.solve(
                                (
                                    self.alpha * self.sigma2 * self.inv_S
                                    + np.matmul(self.Q.T, dPQ)
                                ),
                                (np.matmul(self.Q.T, F)),
                            )
                        ),
                    )
                )
            )
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(
                np.matmul(QtW.T, np.matmul(self.S, QtW))
            )

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.

        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.


        """
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + np.dot(G, self.W)
        else:
            if self.low_rank is False:
                self.TY = self.Y + np.dot(self.G, self.W)

            elif self.low_rank is True:
                self.TY = self.Y + np.matmul(
                    self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W))
                )
                return

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        self.q = np.inf

        xPx = np.dot(
            np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1)
        )
        yPy = np.dot(
            np.transpose(self.P1), np.sum(np.multiply(self.TY, self.TY), axis=1)
        )
        trPXY = np.sum(np.multiply(self.TY, self.PX))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.


        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.

        self.W: numpy array
            Deformable transformation matrix.
        """
        return self.G, self.W


# Custom argparse type converter
def validate_file(path_str: str, must_already_exist: bool) -> Path:
    """Validates and creates a path object from a string."""
    path: Path = Path(path_str)
    if must_already_exist and not path.exists():
        raise FileNotFoundError(f"No such file or directory '{path.resolve()}'")
    if path.is_dir():
        raise FileNotFoundError(
            f"'{path.resolve()}' is a directory, but a file was expected."
        )
    return path


# Parse command line args
parser = argparse.ArgumentParser(
    description="Optimize the coherent point drift parameters for a label studio dataset."
)
parser.add_argument(
    "path_to_labels",
    type=partial(validate_file, must_already_exist=True),
    help="The filepath to the labelstudio json-min file.",
)
parser.add_argument(
    "path_to_output",
    type=partial(validate_file, must_already_exist=False),
    help="The output file to write the best parameters to.",
)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1000,
    help="The number of trials for the optimization.",
)
alpha_upper_bound_help_text: str = "The upper bound to search for the alpha value "
alpha_upper_bound_help_text += "(called lambda in the paper). Defaults to 1e20."
parser.add_argument(
    "--alpha_upper_bound",
    type=int,
    default=1e20,
    help=alpha_upper_bound_help_text,
)
alpha_lower_bound_help_text: str = "The lower bound to search for the alpha value "
alpha_lower_bound_help_text += "(called lambda in the paper). Defaults to 1e-20."
parser.add_argument(
    "--alpha_lower_bound",
    type=int,
    default=1e-20,
    help=alpha_lower_bound_help_text,
)
parser.add_argument(
    "--beta_upper_bound",
    type=int,
    default=1e5,
    help="The upper bound to search for the beta value. Defaults to 1e5.",
)
parser.add_argument(
    "--beta_lower_bound",
    type=int,
    default=1e-5,
    help="The upper bound to search for the beta value. Defaults to 1e-5.",
)

args = parser.parse_args()

# Optimization logic
NamedPoint = namedtuple("NamedPoint", "name x y")
image_name_regex_pattern: str = r"RC_\d\d\d\d_intraoperative.JPG"
scanned_chart_regex_pattern: str = (
    r"unified_intraoperative_preoperative_flowsheet_v1_1_(front|back)\.(png|jpg)"
)


def read_json_data(path: Path) -> Dict:
    """Reads a json file."""
    return json.loads(open(str(path), "r").read())


def extract_pattern_from_string(s: str, pattern: str) -> Optional[str]:
    """Extracts a regex pattern from a string."""
    try:
        return re.findall(pattern, s)[0]
    except:
        return None


def extract_image_name_from_string(s: str) -> str:
    """Extracts the image name from a string."""
    image_name: str = extract_pattern_from_string(s, image_name_regex_pattern)
    if image_name is not None:
        return image_name

    scanned_chart_name: Optional[str] = extract_pattern_from_string(
        s, scanned_chart_regex_pattern
    )
    if scanned_chart_name is not None:
        return scanned_chart_name

    err_msg: str = f"name {s} does not conform to either the image name "
    err_msg += "regex pattern, or the scanned chart regex pattern."
    raise ValueError(err_msg)


def convert_label_dict_to_namedpoint(lab_dict: Dict) -> NamedPoint:
    """Converts the 'label' dictionary from label-studio json-min format to a NamedPoint."""
    return NamedPoint(
        name=lab_dict["rectanglelabels"][0],
        x=(lab_dict["x"] + 0.5 * lab_dict["width"]) / 100,
        y=(lab_dict["y"] + 0.5 * lab_dict["height"]) / 100,
    )


def named_point_list_to_numpy_array(named_points: List[NamedPoint]) -> np.array:
    """Converts a list of NamedPoints to a numpy array of [x, y] points."""
    return np.array([[np.x, np.y] for np in named_points])


def distance(np1: NamedPoint, np2: NamedPoint) -> float:
    """Calculates the euclidean distance between two NamedPoints."""
    return np.sqrt((np1.x - np2.x) ** 2 + (np1.y - np2.y) ** 2)


def create_deformable_registration(
    X: List[NamedPoint],
    Y: List[NamedPoint],
    alpha: float,
    beta: float,
) -> DeformableRegistration:
    """Creates a DeformableRegistration object from the given parameters."""
    return DeformableRegistration(
        X=named_point_list_to_numpy_array(X),
        Y=named_point_list_to_numpy_array(Y),
        alpha=alpha,
        beta=beta,
    )


def compute_matching_accuracy(
    X: List[NamedPoint],
    Y: List[NamedPoint],
    alpha: float,
    beta: float,
    dist_func: Callable[[NamedPoint, NamedPoint], float] = distance,
    err_func: Callable[[float], float] = lambda err: err,
) -> float:
    """Computes the error for a CPD matching."""
    def_reg: DeformableRegistration = create_deformable_registration(X, Y, alpha, beta)
    def_reg.register()
    transformed_points: List[NamedPoint] = [
        NamedPoint(name, point[0], point[1])
        for (name, point) in list(zip([np.name for np in X], def_reg.TY.tolist()))
    ]
    errors: List[float] = [
        distance(np1, np2) for (np1, np2) in list(zip(transformed_points, Y))
    ]
    return sum(err_func(e) for e in errors)


# Perform optimization
image_name_to_points_map: Dict[str, List[NamedPoint]] = {
    extract_image_name_from_string(d["image"]): [
        convert_label_dict_to_namedpoint(lab) for lab in d["label"]
    ]
    for d in read_json_data(args.path_to_labels)
}

perfect_points_key: List[str] = list(
    filter(
        lambda s: extract_pattern_from_string(s, image_name_regex_pattern),
        image_name_to_points_map,
    )
)
try:
    perfect_points_key = perfect_points_key[0]
except:
    raise Error("Cannot find the scanned chart labels.")

perfect_points: List[NamedPoint] = image_name_to_points_map[perfect_points_key]
regular_points: Dict[str, List[NamedPoint]] = {
    k: v for (k, v) in image_name_to_points_map.items() if k != perfect_points_key
}


def objective(trial):
    alpha: float = trial.suggest_float(
        "alpha", args.alpha_lower_bound, args.alpha_upper_bound
    )
    beta: float = trial.suggest_float(
        "beta", args.beta_lower_bound, args.beta_upper_bound
    )
    accuracies: List[float] = [
        compute_matching_accuracy(v, perfect_points, alpha, beta)
        for v in regular_points.values()
    ]
    return np.mean(accuracies)


study = optuna.create_study()
study.optimize(objective, args.num_trials)

with open(args.path_to_output, "w") as f:
    f.write(json.dumps(study.best_params))
