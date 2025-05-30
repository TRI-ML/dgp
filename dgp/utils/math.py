import numpy as np

from dgp.utils.pose import Pose


class Covariance3D:
    """3D covariance object.

    Parameters
    ----------
    data: np.ndarray[np.float32]
        Array of shape (6, ) or (3, 3). If 6-vector is used,
        the order will be interpreted as [Var(X), Cov(X, Y), Cov(X, Z), Var(Y), COV(Y, Z), Var(Z)].

    validate: bool, default: False
        Validate if the input data satisfy the criteria of a covariance matrix.
        It raises a ValueError if the validation fails.

    """
    def __init__(self, data: np.ndarray, validate: bool = False) -> None:
        assert data.shape == (6, ) or data.shape == (3, 3)
        if data.shape == (3, 3):
            data = self._get_array(data)
        self._data = data
        if validate:
            self._assert_symmetry(self.mat3)
            self._assert_positive_definite(self.mat3)

    @staticmethod
    def _assert_symmetry(mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> None:
        if not np.allclose(mat, mat.T, rtol=rtol, atol=atol):
            raise ValueError(f"{mat} is not symmetric!")

    @staticmethod
    def _assert_positive_definite(mat: np.ndarray) -> None:
        eigenvalues = np.linalg.eigvalsh(mat)
        if not np.all(eigenvalues > 0.0):
            raise ValueError(f"\n{mat} is not positive definite! Eigenvalues={eigenvalues}.")

    @staticmethod
    def _get_mat(data: np.ndarray) -> np.ndarray:
        assert data.shape == (6, ), f"data.shape={data.shape} != (6,)!"
        var_x = data[0]
        cov_xy = data[1]
        cov_xz = data[2]
        var_y = data[3]
        cov_yz = data[4]
        var_z = data[5]

        return np.array(
            [
                [var_x, cov_xy, cov_xz],
                [cov_xy, var_y, cov_yz],
                [cov_xz, cov_yz, var_z],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _get_array(data: np.ndarray) -> np.ndarray:
        assert data.shape == (3, 3), f"data.shape={data.shape} != (3, 3)!"
        return data[np.triu_indices(3)]

    @property
    def arr6(self) -> np.ndarray:
        return self._data

    @property
    def var_x(self) -> float:
        return self.arr6[0]

    @property
    def cov_xy(self) -> float:
        return self.arr6[1]

    @property
    def cov_xz(self) -> float:
        return self.arr6[2]

    @property
    def var_y(self) -> float:
        return self.arr6[3]

    @property
    def cov_yz(self) -> float:
        return self.arr6[4]

    @property
    def var_z(self) -> float:
        return self.arr6[5]

    @property
    def mat3(self) -> np.ndarray:
        return self._get_mat(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.arr6})"

    def tobytes(self) -> bytes:
        return self.arr6.tobytes()

    def __mul__(self, pose: Pose) -> "Covariance3D":
        R = pose.rotation_matrix
        return Covariance3D(data=R.T @ self.mat3 @ R)

    def __rmul__(self, pose: Pose) -> "Covariance3D":
        R = pose.rotation_matrix
        return Covariance3D(data=R @ self.mat3 @ R.T)
