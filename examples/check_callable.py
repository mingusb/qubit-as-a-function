from pathlib import Path
import numpy as np
from qaf import run
from qaf_core import X, density

source = Path(__file__).with_name("callable.qaf").read_text()
branches, (probabilities, x_response) = run(source)
rho = density(branches).reshape(2, 2, 2, 2)
local = np.trace(rho, axis1=1, axis2=3)
np.testing.assert_allclose(probabilities, [0.5, 0, 0, 0.5])
np.testing.assert_allclose(local, np.eye(2) / 2)
np.testing.assert_allclose(x_response, 0.5)
joint = np.trace(rho.reshape(4, 4) @ np.kron(X, X))
np.testing.assert_allclose(joint, 1)
print("Bell populations, local state and joint coherence agree.")
