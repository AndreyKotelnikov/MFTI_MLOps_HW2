#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "entropy.h"

namespace py = pybind11;

PYBIND11_MODULE(entropy_core, m) {
    m.doc() = "Python bindings for entropy calculation";

    m.def("vector_entropy", &vector_entropy, R"pbdoc(
        Compute the entropy of a probability distribution.

        Parameters:
            probs : list of float
                The probability distribution.

        Returns:
            float
                The entropy of the distribution.
    )pbdoc");
}
