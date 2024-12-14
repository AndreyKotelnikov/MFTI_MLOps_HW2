#include "entropy.h"
#include <cmath>

double vector_entropy(const std::vector<double>& probs) {
    double entropy = 0.0;
    for (double p : probs) {
        if (p > 0) {
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}
