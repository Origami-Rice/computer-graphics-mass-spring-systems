#include "fast_mass_springs_step_dense.h"
#include <igl/matlab_format.h>

void fast_mass_springs_step_dense(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXi & b,
  const double delta_t,
  const Eigen::MatrixXd & fext,
  const Eigen::VectorXd & r,
  const Eigen::MatrixXd & M,
  const Eigen::MatrixXd & A,
  const Eigen::MatrixXd & C,
  const Eigen::LLT<Eigen::MatrixXd> & prefactorization,
  const Eigen::MatrixXd & Uprev,
  const Eigen::MatrixXd & Ucur,
  Eigen::MatrixXd & Unext)
{
  double w = 1e10;
  Eigen::MatrixXd y = ((M * (2 * Ucur - Uprev)) / (delta_t * delta_t)) + fext;
  Eigen::MatrixXd f;
  f.resize(E.rows(), 3);
  Unext = Ucur;
  for(int iter = 0; iter < 50; iter++) {
    for (int i = 0; i < E.rows(); ++i) {
        f.row(i) = r[i] * (Unext.row(E(i, 0)) - Unext.row(E(i, 1))).normalized();
    }
    Eigen::MatrixXd l = k * A.transpose() * f + y + w * C.transpose() * C * V;
    Unext = prefactorization.solve(l);
  }
}
