#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <climits>
#include <string>
#include <vector>
#include <omp.h>

using namespace std;

/// error with dimensions in a matrix
vector<vector<double> > d_err{{INT_MAX, INT_MAX, INT_MIN},
                              {INT_MIN, INT_MIN, INT_MAX}};
/// arithmetic error
vector<vector<double> > a_err{{INT_MIN, INT_MIN, INT_MAX},
                              {INT_MAX, INT_MAX, INT_MIN}};

/// @brief Reads a file and transforms the bytes into a 2d vector (a.k.a
/// matrix).
/// @param filename The name of the file with the data.
/// @return 2d vector containing the data.
vector<vector<vector<double> > > read_file(char *filename) {
  try {
    fstream file(filename);
    if (file.fail()) {
      cout << "error (" << errno << "): failed to open file '" << filename
           << "'" << endl;
      return vector<vector<vector<double> > >();
    }

    string row;
    vector<string> data;
    while (getline(file, row))
      data.push_back(row);

    string s_temp;
    vector<string> f_data;

    for (auto i : data) {
      stringstream stream(i);
      while (getline(stream, s_temp, ','))
        f_data.push_back(s_temp);
    }

    int num;

    int d1, d2;

    vector<vector<double> > temp;
    vector<vector<vector<double> > > p_matrices;

    data = f_data;

    vector<double> line;

    num = (int)stof(data[0]);
    data.erase(data.begin(), data.begin() + 1);


    for (int i = 0; i < num; ++i) {
      d1 = (int)stof(data[0]);
      d2 = (int)stof(data[1]);
      data.erase(data.begin(), data.begin() + 2);

      for (int j = 0; j < d1; ++j) {
        for (int k = 0; k < d2; ++k) {
          double current = (double)stof(data[k]);
          line.push_back(current);
        }
        temp.push_back(line);
        line.clear();
        data.erase(data.begin(), data.begin() + d2);
      }
      p_matrices.push_back(temp);
      temp.clear();
    }
    return p_matrices;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return {{}};
  }
}

/// @brief Adds to matrices together.
/// @param m1 first matrix.
/// @param m2 second matrix.
/// @return The sum of the matrices.
vector<vector<double> > sum_matrix(vector<vector<double> > m1,
                                   const vector<vector<double> > m2) {
  //   NB: in the future it may be important to use the compiler defined sizes
  //   for platform portability
  size_t d1 = static_cast<size_t>(m1.size()),
            d2 = static_cast<size_t>(m1[0].size());

  // If the matrices do not have the same dimensions we return a dimension error
  // for the GUI to handle
  if (d1 != static_cast<size_t>(m2.size()) || d2 != static_cast<size_t>(m2[0].size()))
    return d_err;


  for (size_t i = 0; i < d1; ++i)
    for (size_t j = 0; j < d2; ++j)
      m1[i][j] -= m2[i][j];

  return m1;
}

vector<vector<double> > sub_matrix(vector<vector<double> > m1,
                                   const vector<vector<double> > m2) {
  //   NB: in the future it may be important to use the compiler defined sizes
  //   for platform portability
  const int d1 = static_cast<int>(m1.size()),
            d2 = static_cast<int>(m1[0].size());

  // If the matrices do not have the same dimensions we return a dimension error
  // for the GUI to handle
  if (d1 != static_cast<int>(m2.size()) || d2 != static_cast<int>(m2[0].size()))
    return d_err;


  for (int i = 0; i < d1; ++i)
    for (int j = 0; j < d2; ++j)
      m1[i][j] -= m2[i][j];

  return m1;
}

/// @brief Creates a random matrix of given dimensions filled with values in
/// specified range.
/// @param d1 Number of rows.
/// @param d2 Number of columns.
/// @param min Lower bound of values.
/// @param max Upper bound of values.
/// @return The generated matrix.
vector<vector<double> > generate_random_matrix(const int d1, const int d2,
                                               const double min,
                                               const double max) {
  // must be a 1x1 matrix at the minimum
  if (d1 < 1 || d2 < 1)
    return d_err;

  vector<vector<double> > matrix;
  vector<double> line;

  random_device rd;
  default_random_engine eng(rd());
  uniform_real_distribution<double> distr(min, max);


  for (int i = 0; i < d1; ++i) {
    for (int j = 0; j < d2; ++j)
      line.push_back(distr(eng));
    matrix.push_back(line);
    line.clear();
  }

  return matrix;
}

/// @brief Creates a random matrix of given dimensions filled with values in
/// specified range.
/// @param d1 Number of rows.
/// @param d2 Number of columns.
/// @param min Lower bound of values.
/// @param max Upper bound of values.
/// @param sparse The inverse probability of a non-zero entry. For example,
///        a value of 2 means that each entry has a 50% chance of being zero.
/// @return The generated matrix.
vector<vector<double>> generate_random_matrix_sparse(const int d1, const int d2,
                                                      const double min,
                                                      const double max,
                                                      const int sparse) {
  // must be a 1x1 matrix at the minimum
  if (d1 < 1 || d2 < 1)
    return {{0.0}};

  vector<vector<double>> matrix(d1, vector<double>(d2, 0.0));  // initialize with zeros

  random_device rd;
  default_random_engine eng(rd());
  uniform_real_distribution<double> distr(min, max);

  for (int i = 0; i < d1; ++i) {
    for (int j = 0; j < d2; ++j) {
      if ((rand() % sparse) < 1) {  // non-zero entry
        matrix[i][j] = distr(eng);
      }
    }
  }

  return matrix;
}


/// @brief Multiplies two matrices together.
/// @param m1 First matrix.
/// @param m2 Second matrix.
/// @return Product matrix.
vector<vector<double> > mult_matrix(const vector<vector<double> > m1,
                                    const vector<vector<double> > m2) {
  const int r1 = static_cast<int>(m1.size()),
            c1 = static_cast<int>(m1[0].size()),
            r2 = static_cast<int>(m2.size()),
            c2 = static_cast<int>(m2[0].size());
  //   columns of first matrix must equal rows of second
  if (c1 != r2)
    return d_err;

  vector<vector<double> > m3(r1);

  // Try using the std::par_unseq execution policy from C++20 (it's actually
  // pretty cool how it implements SIMD)

  for (auto i = 0; i < r1; i++)
    m3[i] = vector<double>(c2, 0);


  for (int i = 0; i < r1; ++i)
    for (int j = 0; j < c2; ++j)
      for (int k = 0; k < r2; ++k)
        m3[i][j] += m1[i][k] * m2[k][j];

  return m3;
}

/// @brief Scales the matrix upwards by a given constant (i.e. multiply every
/// value in the matrix).
/// @param m1 The matrix.
/// @param s Scaling constant.
/// @return The updated matrix.
vector<vector<double> > scale_up(vector<vector<double> > m1, const double s) {
  const int d1 = static_cast<int>(m1.size()),
            d2 = static_cast<int>(m1[0].size());

  for (auto i = 0; i < d1; ++i)
    for (auto j = 0; j < d2; ++j)
      m1[i][j] *= s;
  return m1;
}

/// @brief Scales the matrix downwards by a given constant (i.e. divides every
/// value in the matrix).
/// @param m1 The matrix.
/// @param s Scaling constant.
/// @return The update matrix.
vector<vector<double> > scale_down(vector<vector<double> > m1, const double s) {
  // cannot divide by zero
  // AS 2023: changed it to s == 0 (instead of !s) to prevent potential float-
  // 	      int point precision errs
  if (s == 0.0)
    return a_err;

  const int d1 = static_cast<int>(m1.size()),
            d2 = static_cast<int>(m1[0].size());

  for (auto i = 0; i < d1; ++i)
    for (auto j = 0; j < d2; ++j)
      m1[i][j] /= s;
  return m1;
}

/// @brief Transposes the matrix.
/// @param m1 Input matrix.
/// @return Transposed matrix.
vector<vector<double> > transpose(const vector<vector<double> > m1) {
  const int d1 = static_cast<int>(m1.size()),
            d2 = static_cast<int>(m1[0].size());

  vector<vector<double> > m2(d2);


  for (auto i = 0; i < d2; ++i)
    m2[i] = vector<double>(d1);


  for (auto i = 0; i < d1; ++i)
    for (auto j = 0; j < d2; ++j)
      m2[j][i] = m1[i][j];

  return m2;
}

/// @brief Saves the current working set of matrices.
/// @param matrices All of the matrices in the current process.
/// @param filename File to save matrices.
/// @return True on success, false otherwise.
bool save_file(const vector<vector<vector<double> > > matrices,
               const char *filename) {
  fstream f;
  f.open(filename, fstream::out | fstream::trunc);
  if (f.fail()) {
    cout << "error (" << errno << "): error opening file" << endl;
    return false;
  }
  auto size = matrices.size();
  f << size;
  f << '\n';

  size_t d1, d2;

  for (auto matrix : matrices) {
    d1 = matrix.size();
    d2 = matrix[0].size();
    f << d1;
    f << ',';
    f << d2;
    f << '\n';
    for (size_t j = 0; j < d1; ++j)
      for (size_t k = 0; k < d2; ++k) {
        f << matrix[j][k];
        if (j != d1 - 1 || k != d2 - 1)
          f << ',';
        else if (j == d1 - 1 && k == d2 - 1 && size != 1)
          f << '\n';
      }
    size--;
  }
  return true;
}

/// @brief Swaps the rows of a matrix.
/// @param m The input matrix.
/// @param i First row.
/// @param j Second row.
void swap_row(vector<vector<double> > &m, const int i, const int j) {
  // NB: the matrix is passed by reference to avoid copy on call since matrix
  // could be large which could hurt performance.
  const int n = static_cast<int>(m.size());
  for (int k = 0; k <= n; k++) {
    const double temp = m[i][k];
    m[i][k] = m[j][k];
    m[j][k] = temp;
  }
}

/// @brief Performs forward elimination on the matrix.
/// @param m The input matrix.
/// @return -1 on failure, positive constant otherwise.
int forward_elimination(vector<vector<double> > &m) {
  const int n = static_cast<int>(m.size());
  for (int k = 0; k < n; k++) {
    int i_max = k;
    int v_max = m[i_max][k];

    for (int i = k + 1; i < n; i++)
      if (abs(m[i][k]) > v_max)
        v_max = m[i][k], i_max = i;

    if (!m[k][i_max])
      return k;

    if (i_max != k)
      swap_row(m, k, i_max);

    for (int i = k + 1; i < n; i++) {
      double f = m[i][k] / m[k][k];

      for (int j = k + 1; j <= n; j++)
        m[i][j] -= m[k][j] * f;

      m[i][k] = 0;
    }
  }
  return -1;
}

/// @brief Performs a backwards substition for a given matrix.
/// @param m Input matrix.
/// @return the result.
vector<double> backward_substitution(vector<vector<double> > &m) {
  const int n = static_cast<int>(m.size());
  vector<double> x(n);

  for (int i = n - 1; i >= 0; i--) {
    x[i] = m[i][n];

    for (int j = i + 1; j < n; j++)
      x[i] -= m[i][j] * x[j];

    x[i] = x[i] / m[i][i];
  }

  vector<double> res(begin(x), end(x));
  return res;
}

// gaussian elimination with partial pivoting
// returns true if successful, false if A is singular
// Modifies both A and b to store the results
bool gaussian_elimination(std::vector<std::vector<double> > &A, std::vector<double> &b) {
    const size_t n = A.size();

    for (size_t i = 0; i < n; i++) {

        // find pivot row and swap
        size_t max = i;
        for (size_t k = i + 1; k < n; k++)
            if (abs(A[k][i]) > abs(A[max][i]))
                max = k;
        std::swap(A[i], A[max]);
        std::swap(b[i], b[max]);

        // singular or nearly singular
        if (abs(A[i][i]) <= 1e-10)
            return false;

        // pivot within A and b
        for (size_t k = i + 1; k < n; k++) {
            double t = A[k][i] / A[i][i];
            for (size_t j = i; j < n; j++) {
                A[k][j] -= A[i][j] * t;
            }
            b[k] -= b[i] * t;
        }
    }

    // back substitution
    for (int i = n - 1; i >= 0; i--) {
        for (size_t j = i + 1; j < n; j++)
            b[i] -= A[i][j] * b[j];
        b[i] = b[i] / A[i][i];
    }
    return true;
}

/**
 * @brief As a prerequisite for the Jacobi Method, the matrix must be diagonally dominant,
 * meaning that the elements on the diagonal indices of the matrix are greater or equal to
 * the sum of the rest of the elements in that row.
 * 
 * @param denseMatrix 
 * @return true If is diagonally dominant
 * @return false Otherwise
 */
bool diagonally_dominant(std::vector<std::vector<double>> denseMatrix) {
    for (size_t i = 0; i < denseMatrix.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < denseMatrix[i].size(); ++j) {
            if (j != i) {
                sum += std::abs(denseMatrix[i][j]);
            }
        }
        if (denseMatrix[i][i] < sum) {
            return false;
        }
    }
    
    return true;
}


/**
 * @brief The Jacobi Method is an iterative method for determining the solutions of a strictly
 * diagonally dominant matrix A. Through each iteration, the values of x[i] are approximated through
 * the formula x[i] = B[i]
 * 
 * @param denseMatrix 
 * @param B 
 * @param iterations 
 */
std::vector<double> jacobi_method(std::vector<std::vector<double>> denseMatrix, std::vector<double> B, int maxIterations) {
    if (diagonally_dominant(denseMatrix) == false) {
        throw std::invalid_argument("Input matrix is not diagonally dominant");
    }
    std::vector<double> xValues(B.size(), 0.0);
    std::vector<double> approxValues(B.size(), 0.0);
    int iterations = 0;
    while (iterations < maxIterations) {
        for (size_t i = 0; i < denseMatrix.size(); ++i) {
            double sum = 0.0;
            const size_t m = static_cast<size_t>(denseMatrix[i].size());
            for (size_t j = 0; j < m; ++j) {
                if (j != i) {
                    sum += denseMatrix[i][j] * xValues[j];
                }
            }
            approxValues[i] = (B[i] - sum) / denseMatrix[i][i];
        
          xValues = approxValues;
          iterations++;
      }  
    
    }
    return approxValues;
}


/**
 * @brief Perform LU Factorization on a dense matrix A
 * 
 * @param A 
 * @return std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
 */
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> lu_factorization(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0)); 
    std::vector<std::vector<double>> U = A; 

    for (int k = 0; k < n; k++) {
      L[k][k] = 1.0; 


      for (int i = k+1; i < n; i++) {
        double factor = U[i][k] / U[k][k]; 
        L[i][k] = factor; 
        for (int j = k; j < n; j++) {
          U[i][j] -= factor * U[k][j]; 
        }
      }
    }

    return std::make_pair(L, U);
}