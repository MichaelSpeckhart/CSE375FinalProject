#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <tbb/tbb.h>
#include "functionsCSR.cc"
#include "functions.cc"

#include <chrono>
class timer {
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> lastTime;
    timer() : lastTime(std::chrono::high_resolution_clock::now()) {}
    inline double elapsed() {
        std::chrono::time_point<std::chrono::high_resolution_clock> thisTime=std::chrono::high_resolution_clock::now();
        double deltaTime = std::chrono::duration<double>(thisTime-lastTime).count();
        lastTime = thisTime;
        return deltaTime;
    }
};

namespace parallel {
using namespace std;

//this matrix code is correct and uses TBB, but it is slow
// template<typename T>
// class VectorPair{
//     public:
//     vector<size_t> vec1;
//     vector<T> vec2;
// };
// template<typename T>
// bool compareVectorPairFirstElement(const VectorPair<T>& a, const VectorPair<T>& b) {
//     if (a.vec1.empty() && b.vec1.empty()) {
//         return false;
//     } else if (a.vec1.empty()) {
//         return true;
//     } else if (b.vec1.empty()) {
//         return false;
//     } else {
//         return a.vec1[0] < b.vec1[0];
//     }
// }


// /// @brief Multiplies two compressed sparse row(CSR) matrixes
// /// @exception The number of columns in m1 must equal the number of rows in m2
// /// @tparam T The type of the matrixes
// /// @param m1 The first CSR matrix to multiply
// /// @param m2 The second CSR matrix to multiply
// /// @return The dot product of m1 and m2
// template <typename T>
// CSRMatrix<T> multiply_matrixCSR(CSRMatrix<T> m1, CSRMatrix<T> m2)
// {
//     if (m1.numColumns != m2.numRows)
//     {
//         throw std::invalid_argument("The number of columns in the first matrix must match the number of rows in the second matrix.");
//     }
//     CSRMatrix<T> returnMatrix;
//     returnMatrix.numRows = m1.numRows;
//     returnMatrix.numColumns = m2.numColumns;
//     returnMatrix.row_ptr.push_back(0);
//     CSRMatrix<T> m2t = transpose_matrixCSR(m2);
//     for (size_t i = 0; i < m1.numRows; i++)
//     {
//         tbb::concurrent_vector<VectorPair<T>> columns;
//         tbb::parallel_for( tbb::blocked_range<size_t>(0, m2t.numRows), [&](tbb::blocked_range<size_t> r){
//         VectorPair<T> v;
//         for(size_t j = r.begin(); j < r.end(); j++)
//         {
//             T sum = 0;
//             size_t a1 = m1.row_ptr.at(i);
//             size_t b1 = m1.row_ptr.at(i + 1);
//             size_t a2 = m2t.row_ptr.at(j);
//             size_t b2 = m2t.row_ptr.at(j + 1);
//             while (a1 < b1 && a2 < b2)
//             {
//                 if (m1.col_ind.at(a1) < m2t.col_ind.at(a2))
//                 {
//                     a1++;
//                 }
//                 else if (m1.col_ind.at(a1) > m2t.col_ind.at(a2))
//                 {
//                     a2++;
//                 }
//                 else if (m1.col_ind.at(a1) == m2t.col_ind.at(a2))
//                 {
//                     sum += m1.val.at(a1) * m2t.val.at(a2);
//                     a1++;
//                     a2++;
//                 }
//             }
//             if (sum != 0)
//             {
//                 v.vec1.push_back(j);
//                 v.vec2.push_back(sum);
//             }
//         }
//         columns.push_back(v);
//         });
//         //sort and push on
//         std::sort(columns.begin(),columns.end(),compareVectorPairFirstElement<T>);
//         for(size_t i = 0; i <columns.size();i++){
//             returnMatrix.col_ind.insert(returnMatrix.col_ind.end(),columns[i].vec1.cbegin(),columns[i].vec1.cend());
//             returnMatrix.val.insert(returnMatrix.val.end(),columns[i].vec2.cbegin(),columns[i].vec2.cend());
//         }
//         //cerr << (returnMatrix.val.size()) << endl;
//         returnMatrix.row_ptr.push_back(returnMatrix.val.size());
//     }
//     return returnMatrix;
// }


template<typename T>
class VectorPair{
    public:
    vector<size_t> vec1;
    vector<T> vec2;

    VectorPair() : vec1(), vec2() {} // default constructor
};

/// @brief Adds two compressed spares row(CSR) matrixes together
/// @exception The two matrixes must have the same dimensions
/// @tparam T The type of both matrixes
/// @param m1 The first matrix too add
/// @param m2 The second matrix too add
/// @return m1+m2
template <typename T>
CSRMatrix<T> add_matrixCSR(CSRMatrix<T> m1, CSRMatrix<T> m2,size_t threads)
{
    if (m1.numRows != m2.numRows)
    {
        throw std::invalid_argument("The number of rows in the first matrix must match the number of rows in the second matrix.");
    }
    if (m1.numColumns != m2.numColumns)
    {
        throw std::invalid_argument("The number of columns in the first matrix must match the number of columns in the second matrix.");
    }
    CSRMatrix<T> returnMatrix;
    returnMatrix.numRows = m1.numRows;
    returnMatrix.numColumns = m1.numColumns;
    returnMatrix.row_ptr.push_back(0);
    const auto processor_count = threads;
    std::vector<vector<VectorPair<T>>> result;
    result.resize(processor_count);
    //the vector of threads
  	std::vector<std::thread> threadPool;
	threadPool.reserve(processor_count);
    //lambda to run
    auto do_work = [&](int k){
        vector<VectorPair<T>> v;
		for (size_t i = k; i < m1.numRows; i += processor_count){
        VectorPair<T> row;
        size_t a1 = m1.row_ptr.at(i);
        size_t b1 = m1.row_ptr.at(i + 1);
        size_t a2 = m2.row_ptr.at(i);
        size_t b2 = m2.row_ptr.at(i + 1);
        while (a1 < b1 && a2 < b2)
        {
            if (m1.col_ind.at(a1) < m2.col_ind.at(a2))
            {
                row.vec1.push_back(m1.col_ind.at(a1));
                row.vec2.push_back(m1.val.at(a1));
                a1++;
            }
            else if (m1.col_ind.at(a1) > m2.col_ind.at(a2))
            {
                row.vec1.push_back(m2.col_ind.at(a2));
                row.vec2.push_back(m2.val.at(a2));
                a2++;
            }
            else if (m1.col_ind.at(a1) == m2.col_ind.at(a2))
            {
                T value = m1.val.at(a1) + m2.val.at(a2);
                if (value != 0)
                {
                    row.vec1.push_back(m1.col_ind.at(a1));
                    row.vec2.push_back(value);
                }
                a1++;
                a2++;
            }
        }
        while (a1 < b1)
        {
            row.vec1.push_back(m1.col_ind.at(a1));
            row.vec2.push_back(m1.val.at(a1));
            a1++;
        }
        while (a2 < b2)
        {
            row.vec1.push_back(m2.col_ind.at(a2));
            row.vec2.push_back(m2.val.at(a2));
            a2++;
        }
        v.push_back(row);
        }
        result[k] = v;
	};

	//have the threads do the work
	for (size_t i = 0; i < processor_count; ++i){
      	threadPool.push_back(thread(do_work,i));
    }
	//join the threads
	for(auto & val : threadPool){
    	val.join();
	}
    //merge results
    for (size_t i = 0; i < m1.numRows; i++){
        returnMatrix.col_ind.insert(returnMatrix.col_ind.end(),result[i%processor_count][i/processor_count].vec1.cbegin(),result[i%processor_count][i/processor_count].vec1.cend());
        returnMatrix.val.insert(returnMatrix.val.end(),result[i%processor_count][i/processor_count].vec2.cbegin(),result[i%processor_count][i/processor_count].vec2.cend());
        returnMatrix.row_ptr.push_back(returnMatrix.val.size());
    }
    return returnMatrix;
}

/// @brief Multiplies two compressed sparse row(CSR) matrixes
/// @exception The number of columns in m1 must equal the number of rows in m2
/// @tparam T The type of the matrixes
/// @param m1 The first CSR matrix to multiply
/// @param m2 The second CSR matrix to multiply
/// @return The dot product of m1 and m2
template <typename T>
CSRMatrix<T> multiply_matrixCSR(CSRMatrix<T> m1, CSRMatrix<T> m2,size_t threads)
{
    if (m1.numColumns != m2.numRows)
    {
        throw std::invalid_argument("The number of columns in the first matrix must match the number of rows in the second matrix.");
    }
    CSRMatrix<T> returnMatrix;
    returnMatrix.numRows = m1.numRows;
    returnMatrix.numColumns = m2.numColumns;
    returnMatrix.row_ptr.push_back(0);
    CSRMatrix<T> m2t = transpose_matrixCSR(m2);
    const auto processor_count = threads;
    std::vector<vector<VectorPair<T>>> result;
    result.resize(processor_count);
    //the vector of threads
  	std::vector<std::thread> threadPool;
	threadPool.reserve(processor_count);
    //lambda to run
    auto do_work = [&](int k){
        vector<VectorPair<T>> v;
		for (size_t i = k; i < m1.numRows; i += processor_count){
        VectorPair<T> row;
        for (size_t j = 0; j < m2t.numRows; j++)
        {
            T sum = 0;
            size_t a1 = m1.row_ptr.at(i);
            size_t b1 = m1.row_ptr.at(i + 1);
            size_t a2 = m2t.row_ptr.at(j);
            size_t b2 = m2t.row_ptr.at(j + 1);
            while (a1 < b1 && a2 < b2)
            {
                if (m1.col_ind.at(a1) < m2t.col_ind.at(a2))
                {
                    a1++;
                }
                else if (m1.col_ind.at(a1) > m2t.col_ind.at(a2))
                {
                    a2++;
                }
                else if (m1.col_ind.at(a1) == m2t.col_ind.at(a2))
                {
                    sum += m1.val.at(a1) * m2t.val.at(a2);
                    a1++;
                    a2++;
                }
            }
            if (sum != 0)
            {
                row.vec1.push_back(j);
                row.vec2.push_back(sum);
            }
        }
        v.push_back(row);
        }
        result[k] = v;
	};

	//have the threads do the work
	for (size_t i = 0; i < processor_count; ++i){
      	threadPool.push_back(thread(do_work,i));
    }
	//join the threads
	for(auto & val : threadPool){
    	val.join();
	}
    //merge results
    for (size_t i = 0; i < m1.numRows; i++){
        returnMatrix.col_ind.insert(returnMatrix.col_ind.end(),result[i%processor_count][i/processor_count].vec1.cbegin(),result[i%processor_count][i/processor_count].vec1.cend());
        returnMatrix.val.insert(returnMatrix.val.end(),result[i%processor_count][i/processor_count].vec2.cbegin(),result[i%processor_count][i/processor_count].vec2.cend());
        returnMatrix.row_ptr.push_back(returnMatrix.val.size());
    }
    return returnMatrix;
}

/// @brief Find the max value in a compressed sparse row(CSR) matrix
/// @tparam T The type of the matrix
/// @param m The CSR matrix to find the max value of
/// @return The max value in the matrix
template <typename T>
T find_max_CSR(CSRMatrix<T> m1)
{
    return tbb::parallel_reduce(
        tbb::blocked_range<int>(0, m1.val.size()),
        m1.val[0],
        [&](const tbb::blocked_range<int>& r, T max_value) {
            cerr << "hi" << m1.val.size() << endl;
            for (int i = r.begin(); i != r.end(); ++i)
            {
                if (m1.val[i] > max_value)
                {
                    max_value = m1.val[i];
                }
            }
            return max_value;
        },
        [](T x, T y) { return std::max(x, y); }
    );}


// /**
//  * @brief As a prerequisite for the Jacobi Method, the matrix must be diagonally dominant,
//  * meaning that the elements on the diagonal indices of the matrix are greater or equal to
//  * the sum of the rest of the elements in that row.
//  * 
//  * @param denseMatrix 
//  * @return true If is diagonally dominant
//  * @return false Otherwise
//  */
// template <typename T>
// bool diagonally_dominant(CSRMatrix<T> m1) {
//     for (size_t i = 0; i < m1.numRows; ++i) {
//         size_t a1 = m1.row_ptr.at(i);
//         size_t b1 = m1.row_ptr.at(i + 1);
//         T sum = 0.0;
//         T diagonal = 0.0;
//         while(a1 < b1){
//             if(m1.col_ind[a1] == i){
//                 diagonal = std::abs(m1.val[a1]);
//             }else{
//                 sum += std::abs(m1.val[a1]);
//             }
//             a1++;
//         }
//         if (diagonal < sum) {
//             return false;
//         }
//     }
    
//     return true;
// }

/**
 * @brief The Jacobi Method is an iterative method for determining the solutions of a strictly
 * diagonally dominant matrix A. Through each iteration, the values of x[i] are approximated through
 * the formula x[i] = B[i]
 * 
 * @param denseMatrix 
 * @param B 
 * @param iterations 
 */
template <typename T>
std::vector<T> jacobi_method(CSRMatrix<T> m1, std::vector<T> B, int maxIterations) {
    if (diagonally_dominant(m1) == false) {
        throw std::invalid_argument("Input matrix is not diagonally dominant");
    }
    std::vector<T> xValues(B.size(), 0.0);
    std::vector<T> approxValues(B.size(), 0.0);
    int iterations = 0;
    while (iterations < maxIterations) {
        tbb::parallel_for( tbb::blocked_range<size_t>(0, m1.numRows), [&](tbb::blocked_range<size_t> r){
        for(size_t i = r.begin(); i < r.end(); i++){
            size_t a1 = m1.row_ptr.at(i);
            size_t b1 = m1.row_ptr.at(i + 1);
            T sum = 0.0;
            T diagonal = 0.0;
            while(a1 < b1){
                if(m1.col_ind[a1] == i){
                    diagonal = m1.val[a1];
                }else{
                    sum += m1.val[a1] * xValues[m1.col_ind[a1]];
                }
                a1++;
            }
            //no devide by zero error becuase of diagonally dominant check
            approxValues[i] = (B[i] - sum) / diagonal;
            xValues = approxValues;
            iterations++;
        }});
    
}
return approxValues;
}

}

#include <iomanip>
void printMatrix(vector<vector<double>> &A) {
    // Set the width of each output element to 8 characters
    const int width = 2;
    // Loop over each row of the matrix
    for (size_t i = 0; i < A.size(); i++) {
        // Loop over each element in the row
        for (size_t j = 0; j < A[i].size(); j++) {
            // Set the output width for the element
            cout << setw(width) << A[i][j];
        }
        // End the row with a newline
        cout << endl;
    }
}

#include <iostream>
#include <vector>
#include <cassert>

int main() {
    size_t N = 100000;
    vector<vector<double>> dense(N, vector<double>(N, 0.0));
    vector<double> B(N, 1.0);
    for (int i = 0; i < N; ++i) {
        dense[i][i] = 2.0;
        if (i >0) {
            dense[i][i-1] = -1.0;
        } 
        if (i < N - 1) {
            dense[i][i+1] = -1.0;
        }
    }
    CSRMatrix<double> sparse = from_vector(dense);
    // vector<double> resultCSR = parallel::jacobi_method(sparse, B, 20);
    // vector<double> resultDense = parallel::jacobi_method(dense, B, 20);
    // for(size_t i = 0 ; i < resultDense.size();i++){
    //     if(resultDense[i] != resultCSR[i]){
    //         cerr<< "yikes!" << endl;
    //     }
    // }
    // cerr << "Yay!" << endl;
    timer stopwatch;
    std::vector<vector<double> > parallel;
    std::vector<vector<double> > serial;
    parallel.resize(16);
    serial.resize(16);
    for(size_t i = 0 ; i <16;i++){
        for(size_t j = 0; j <5;j++){
            tbb::task_arena arena(i+1);
	        	arena.execute([&]() {
                stopwatch.elapsed();
                parallel::jacobi_method(sparse, B, 20);
                parallel[i].push_back(stopwatch.elapsed());
            });
            if(i == 0){
                stopwatch.elapsed();
                jacobi_method(sparse, B, 20);
                serial[i].push_back(stopwatch.elapsed());
            }
            
        }
    }
    for(size_t i = 0; i < parallel.size();i++){
        double time = 0;
        for(size_t j = 0 ; j < parallel[0].size();j++){
            time += parallel[i][j];
        }
        cerr<< time/parallel[0].size() << ",";
    }
    cerr<< endl;
    for(size_t i = 0; i < serial.size();i++){
        double time = 0;
        for(size_t j = 0 ; j < serial[0].size();j++){
            time += serial[i][j];
        }
        cerr<< time/serial[0].size() << ",";
    }
    cerr<< endl;
  return 0;
}
// int main() {
//     timer stopwatch;
//     std::vector<double> denseResults;
//     std::vector<double> sparseResults;
//     std::vector<int> size = {10000,15000,20000,25000,30000,35000,40000};
//     //std::vector<int> size = {500,600,700,800,900,1000};
//     for(size_t i = 0 ; i <size.size();i++){
//         size_t N = size[i];
//     vector<vector<double>> dense(N, vector<double>(N, 0.0));
//     vector<double> B(N, 1.0);
//     for (int i = 0; i < N; ++i) {
//         dense[i][i] = 2.0;
//         if (i >0) {
//             dense[i][i-1] = -1.0;
//         } 
//         if (i < N - 1) {
//             dense[i][i+1] = -1.0;
//         }
//     }
//     CSRMatrix<double> sparseMatrix = from_vector(dense);
//         stopwatch.elapsed();
//         parallel::jacobi_method(dense, B, 20);
//         denseResults.push_back(stopwatch.elapsed());
//         stopwatch.elapsed();
//         parallel::jacobi_method(sparseMatrix, B, 20);
//         sparseResults.push_back(stopwatch.elapsed());
//     }
//     for(auto val : denseResults){
//         cerr<< val << ",";
//     }
//     cerr<< endl;
//     for(auto val : sparseResults){
//         cerr<< val << ",";
//     }
//     cerr<< endl;
//     return 0;
// }


// int main() {
//     CSRMatrix<double> m1 = load_fileCSR<double>("../../data/matrices/1138_bus.mtx");
//     //CSRMatrix<double> m2 = transpose_matrixCSR<double>(m1);

//     CSRMatrix<double> m3 = multiply_matrixCSR<double>(m1, m1);
//     CSRMatrix<double> m4 = parallel::multiply_matrixCSR<double>(m1, m1);

//     return 0;
// }

// int main() {
//     timer stopwatch;
//     std::vector<vector<double> > parallel;
//     std::vector<vector<double> > serial;
//     parallel.resize(16);
//     serial.resize(16);
//     for(size_t i = 0 ; i <16;i++){
//         for(size_t j = 0; j <5;j++){
//             std::vector<std::vector<double> > A = generate_random_matrix(3000,3000,1,10000);
//             std::vector<std::vector<double> > B = generate_random_matrix(3000,3000,1,10000);
//             tbb::task_arena arena(i+1);
// 	        	arena.execute([&]() {
//                 stopwatch.elapsed();
//                 parallel::gaussian_elimination_parallel(A);
//                 parallel[i].push_back(stopwatch.elapsed());
//             });
//             stopwatch.elapsed();
//             parallel::gaussian_elimination(B);
//             serial[i].push_back(stopwatch.elapsed());
//         }
//     }
//     for(size_t i = 0; i < parallel.size();i++){
//         double time = 0;
//         for(size_t j = 0 ; j < parallel[0].size();j++){
//             time += parallel[i][j];
//         }
//         cerr<< time/parallel[0].size() << ",";
//     }
//     cerr<< endl;
//     for(size_t i = 0; i < serial.size();i++){
//         double time = 0;
//         for(size_t j = 0 ; j < serial[0].size();j++){
//             time += serial[i][j];
//         }
//         cerr<< time/serial[0].size() << ",";
//     }
//     cerr<< endl;
    


//     return 0;
// }

// int main() {
//     timer stopwatch;
//     std::vector<double> dense;
//     std::vector<double> sparse;
//     // std::vector<int> size = {1000,1500,2000,2500,3000,3500,4000,4500};
//     std::vector<int> size = {500,600,700,800,900,1000};
//     for(size_t i = 0 ; i <size.size();i++){
//         std::vector<std::vector<double> > A = generate_random_matrix_sparse(size[i],size[i],1,10000,3);
//         CSRMatrix<double> B = from_vector(A);
//         stopwatch.elapsed();
//         mult_matrix(A,A);
//         dense.push_back(stopwatch.elapsed());
//         stopwatch.elapsed();
//         multiply_matrixCSR(B, B);
//         sparse.push_back(stopwatch.elapsed());
//     }
//     for(auto val : dense){
//         cerr<< val << ",";
//     }
//     cerr<< endl;
//     for(auto val : sparse){
//         cerr<< val << ",";
//     }
//     cerr<< endl;
//     return 0;
// }

// int main() {
//     timer stopwatch;
//     std::vector<vector<double> > dense;
//     std::vector<vector<double> > sparse;
//     std::vector<int> size = {1,2,3,4,5,10,20,50,100,200,500,1000};
//     dense.resize(size.size());
//     sparse.resize(size.size());
//     for(size_t i = 0 ; i <size.size();i++){
//         for(size_t j = 0; j <5;j++){
//             std::vector<std::vector<double> > A = generate_random_matrix_sparse(1500,1500,1,10000,size[i]);
//             CSRMatrix<double> B = from_vector(A);
//             stopwatch.elapsed();
//             mult_matrix(A,A);
//             dense[i].push_back(stopwatch.elapsed());
//             stopwatch.elapsed();
//             multiply_matrixCSR(B, B);
//             sparse[i].push_back(stopwatch.elapsed());
//         }
//     }
//     for(size_t i = 0; i < dense.size();i++){
//         double time = 0;
//         for(size_t j = 0 ; j < dense[0].size();j++){
//             time += dense[i][j];
//         }
//         cerr<< time/dense[0].size() << ",";
//     }
//     cerr<< endl;
//     for(size_t i = 0; i < sparse.size();i++){
//         double time = 0;
//         for(size_t j = 0 ; j < sparse[0].size();j++){
//             time += sparse[i][j];
//         }
//         cerr<< time/sparse[0].size() << ",";
//     }
//     cerr<< endl;
//     return 0;
// }

// int main() {
//     timer stopwatch;
//     std::vector<vector<double> > parallel;
//     std::vector<vector<double> > serial;
//     parallel.resize(16);
//     serial.resize(16);
//     CSRMatrix<double> m1 = load_fileCSR<double>("stomach.mtx");
//     CSRMatrix<double> m3 = transpose_matrixCSR(m1);
//     // CSRMatrix<double> one = add_matrixCSR(m3,m1);
//     // CSRMatrix<double> two = parallel::add_matrixCSR(m3,m1,16);
//     // if(one.numRows != two.numRows){
//     //     cerr << "yikes" << endl;
//     // }
//     // if(one.numColumns != two.numColumns){
//     //     cerr << "yikes"<< endl;
//     // }
//     // if(one.row_ptr.size() != two.row_ptr.size()){
//     //     cerr << "yikes"<< endl;
//     // }
//     // if(one.col_ind.size() != two.col_ind.size()){
//     //     cerr << "yikes"<< endl;
//     // }
//     // if(one.val.size() != two.val.size()){
//     //     cerr << "yikes"<< endl;
//     // }
//     // for(size_t i = 0 ; i <one.row_ptr.size();i++){
//     //     if(one.row_ptr[i] != two.row_ptr[i]){
//     //     cerr << "yikes"<< endl;
//     // }
//     // }
//     // for(size_t i = 0 ; i <one.col_ind.size();i++){
//     //      if(one.col_ind[i] != two.col_ind[i]){
//     //     cerr << "yikes"<< endl;
//     // }
//     // }
//     // for(size_t i = 0 ; i <one.val.size();i++){
//     //     if(one.val[i] != two.val[i]){
//     //     cerr << "yikes"<< endl;
//     // }
//     // }
//     // cerr<< "yay?";
//     for(size_t i = 0 ; i <16;i++){
//         for(size_t j = 0; j <10;j++){
//             stopwatch.elapsed();
//             parallel::add_matrixCSR(m1,m3,i+1);
//             parallel[i].push_back(stopwatch.elapsed());
//             stopwatch.elapsed();
//             add_matrixCSR(m1,m3);
//             serial[i].push_back(stopwatch.elapsed());
//         }
//     }
//     for(size_t i = 0; i < parallel.size();i++){
//         double time = 0;
//         for(size_t j = 0 ; j < parallel[0].size();j++){
//             time += parallel[i][j];
//         }
//         cerr<< time/parallel[0].size() << ",";
//     }
//     cerr<< endl;
//     for(size_t i = 0; i < serial.size();i++){
//         double time = 0;
//         for(size_t j = 0 ; j < serial[0].size();j++){
//             time += serial[i][j];
//         }
//         cerr<< time/serial[0].size() << ",";
//     }
//     cerr<< endl;
//     return 0;
// }
