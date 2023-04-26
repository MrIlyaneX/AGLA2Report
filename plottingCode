// Ilia_Mistiurin
// i.mistiurin@innopolis.university
// DSAI-04
#include <bits/stdc++.h>
#include <cstdio>
#include <math.h>

#ifdef WIN32
#define GNUPLOT_NAME "C:\\gnuplot\\bin\\gnuplot -persist"
#else
#define GNUPLOT_NAME "gnuplot -persist"
#endif

// int main is in the bottom of the code

// declarations of the classes used (implemented)
template<typename T>
class Matrix;
template<typename T>
class AugmentedMatrix;
template<typename T>
class SquareMatrix;
template<typename T>
class IdentityMatrix;
template<typename T>
class EliminationMatrix;
template<typename T>
class PermutationMatrix;
template<typename T>
class ColumnVector;

//Matrix class is a class that presents 2d matrix and operations over it.
template<typename T>
class Matrix {
public:

    //Default ctor. creates 0*0 matrix
    Matrix();

    //Create row_number*column_number matrix filled with 0's
    Matrix(int row_number, int column_number, T default_value = T{});

    //cope ctor
    Matrix(const Matrix<T>& item);

    //move ctor
    Matrix(Matrix<T>&& other) noexcept;

    //move assignment
    Matrix<T>& operator=(Matrix<T>&& other) noexcept;

    //std::initializer_list ctor
    Matrix(std::initializer_list<std::initializer_list<T>> list);

    //d-ctor
    ~Matrix();

    //simple overloading by
    auto& operator[](int row_index);

    // return size of matrix in pair; value package principle is -> row_size : column_size
    std::pair<int, int> size() const;

    // Swap function. swaps two obj. of class Matrix
    void swap(Matrix<T>& first_matrix_, Matrix<T>& second_matrix_) noexcept;

    // Overloaded operators, I will not make comments for each one since all details mostly similar to the
    // previous assignment

    // Assignment operator
    Matrix<T>& operator=(const Matrix<T>& item);

    Matrix<T>& operator+=(const Matrix<T>& item);

    Matrix<T> operator+(const Matrix<T>& item);

    Matrix<T>& operator-=(const Matrix<T>& item);

    Matrix<T> operator-(const Matrix<T>& item);

    Matrix<T>& operator*=(const Matrix<T>& item);

    Matrix<T> operator*(const Matrix<T>& item);

    Matrix<T>& operator*=(const int& item);

    Matrix<T> operator*(const int& item);

    bool operator==(const Matrix<T>& item);

    //overloaded ostream operator
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& item) {
        for (const auto& row : item.matrix_) {
            for (int i = 0; i < row.size(); i++) {
                if (row.size() - 1 == i) {
                    os << std::fixed << std::setprecision(4) << row[i];
                } else {
                    os << std::fixed << std::setprecision(4) << row[i] << " ";
                }
            }
            os << '\n';
        }
        return os;
    }

    //overloaded istream operator
    friend std::istream& operator>>(std::istream& input_stream, Matrix<T>& item) {
        for (auto& row : item.matrix_) {
            for (auto& elem : row) {
                input_stream >> elem;
            }
        }
        return input_stream;
    }

    Matrix<T>& Transpose();

    void Resize(int new_row_size, int new_col_size);

    void SetStepCounter(int n);

    int GetStepCounter() const;

protected:

    int row_{};
    int column_{};
    int stepCounter{};
    std::vector<std::vector<T>> matrix_{};

    //swap rows
    void Exchange_rows_(int row1, int row2);

    // Helping function for row subtraction
    // Choose row that you want to decrease, row that you will subtract and coefficient for row increase
    // (coefficient helps to get 0's on the matrix cells)
    void SubtractRow(int subtractFrom, int whatToSubtract, double times = 1.0);

};

template<typename T>
Matrix<T>::Matrix() : row_(0), column_(0), matrix_(column_) {}

template<typename T>
Matrix<T>::Matrix(const int row_number, const int column_number, T default_value)
    : row_(row_number), column_(column_number),
      matrix_(row_, std::vector<T>(column_, default_value)) {}

template<typename T>
Matrix<T>::~Matrix() {
    row_ = 0;
    column_ = 0;
    matrix_ = std::vector<std::vector<T>>(0);
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& item) : row_(item.row_), column_(item.column_) {
    matrix_.resize(row_, std::vector<T>(column_, T()));
    for (int i = 0; i < item.row_; i++) {
        for (int j = 0; j < item.column_; ++j) {
            matrix_[i][j] = item.matrix_[i][j];
        }
    }
}

template<typename T>
Matrix<T>::Matrix(Matrix<T>&& other) noexcept {
    matrix_ = std::move(other.matrix_);
    row_ = other.row_;
    column_ = other.column_;
    other.row_ = 0;
    other.column_ = 0;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) noexcept {
    if (this == &other) return *this;

    matrix_ = other.matrix_;
    row_ = other.row_;
    column_ = other.column_;

    other.matrix_ = std::move(other.matrix_);
    other.row_ = 0;
    other.column_ = 0;

    return *this;
}

template<typename T>
Matrix<T>::Matrix(const std::initializer_list<std::initializer_list<T>> list) {
    matrix_.reserve(list.size());
    int mx{};
    int mn{};
    for (const auto& element : list) {
        std::vector<T> tmp;
        tmp.reserve(element.size());
        for (const auto& element2 : element) {
            tmp.push_back(element2);
        }
        mx = ((mx < tmp.size()) ? tmp.size() : mx);
        mn = ((mn > tmp.size()) ? tmp.size() : mn);
        matrix_.push_back(std::move(tmp));
    }
    if (mx != mn) {
        for (auto& element : matrix_) element.resize(mx);
    }
    row_ = matrix_.size();
    column_ = matrix_[0].size();
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& item) {
    Matrix tmp(item);
    swap(tmp, *this);
    return *this;
}

template<typename T>
void Matrix<T>::swap(Matrix<T>& first_matrix_, Matrix<T>& second_matrix_) noexcept {
    std::swap(first_matrix_.matrix_, second_matrix_.matrix_);
    std::swap(first_matrix_.row_, second_matrix_.row_);
    std::swap(first_matrix_.column_, second_matrix_.column_);
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& item) {
    if (item.size() == size()) {
        Matrix tmp_matrix_(*this);
        for (int i = 0; i < item.row_; i++) {
            for (int j = 0; j < item.column_; ++j) {
                tmp_matrix_.matrix_[i][j] += item.matrix_[i][j];
            }
        }
        swap(tmp_matrix_, *this);
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& item) {
    Matrix tmp_matrix_(*this);
    tmp_matrix_ += item;
    return tmp_matrix_;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& item) {
    if (item.size() == size()) {
        Matrix tmp_matrix_(*this);
        for (int i = 0; i < item.row_; i++) {
            for (int j = 0; j < item.column_; ++j) {
                tmp_matrix_.matrix_[i][j] -= item.matrix_[i][j];
            }
        }
        swap(tmp_matrix_, *this);
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& item) {
    Matrix tmp_matrix_(*this);
    tmp_matrix_ -= item;
    return tmp_matrix_;
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& item) {
    if (column_ == item.row_) {
        Matrix tmp_matrix_(row_, item.column_);
        for (int i = 0; i < row_; i++) {
            for (int k = 0; k < item.row_; k++) {
                for (int j = 0; j < item.column_; j++) {
                    tmp_matrix_.matrix_[i][j] += (matrix_[i][k] * item.matrix_[k][j]);
                }
            }
        }
        swap(tmp_matrix_, *this);
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& item) {
    Matrix tmp_matrix_(*this);
    tmp_matrix_ *= item;
    return tmp_matrix_;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const int& item) {
    Matrix tmp_matrix_(*this);
    tmp_matrix_ *= item;
    return tmp_matrix_;
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(const int& item) {
    Matrix tmp_matrix_(*this);
    for (int i = 0; i < row_; i++) {
        for (int j = 0; j < column_; ++j) {
            tmp_matrix_.matrix_[i][j] *= item;
        }
    }
    swap(tmp_matrix_, *this);
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::Transpose() {
    Matrix tmp_matrix_(column_, row_);
    for (int i = 0; i < column_; i++) {
        for (int j = 0; j < row_; ++j) {
            tmp_matrix_.matrix_[i][j] = matrix_[j][i];
        }
    }
    swap(tmp_matrix_, *this);
    return *this;
}

template<typename T>
void Matrix<T>::Exchange_rows_(int row1, int row2) {
    for (int i = 0; i < column_; ++i) {
        double temp = matrix_[row1][i];
        matrix_[row1][i] = matrix_[row2][i];
        matrix_[row2][i] = temp;
    }
}

template<typename T>
void Matrix<T>::Resize(int new_row_size, int new_col_size) {
    matrix_.resize(new_row_size, std::vector<T>(new_col_size, T()));
    row_ = new_row_size;
    column_ = new_col_size;
}

template<typename T>
bool Matrix<T>::operator==(const Matrix<T>& item) {
    if (this->size() == item.size()) {
        for (int i = 0; i < row_; i++) {
            for (int j = 0; j < column_; ++j) {
                if (matrix_[i][j] != item.matrix_[i][j]) return false;
            }
        }
    } else {
        return false;
    }
    return true;
}

template<typename T>
auto& Matrix<T>::operator[](int row_index) {
    if (row_index < 0 || (row_index >= row_ && row_index >= column_)) {
        throw std::out_of_range("Index is out of range.");
    }
    return matrix_[row_index];
}

template<typename T>
std::pair<int, int> Matrix<T>::size() const {
    return {row_, column_};
}

template<typename T>
void Matrix<T>::SubtractRow(int subtractFrom, int whatToSubtract, double times) {
    for (int i = 0; i < (*this).column_; i++) {
        (*this)[subtractFrom][i] -= ((*this)[whatToSubtract][i] * times);
    }
}

template<typename T>
void Matrix<T>::SetStepCounter(int n) {
    stepCounter = n;
}

template<typename T>
int Matrix<T>::GetStepCounter() const {
    return stepCounter;
}


// --------------------------------------------------
//
//               AugmentedMatrix section
//
//---------------------------------------------------

template<typename T>
class AugmentedMatrix : public Matrix<T> {
public:
    AugmentedMatrix() : Matrix<T>(0, 0) {}

    AugmentedMatrix(Matrix<T>& A, Matrix<T>& B);

    AugmentedMatrix(Matrix<T> matrix) : Matrix<T>(matrix) {}

    // This version uses augmented matrix
    void ForwardEliminationAugmented();

    void BackwardSubstitutionAugmentedMatrix();

    void DiagonalFormAugmented();

    SquareMatrix<T> ExtractInverse();
};

template<typename T>
AugmentedMatrix<T>::AugmentedMatrix(Matrix<T>& A, Matrix<T>& B) : Matrix<T>(A.size().first, A.size().second * 2, 0) {
    int n = A.size().first;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            (*this)[i][j] = A[i][j];
            (*this)[i][j + n] = B[i][j];
        }
    }
}

template<typename T>
void AugmentedMatrix<T>::ForwardEliminationAugmented() {
    int n = (*this).row_;
    double error = 1e-7;
    for (int i = 0; i < n; ++i) {
        // In this part I find the maximal value (pivot) and swap current row with 'pivoting' row
        // I do this for future increased double precision of calculations
        double pivot = (*this)[i][i];
        int index = i;
        for (int j = i; j < n; ++j) {
            if (std::abs((*this)[j][i]) > pivot) {
                pivot = std::abs((*this)[j][i]);
                index = j;
            }
        }
        if (index != i) {
            PermutationMatrix<T> permutation_matrix(i, index, n);
            (*this) = permutation_matrix * (*this);
        }
        // Simple row subtraction
        for (int j = i + 1; j < n; ++j) {
            if (std::abs((*this)[j][i]) < error) {
                (*this)[j][i] = 0;
                continue;
            }
            EliminationMatrix<T> elimination_matrix((*this), j, i);
            (*this) = elimination_matrix * (*this);
            (*this)[j][i] = 0;
        }
    }
}

template<typename T>
void AugmentedMatrix<T>::BackwardSubstitutionAugmentedMatrix() {
    int n = (*this).row_; // since we use augmented matrix
    double error = 1e-7;
    for (int i = n - 1; i > 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            if (std::abs((*this)[j][i]) < error) {
                (*this)[j][i] = 0;
                continue;
            }
            EliminationMatrix<T> elimination_matrix((*this), j, i);
            (*this) = elimination_matrix * (*this);
            (*this)[j][i] = 0;
        }
    }
}

template<typename T>
void AugmentedMatrix<T>::DiagonalFormAugmented() {
    // dividing all diagonal elements by themselves to get identity matrix (diagonal form)
    int n = (*this).row_;
    double error = 1e-7;
    for (int i = 0; i < n; i++) {
        for (int j = n; j < 2 * n; ++j) {
            if (std::abs((*this)[i][j]) < error) {
                (*this)[i][j] = 0;
                continue;
            }
            (*this)[i][j] /= (*this)[i][i];
        }
        (*this)[i][i] /= (*this)[i][i];
    }
}
template<typename T>
SquareMatrix<T> AugmentedMatrix<T>::ExtractInverse() {
    int n = (*this).row_;
    int m = (*this).column_ / 2;
    SquareMatrix<T> square_matrix(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = m; j < m * 2; ++j) {
            square_matrix[i][j - m] = (*this)[i][j];
        }
    }
    return square_matrix;
}


// --------------------------------------------------
//
//               SquareMatrix section
//
//---------------------------------------------------

template<typename T>
class SquareMatrix : public Matrix<T> {
public:

    SquareMatrix() : Matrix<T>() {}

    //Create square number_square_matrix_*number_square_matrix_ matrix filled with 0's
    explicit SquareMatrix(const int& number_square_matrix_, T default_value = T{}) : Matrix<T>(number_square_matrix_,
                                                                                               number_square_matrix_,
                                                                                               default_value) {}

    //cope ctor
    SquareMatrix(const SquareMatrix<T>& item) : Matrix<T>(item) {}

    //std::initializer_list ctor
    SquareMatrix(std::initializer_list<std::initializer_list<T>> list) : Matrix<T>(list) {}

    SquareMatrix(Matrix<T> matrix);

    // Starter for all methods for Ax=b finding of vector x
    void FindAx(Matrix<T>& columnVector);

    // Forward Elimination operations -> creates form current matrix the Upper Triangular matrix
    // With the 'columnVector' variable row actions as with the original are going
    void ForwardElimination(Matrix<T>& columnVector);

    void ForwardElimination();

    // Backward Substitution operations to get rid of last elements upper the main diagonal
    // With the 'columnVector' variable row actions as with the original are going
    void BackwardSubstitution(Matrix<T>& columnVector);

    // Gets the matrix to the diagonal form
    // With the 'columnVector' variable row actions as with the original are going
    void DiagonalForm(Matrix<T>& columnVector);

    void FindDeterminant();

    SquareMatrix<T> GetInverseOfMatrix();

};

template<typename T>
SquareMatrix<T>::SquareMatrix(Matrix<T> matrix) : Matrix<T>(matrix) {}

template<typename T>
void SquareMatrix<T>::ForwardElimination(Matrix<T>& columnVector) {
    int n = (*this).row_;
    double error = 1e-7;
    for (int i = 0; i < n; ++i) {
        // In this part I find the maximal value (pivot) and swap current row with 'pivoting' row
        // I do this for future increased double precision of calculations
        double pivot = std::abs((*this)[i][i]);
        int index = i;
        for (int j = i; j < n; ++j) {
            if (std::abs((*this)[j][i]) > pivot) {
                pivot = std::abs((*this)[j][i]);
                index = j;
            }
        }
        if (index != i) {
            PermutationMatrix<T> permutation_matrix(i, index, n);
            (*this) = permutation_matrix * (*this);

            columnVector = permutation_matrix * columnVector;
        }
        // Simple row subtraction
        for (int j = i + 1; j < n; ++j) {
            if (std::abs((*this)[j][i]) < error) {
                (*this)[j][i] = 0;
                continue;
            }
            EliminationMatrix<T> elimination_matrix((*this), j, i);
            (*this) = elimination_matrix * (*this);
            columnVector = elimination_matrix * columnVector;

            (*this)[j][i] = 0;
        }
    }
}

template<typename T>
void SquareMatrix<T>::ForwardElimination() {
    int n = (*this).row_;
    double error = 1e-7;
    for (int i = 0; i < n; ++i) {
        // In this part I find the maximal value (pivot) and swap current row with 'pivoting' row
        // I do this for future increased double precision of calculations
        double pivot = std::abs((*this)[i][i]);
        int index = i;
        for (int j = i; j < n; ++j) {
            if (std::abs((*this)[j][i]) > pivot) {
                pivot = std::abs((*this)[j][i]);
                index = j;
            }
        }
        if (index != i) {
            PermutationMatrix<T> permutation_matrix(i, index, n);
            (*this) = permutation_matrix * (*this);
        }
        // Simple row subtraction
        for (int j = i + 1; j < n; ++j) {
            if (std::abs((*this)[j][i]) < error) {
                (*this)[j][i] = 0;
                continue;
            }
            EliminationMatrix<T> elimination_matrix((*this), j, i);
            (*this) = elimination_matrix * (*this);
            (*this)[j][i] = 0;
        }
    }
}

template<typename T>
void SquareMatrix<T>::BackwardSubstitution(Matrix<T>& columnVector) {
    int n = (*this).row_;
    double error = 1e-7;
    for (int i = n - 1; i > 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            if (std::abs((*this)[j][i]) < error) {
                (*this)[j][i] = 0;
                continue;
            }
            EliminationMatrix<T> elimination_matrix((*this), j, i);
            (*this) = elimination_matrix * (*this);
            columnVector = elimination_matrix * columnVector;

            (*this)[j][i] = 0;
        }
    }
}

template<typename T>
void SquareMatrix<T>::DiagonalForm(Matrix<T>& columnVector) {
    // dividing all diagonal elements by themselves to get identity matrix (diagonal form)
    double error = 1e-7;
    for (int i = 0; i < (*this).row_; i++) {
        columnVector[i][0] /= (*this)[i][i];
        if (error >= std::abs(columnVector[i][0])) {
            columnVector[i][0] = 0;
        }
        (*this)[i][i] /= (*this)[i][i];
    }
    std::cout << (*this) << columnVector;
}

template<typename T>
void SquareMatrix<T>::FindDeterminant() {
    (*this).stepCounter = 1;

    ForwardElimination();

    double determinant = 1.0;
    int matrixSize = (*this).size().first;
    for (int i = 0; i < matrixSize; ++i) {
        determinant *= (*this)[i][i];
    }

}

template<typename T>
SquareMatrix<T> SquareMatrix<T>::GetInverseOfMatrix() {
    IdentityMatrix<T> identity_matrix((*this).size().first);
    AugmentedMatrix<T> augmented_matrix((*this), identity_matrix);
    augmented_matrix.SetStepCounter(1);

    augmented_matrix.ForwardEliminationAugmented();

    augmented_matrix.BackwardSubstitutionAugmentedMatrix();

    augmented_matrix.DiagonalFormAugmented();

    SquareMatrix<T> answer = augmented_matrix.ExtractInverse();
    return answer;
}

template<typename T>
void SquareMatrix<T>::FindAx(Matrix<T>& columnVector) {
    (*this).SetStepCounter(1);
    std::cout << "step #0:\n" << (*this) << columnVector;

    // Create UpperTriangular matrix
    ForwardElimination(columnVector);

    // Create Diagonal Matrix
    BackwardSubstitution(columnVector);

    // Normalizing Diagonal Matrix
    DiagonalForm(columnVector);

    std::cout << "result:\n" << columnVector;
}



// --------------------------------------------------
//
//               Identity matrix section
//
//---------------------------------------------------

template<typename T>
class IdentityMatrix : public SquareMatrix<T> {
public:
    IdentityMatrix() : SquareMatrix<T>(0) {}

    IdentityMatrix(int size);
};

template<typename T>
IdentityMatrix<T>::IdentityMatrix(int size) : SquareMatrix<T>(size, 0) {
    for (int i = 0; i < size; ++i) {
        (*this)[i][i] = static_cast<T>(1);
    }
}


// --------------------------------------------------
//
//               PermutationMatrix section
//
//---------------------------------------------------

template<typename T>
class PermutationMatrix : public IdentityMatrix<T> {
public:
    PermutationMatrix(int row1, int row2, int size);
};

template<typename T>
PermutationMatrix<T>::PermutationMatrix(int row1, int row2, int size)
    : IdentityMatrix<T>(size) {
    (*this).Exchange_rows_(row1, row2);
}


// --------------------------------------------------
//
//               EliminationMatrix section
//
//---------------------------------------------------

template<typename T>
class EliminationMatrix : public IdentityMatrix<T> {
public:
    EliminationMatrix(Matrix<T>& matrix, int row, int column);
};

template<typename T>
EliminationMatrix<T>::EliminationMatrix(Matrix<T>& matrix, int row, int column)
    : IdentityMatrix<T>(matrix.size().first) {
    row;
    column;
    double times = (double) (matrix[row][column] / (double) matrix[column][column]);
    (*this).SubtractRow(row, column, times);
}

// --------------------------------------------------
//
//               ColumnVector section
//
//---------------------------------------------------


// ColumnVector class derived from Matrix, but no need was for this action
template<typename T>
class ColumnVector : public Matrix<T> {
public:
    explicit ColumnVector(int size) : Matrix<T>(size, 1, 0) {}
    ColumnVector() : Matrix<T>() {}

    ColumnVector(Matrix<T> matrix);

    ColumnVector<T> LeastSquares(Matrix<T>& A, Matrix<T>& B);

    double norm();
};

template<typename T>
double ColumnVector<T>::norm() {
    double t = 0.0;
    for (int i = 0; i < (*this).row_; i++) {
        t += ((*this)[i][0] * (*this)[i][0]);
    }
    t = sqrt(t);
    for (int i = 0; i < (*this).row_; i++) {
        (*this)[i][0] /= t;
    }
    return t;
}
template<typename T>
ColumnVector<T>::ColumnVector(Matrix<T> matrix) {
    (*this) = ColumnVector<T>(matrix.size().first);
    for (int i = 0; i < (*this).row_; ++i) {
        (*this)[i][0] = matrix[i][0];
    }
}
template<typename T>
ColumnVector<T> ColumnVector<T>::LeastSquares(Matrix<T>& A, Matrix<T>& b) {
    //std::cout << "A:\n" << A;


    Matrix<T> AT = A.Transpose();
    b = AT * b;
    A.Transpose();

    SquareMatrix<T> ATA = AT * A;
    SquareMatrix<T> Inverse = ATA.GetInverseOfMatrix();

    std::cout << "A_T*A:\n" << ATA << "(A_T*A)^-1:\n" << Inverse << "A_T*b:\n" << b;
    ColumnVector<T> x = Inverse * b;

    return x;

}


// main class, nothing special here
int main() {
#ifdef WIN32
    FILE* pipe = _popen(GNUPLOT_NAME, "w");
#else
    FILE* pipe = popen(GNUPLOT_NAME, "w");
#endif

    std::default_random_engine _random{std::random_device{}()};
    std::uniform_real_distribution<double> interval(-1000, 1000);

    int n = 50, degree = 3;

    
    // for reading points or saving them; + putting in gnuplot using temporary file
    std::ofstream toFile("data.txt");

    //std::ifstream fromFile("data.txt");

    std::vector<double> X(n), Y(n);
    for (int i = 0; i < n; ++i) {
        X[i] = interval(_random);
        Y[i] = interval(_random);
        toFile << X[i] << " " << Y[i] << "\n";
        //fromFile >> X[i] >> Y[i];
    }
    toFile.close();
    //fromFile.close();

    Matrix<double> b(n, 1);
    Matrix<double> A(n, degree + 1);

    for (int i = 0; i < n; ++i) {
        b[i][0] = Y[i];
        for (int j = 0; j <= degree; j++) {
            A[i][j] = std::pow(X[i], j);
        }
    }

    ColumnVector<double> x;
    x = x.LeastSquares(A, b);

    std::cout << "x~:\n" << x;


    fprintf(pipe, "plot [-2000:2000][-2000:2000] '%s' with points, %lf*x**3 + %lf*x**2 + %lf*x + %lf\n",
            "data.txt", x[3][0], x[2][0], x[1][0], x[0][0]);

    fflush(pipe);



#ifdef WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
}
