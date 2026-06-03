#include "MatLib.h"
#include <immintrin.h>
#include <random>
#include <utility>
#include <algorithm>
#include<fstream>
#include <cstring>

Mat::Mat(int row,int col): row(row),col(col),data(row*col,0.0f){}
float& Mat::operator()(int r, int c){return data[r*col+c];}
const float& Mat::operator()(int r, int c) const{return data[r*col+c];}
void Mat::set_identity(){
     if (row != col) {
        cerr << "Error: Identity matrix must be square." << endl;
        return;
    }
    for (int i = 0; i < row; i++) (*this)(i,i) = 1.0f;
}
void Mat::set_zeros(){
    std::fill(data.begin(), data.end(), 0.0f);
}
void Mat::set_ones(float value){
    std::fill(data.begin(), data.end(), value);
}
Mat Mat::identity(int size){
    Mat I(size, size);
    I.set_identity();
    return I;
}
Mat Mat::zeros(int row, int col){
    Mat Z(row, col);
    Z.set_zeros();
    return Z;
}
Mat Mat::ones(int row, int col, float value){
    Mat O(row, col);
    O.set_ones(value);
    return O;
}
void Mat::transpose(Mat& dst) const{
    if (this == &dst) {
        cerr << "Error: In-place transpose is not supported. Use the transpose() member function instead." << endl;
        return;
    }
    if (dst.row != col || dst.col != row) {
        cerr << "Error: Destination matrix dimensions must be the transpose of source." << endl;
        return;
    }
    const float* __restrict pSrc = this->data.data();
    float* __restrict pDst = dst.data.data();
    const int block = 32;
    for (int i = 0; i < row; i += block)
        for (int j = 0; j < col; j += block)
            for (int ii = i; ii < min(i + block, row); ii++)
                for (int jj = j; jj < min(j + block, col); jj++)
                    pDst[jj * dst.col + ii] = pSrc[ii * col + jj];
}
void Mat::append_row(const vector<float>& new_row){
    if (new_row.size() != col) {
        cerr << "Error: New row size must match the number of columns." << endl;
        return;
    }
    data.resize((row + 1) * col);
    std::copy(new_row.begin(), new_row.end(), data.begin() + row * col);
    row++;
}
void Mat::append_col(const vector<float>& new_col){
    if (new_col.size() != row) {
        cerr << "Error: New column size must match the number of rows." << endl;
        return;
    }
    vector<float> new_data((row) * (col + 1));
    for (int i = 0; i < row; i++) {
        std::copy(data.begin() + i * col, data.begin() + i * col + col, new_data.begin() + i * (col + 1));
        new_data[i * (col + 1) + col] = new_col[i];
    }
    std::swap(data, new_data);
    col++;
}
void Mat::append_row(float value) {
    data.resize((row + 1) * col);
    std::fill(data.begin() + row * col, data.begin() + row * col + col, value);
    row++;
}
void Mat::append_col(float value) {
    vector<float> new_data((row) * (col + 1));
    for (int i = 0; i < row; i++) {
        std::copy(data.begin() + i * col, data.begin() + i * col + col, new_data.begin() + i * (col + 1));
        new_data[i * (col + 1) + col] = value;
    }
    std::swap(data, new_data);
    col++;
}
void Mat::row_col_swap(){
    if (row != 1 && col != 1) {
        cerr << "Error: Row-Column swap is only applicable for vectors." << endl;
        return;
    }
    std::swap(row, col);
}
const int Mat::size() const { return data.size(); }
void Mat::rand(float min_val, float max_val){
    static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_real_distribution<float> dis(min_val, max_val);
    int total_size = row * col;
    for (int i = 0; i < total_size; i++) {
        data[i] = dis(gen);
    }
}
void Mat::print() const{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cout << (*this)(i, j) << " ";
        }
        cout << endl;
    }
}
void Mat::savetxt(const string& filename) const {
    ofstream outfile(filename);
    
    // Kiểm tra xem file có mở thành công không
    if (!outfile.is_open()) {
        cerr << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }

    // Ghi kích thước ma trận ở dòng đầu tiên (để sau này loadmat đọc lại được)
    outfile << row << " " << col << "\n";

    // Ghi dữ liệu ma trận, mỗi hàng xuống dòng 1 lần cho dễ đọc bằng mắt thường
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            outfile << (*this)(i, j) << " ";
        }
        outfile << "\n";
    }

    outfile.close();
    cout << "Da luu ma tran vao file text: " << filename << endl;
}
Mat& Mat::loadmat(const string& txtFilename, int start, int end) {
    ifstream infile(txtFilename);
    if (!infile.is_open()) {
        cerr << "Error: Could not open file " << txtFilename << endl;
        return *this;
    }
 
    // 1. Đếm file_col từ dòng đầu tiên, file_row bằng cách đếm tổng số dòng
    int file_row = 0, file_col = 0;
    {
        string line;
        // Đọc dòng đầu để đếm cột
        if (getline(infile, line)) {
            file_row = 1;
            istringstream iss(line);
            float v;
            while (iss >> v) file_col++;
        }
        // Đếm các dòng còn lại
        while (getline(infile, line)) {
            if (!line.empty()) file_row++;
        }
    }
 
    if (file_col == 0 || file_row == 0) {
        cerr << "Error: File is empty or malformed: " << txtFilename << endl;
        infile.close();
        return *this;
    }
 
    int target_row = file_row;
    int target_col = file_col;
    int start_idx = 1;
 
    // 2. Xử lý nếu start và end khác 0
    if (start != 0 || end != 0) {
        target_row = end - start + 1;
        start_idx = start;
    }
 
    // 3. Kiểm tra bounds
    if (start_idx < 1 || start_idx + target_row - 1 > file_row) {
        cerr << "Error: start/end out of bounds! (file has " << file_row
             << " rows, requested rows " << start_idx
             << " to " << (start_idx + target_row - 1) << ")" << endl;
        infile.close();
        return *this;
    }
 
    // 4. Khởi tạo ma trận và quay về đầu file
    *this = Mat(target_row, target_col);
    infile.clear();
    infile.seekg(0);
 
    // 5. Bỏ qua (start_idx - 1) hàng đầu
    float dummy_val;
    int elements_to_skip = (start_idx - 1) * target_col;
    for (int i = 0; i < elements_to_skip; i++) infile >> dummy_val;
 
    // 6. Đọc dữ liệu
    for (int i = 0; i < target_row; i++)
        for (int j = 0; j < target_col; j++)
            infile >> (*this)(i, j);
 
    infile.close();
    cout << "Loaded " << target_row << "x" << target_col << " from: " << txtFilename << endl;
    return *this;
}
Mat& Mat::copyfrom(const Mat& src, string direction, int start, int end) {
    if (direction == "row") {
        if (start < 0 || end > src.row || start > end) {
            cerr << "Error: Invalid row range for copying." << endl;
            return *this;
        }
        int num_rows = end - start;
        for (int i = 0; i < num_rows; i++) {
            // Đã đổi src.col thành this->col ở điểm đến
            std::copy(src.data.begin() + (start + i) * src.col, 
                      src.data.begin() + (start + i) * src.col + src.col, 
                      this->data.begin() + i * this->col);
        }
    } else if (direction == "col") {
        if (start < 0 || end > src.col || start > end) {
            cerr << "Error: Invalid column range for copying." << endl;
            return *this;
        }
        int num_cols = end - start;
        for (int i = 0; i < src.row; i++) {
            for (int j = 0; j < num_cols; j++) {
                (*this)(i, j) = src(i, start + j);
            }
        }
    } else {
        cerr << "Error: Direction must be 'row' or 'col'." << endl;
    }
    return *this;
}
Mat& Mat::copyexcept(const Mat& src, string direction, int start_except, int end_except) {
    if (direction == "row") {
        if (start_except < 0 || end_except > src.row || start_except > end_except) {
            cerr << "Error: Invalid row range for copying." << endl;
            return *this;
        }
        int num_rows = src.row - (end_except - start_except);
        int dst_row = 0;
        for (int i = 0; i < src.row; i++) {
            // SỬA LỖI: Đổi > thành >=
            if (i < start_except || i >= end_except) {
                // Đã đổi src.col thành this->col ở điểm đến
                std::copy(src.data.begin() + i * src.col, 
                          src.data.begin() + i * src.col + src.col, 
                          this->data.begin() + dst_row * this->col);
                dst_row++;
            }
        }
    } else if (direction == "col") {
        if (start_except < 0 || end_except > src.col || start_except > end_except) {
            cerr << "Error: Invalid column range for copying." << endl;
            return *this;
        }
        int num_cols = src.col - (end_except - start_except);
        for (int i = 0; i < src.row; i++) {
            int dst_col = 0;
            for (int j = 0; j < src.col; j++) {
                // Khối lệnh này của bạn ban đầu đã đúng
                if (j < start_except || j >= end_except) {
                    (*this)(i, dst_col) = src(i, j);
                    dst_col++;
                }
            }
        }
    } else {
        cerr << "Error: Direction must be 'row' or 'col'." << endl;
    }
    return *this;
}
Mat& Mat::operator=(const Mat& other){
    if (this == &other) {
            return *this; 
    }
    if (this->data.size() != other.data.size()) {
            this->data.resize(other.data.size());
    }
    this->row = other.row;
    this->col = other.col;
    std::copy(other.data.begin(), other.data.end(), this->data.begin());
    return *this;
}
Mat& Mat::operator+=(const Mat& other){
    if (other.row == 1 && this->col == other.col) {
        apply_broadcast_row(*this, other, *this, [](float a, float b) { return a + b;});
        return *this;
    }
    if (row != other.row || col != other.col) {
        cerr << "Error: Matrix dimensions must match for addition." << endl;
        return *this;
    }
    apply(*this, other, *this, [](float a, float b) { return a + b; });
    return *this;
}
Mat& Mat::operator-=(const Mat& other){
    if (other.row == 1 && this->col == other.col) {
        apply_broadcast_row(*this, other, *this, [](float a, float b) { return a - b;});
        return *this;
    }
    if (row != other.row || col != other.col) {
        cerr << "Error: Matrix dimensions must match for subtraction." << endl;
        return *this;
    }
    apply(*this, other, *this, [](float a, float b) { return a - b; });
    return *this;
}
Mat& Mat::operator*=(float scalar){
    apply(*this, *this, [scalar](float x) { return x * scalar; });
    return *this;
}
Mat& Mat::transpose() {
    Mat temp(col, row);
    const float* __restrict pSrc = this->data.data();
    float* __restrict pDst = temp.data.data();
    const int block = 32;
    for (int i = 0; i < row; i += block)
        for (int j = 0; j < col; j += block)
            for (int ii = i; ii < min(i + block, row); ii++)
                for (int jj = j; jj < min(j + block, col); jj++)
                    pDst[jj * temp.col + ii] = pSrc[ii * col + jj];
    *this = temp;
    return *this;
}
Mat operator*(const Mat& A, const Mat& B){
    if (A.col != B.row) {
        cerr << "Error: Incompatible dimensions for matrix multiplication." << endl;
        return Mat(0, 0);
    }
    Mat result(A.row, B.col);
    mxm(A, B, result);
    return result;
}
void Loss_History::save(const float loss) {Loss.push_back(loss);}
void Loss_History::print(){
    if (Loss.size()==0) return;
	for (int i = 0;i<Loss.size();i++){
		cout << "\nLoss " << i <<": " << Loss[i];
	}
	cout<<"\nLoss final: "<<Loss[Loss.size() - 1] << " end at "<<Loss.size() <<" epoch\n";
}
void Loss_History::print_final(){if (Loss.size()==0) return; cout<<"\nLoss final: "<<Loss[Loss.size() - 1] << " end at "<<Loss.size() <<" epoch\n";}
void Init_Node(vector<Mat>& W, vector<Mat>& B, vector <int> hidden_nodes, int input_dim, int output_dim){
    for (int i = 0; i < W.size(); i++) {
        if (i == 0) {
            W[i] = Mat(input_dim, hidden_nodes[0]);
            B[i] = Mat(1, hidden_nodes[0]);
        } else if (i == W.size() - 1) {
            W[i] = Mat(hidden_nodes[W.size() - 2], output_dim);
            B[i] = Mat(1, output_dim);
        } else {
            W[i] = Mat(hidden_nodes[i - 1], hidden_nodes[i]);
            B[i] = Mat(1, hidden_nodes[i]);
        }
    }
}
void mxm_block_tilt(const Mat& src1, const Mat& src2, Mat& dst){
    if (src1.col != src2.row || src1.row != dst.row || src2.col != dst.col) {
        cerr << "Error: Incompatible dimensions for matrix multiplication." << endl;
        return;
    }
    int BS = 32;
    int M32 = (dst.col / 32) * 32;
    const float* __restrict pA = src1.data.data();
    const float* __restrict pB = src2.data.data();
    float* __restrict pC = dst.data.data();
    for (int ii = 0; ii < src1.row; ii += BS) {
        for (int jj = 0; jj < M32; jj += BS) {
            for (int i = ii; i < ii + BS && i < src1.row; i++) {
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();
                __m256 c4 = _mm256_setzero_ps();
                for (int kk = 0; kk < src1.col; kk++) {
                    __m256 a_vec = _mm256_set1_ps(pA[i * src1.col + kk]);
                    int b_offset = kk * src2.col + jj;
                    c1 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset),      c1);
                    c2 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset + 8),  c2);
                    c3 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset + 16), c3);
                    c4 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset + 24), c4);
                }
                int c_offset = i * dst.col + jj;
                _mm256_storeu_ps(pC + c_offset,      c1);
                _mm256_storeu_ps(pC + c_offset + 8,  c2);
                _mm256_storeu_ps(pC + c_offset + 16, c3);
                _mm256_storeu_ps(pC + c_offset + 24, c4);
            }
        }
        if (M32 < dst.col) {
            for (int i = ii; i < ii + BS && i < src1.row; i++) {
                int j = M32;
                for (; j + 8 <= dst.col; j += 8) {
                    __m256 c = _mm256_setzero_ps();
                    for (int kk = 0; kk < src1.col; kk++) {
                        __m256 a_vec = _mm256_set1_ps(pA[i * src1.col + kk]);
                        c = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + kk * src2.col + j), c);
                    }
                    _mm256_storeu_ps(pC + i * dst.col + j, c);
                }
                for (; j < dst.col; j++) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < src1.col; kk++) {
                        acc += pA[i * src1.col + kk] * pB[kk * src2.col + j];
                    }
                    pC[i * dst.col + j] = acc;
                }
            }
        }
    }
}
void mxm(const Mat& src1, const Mat& src2, Mat& dst) {
    if (src1.col != src2.row || src1.row != dst.row || src2.col != dst.col) {
        cerr << "Error: Incompatible dimensions for matrix multiplication." << endl;
        return;
    }

    const float* __restrict pA = src1.data.data();
    const float* __restrict pB = src2.data.data();
    float* __restrict pC = dst.data.data();

    // Mốc chia hết cho 32 để xử lý 4 thanh ghi AVX cùng lúc (4 x 8 float)
    int M32 = (dst.col / 32) * 32;

    for (int i = 0; i < src1.row; i++) {
        
        // 1. Quét phần lõi ma trận (nhảy mỗi bước 32 cột)
        for (int j = 0; j < M32; j += 32) {
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            for (int k = 0; k < src1.col; k++) {
                __m256 a_vec = _mm256_set1_ps(pA[i * src1.col + k]);
                int b_offset = k * src2.col + j;
                c1 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset),      c1);
                c2 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset + 8),  c2);
                c3 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset + 16), c3);
                c4 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset + 24), c4);
            }

            int c_offset = i * dst.col + j;
            _mm256_storeu_ps(pC + c_offset,      c1);
            _mm256_storeu_ps(pC + c_offset + 8,  c2);
            _mm256_storeu_ps(pC + c_offset + 16, c3);
            _mm256_storeu_ps(pC + c_offset + 24, c4);
        }

        // 2. Xử lý phần dư không chia hết cho 32
        int j_remainder = M32;
        
        // Xử lý các khối 8 cột còn lại bằng 1 thanh ghi AVX
        for (; j_remainder + 8 <= dst.col; j_remainder += 8) {
            __m256 c = _mm256_setzero_ps();
            for (int k = 0; k < src1.col; k++) {
                __m256 a_vec = _mm256_set1_ps(pA[i * src1.col + k]);
                c = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + k * src2.col + j_remainder), c);
            }
            _mm256_storeu_ps(pC + i * dst.col + j_remainder, c);
        }

        // Xử lý nốt các cột lẻ cuối cùng (Scalar)
        for (; j_remainder < dst.col; j_remainder++) {
            float acc = 0.0f;
            for (int k = 0; k < src1.col; k++) {
                acc += pA[i * src1.col + k] * pB[k * src2.col + j_remainder];
            }
            pC[i * dst.col + j_remainder] = acc;
        }
    }
}
void mtxm(const Mat& src1, const Mat& src2, Mat& dst) {
    if (src1.row != src2.row || src1.col != dst.row || src2.col != dst.col) {
        cerr << "Error: Incompatible dimensions for matrix multiplication." << endl;
        return;
    }

    int M32 = (dst.col / 32) * 32;
    const float* __restrict pA = src1.data.data();
    const float* __restrict pB = src2.data.data();
    float* __restrict pC = dst.data.data();

    // Vòng lặp ngoài cùng chạy thẳng theo số hàng của ma trận đích C (k)
    for (int k = 0; k < dst.row; k++) {
        
        // 1. Quét phần lõi ma trận với bước nhảy 32 cột
        for (int j = 0; j < M32; j += 32) {
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();

            // Vòng lặp i duyệt qua cột của A (chính là hàng của A^T) và hàng của B
            for (int i = 0; i < src1.row; i++) {
                __m256 a_vec = _mm256_set1_ps(pA[i * src1.col + k]);
                int b_offset = i * src2.col + j;
                
                c1 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset),      c1);
                c2 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset + 8),  c2);
                c3 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset + 16), c3);
                c4 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + b_offset + 24), c4);
            }

            int c_offset = k * dst.col + j;
            _mm256_storeu_ps(pC + c_offset,      c1);
            _mm256_storeu_ps(pC + c_offset + 8,  c2);
            _mm256_storeu_ps(pC + c_offset + 16, c3);
            _mm256_storeu_ps(pC + c_offset + 24, c4);
        }

        // 2. Xử lý phần dư không chia hết cho 32
        int j_remainder = M32;
        
        // Xử lý khối 8 cột
        for (; j_remainder + 8 <= dst.col; j_remainder += 8) {
            __m256 c = _mm256_setzero_ps();
            for (int i = 0; i < src1.row; i++) {
                __m256 a_vec = _mm256_set1_ps(pA[i * src1.col + k]);
                c = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(pB + i * src2.col + j_remainder), c);
            }
            _mm256_storeu_ps(pC + k * dst.col + j_remainder, c);
        }

        // Xử lý cột lẻ cuối cùng
        for (; j_remainder < dst.col; j_remainder++) {
            float acc = 0.0f;
            for (int i = 0; i < src1.row; i++) {
                acc += pA[i * src1.col + k] * pB[i * src2.col + j_remainder];
            }
            pC[k * dst.col + j_remainder] = acc;
        }
    }
}
void mxmt_naive(const Mat& src1, const Mat& src2, Mat& dst){
    if (src1.col != src2.col || src1.row != dst.row || src2.row != dst.col) {
        cerr << "Error: Incompatible dimensions for matrix multiplication." << endl;
        return;
    }
    for (int i = 0;i<dst.row;i++){
        for (int j = 0;j<dst.col;j++){
            dst(i,j) = dot(src1.data.data() + i * src1.col, src2.data.data() + j * src2.col, src1.col);
        }
    }
}
inline float hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);          // Lấy 4 số thấp
    __m128 vhigh = _mm256_extractf128_ps(v, 1);        // Lấy 4 số cao
    vlow  = _mm_add_ps(vlow, vhigh);                   // Cộng chập 4 cao và 4 thấp
    __m128 shuf = _mm_movehl_ps(vlow, vlow);           // Đảo vị trí
    vlow  = _mm_add_ps(vlow, shuf);
    shuf  = _mm_shuffle_ps(vlow, vlow, 0x01);          // Đảo vị trí lần cuối
    vlow  = _mm_add_ps(vlow, shuf);
    return _mm_cvtss_f32(vlow);                        // Trả về số thực cuối cùng
}
void mxmt(const Mat& src1, const Mat& src2, Mat& dst) {
    if (src1.col != src2.col || src1.row != dst.row || src2.row != dst.col) {
        cerr << "Error: Incompatible dimensions for matrix multiplication." << endl;
        return;
    }
    const float* __restrict pA = src1.data.data();
    const float* __restrict pB = src2.data.data();
    float* __restrict pC = dst.data.data();
    int K = src1.col;            // Chiều dài chung (Reduction dimension)
    int K8 = (K / 8) * 8;        // Mốc chia hết cho 8 cho vòng lặp K
    int J4 = (dst.col / 4) * 4;  // Mốc chia hết cho 4 cho vòng lặp J
    for (int i = 0; i < dst.row; i++) {
        const float* ptrA = pA + i * K;
        int j = 0;
        for (; j < J4; j += 4) {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
            const float* ptrB0 = pB + (j + 0) * K;
            const float* ptrB1 = pB + (j + 1) * K;
            const float* ptrB2 = pB + (j + 2) * K;
            const float* ptrB3 = pB + (j + 3) * K;
            int k = 0;
            for (; k < K8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(ptrA + k);
                sum0 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(ptrB0 + k), sum0);
                sum1 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(ptrB1 + k), sum1);
                sum2 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(ptrB2 + k), sum2);
                sum3 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(ptrB3 + k), sum3);
            }
            float s0 = hsum256_ps_avx(sum0);
            float s1 = hsum256_ps_avx(sum1);
            float s2 = hsum256_ps_avx(sum2);
            float s3 = hsum256_ps_avx(sum3);
            for (; k < K; k++) {
                float a_val = ptrA[k];
                s0 += a_val * ptrB0[k];
                s1 += a_val * ptrB1[k];
                s2 += a_val * ptrB2[k];
                s3 += a_val * ptrB3[k];
            }
            pC[i * dst.col + j + 0] = s0;
            pC[i * dst.col + j + 1] = s1;
            pC[i * dst.col + j + 2] = s2;
            pC[i * dst.col + j + 3] = s3;
        }
        for (; j < dst.col; j++) {
            __m256 sum = _mm256_setzero_ps();
            const float* ptrB = pB + j * K;
            int k = 0;
            for (; k < K8; k += 8) {
                 sum = _mm256_fmadd_ps(_mm256_loadu_ps(ptrA + k), _mm256_loadu_ps(ptrB + k), sum);
            }
            float s = hsum256_ps_avx(sum);
            for (; k < K; k++) {
                 s += ptrA[k] * ptrB[k];
            }
            pC[i * dst.col + j] = s;
        }

    }
}
void vector_add(const float* src1,const float* src2,float* dst,int n){
	int i = 0;
	__m256 sum1, sum2, sum3, sum4;
	for(;i<= n - 32;i+=32){
		sum1 = _mm256_add_ps(_mm256_loadu_ps(src1 + i),_mm256_loadu_ps(src2 + i));
		sum2 = _mm256_add_ps(_mm256_loadu_ps(src1 + i + 8),_mm256_loadu_ps(src2 + i + 8));
		sum3 = _mm256_add_ps(_mm256_loadu_ps(src1 + i + 16),_mm256_loadu_ps(src2 + i + 16));
		sum4 = _mm256_add_ps(_mm256_loadu_ps(src1  + i + 24),_mm256_loadu_ps(src2 + i + 24));
		_mm256_storeu_ps(dst + i,sum1);
		_mm256_storeu_ps(dst + i + 8,sum2);
		_mm256_storeu_ps(dst + i + 16,sum3);
		_mm256_storeu_ps(dst + i + 24,sum4);
	}
    for (; i <= n - 8; i += 8) {
        sum1 = _mm256_add_ps(_mm256_loadu_ps(src1 + i),_mm256_loadu_ps(src2 + i));
        _mm256_storeu_ps(dst + i,sum1);
    }
    for (;i<n;i++){
    	dst[i] = src1[i] + src2[i];	
	}
}
void vector_sub(const float* src1,const float* src2,float* dst,int n){
	int i = 0;
	__m256 sum1, sum2, sum3, sum4;
	for(;i<= n - 32;i+=32){
		sum1 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i),_mm256_loadu_ps(src2 + i));
		sum2 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i + 8),_mm256_loadu_ps(src2 + i + 8));
		sum3 = _mm256_sub_ps(_mm256_loadu_ps(src1  + i + 16),_mm256_loadu_ps(src2 + i + 16));
		sum4 = _mm256_sub_ps(_mm256_loadu_ps(src1  + i + 24),_mm256_loadu_ps(src2 + i + 24));
		_mm256_storeu_ps(dst + i,sum1);
		_mm256_storeu_ps(dst + i + 8,sum2);
		_mm256_storeu_ps(dst + i + 16,sum3);
		_mm256_storeu_ps(dst + i + 24,sum4);
	}
    for (; i <= n - 8; i += 8) {
        sum1 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i),_mm256_loadu_ps(src2 + i));
        _mm256_storeu_ps(dst + i,sum1);
    }
    for (;i<n;i++){
    	dst[i] = src1[i] - src2[i];	
	}
}
float sum_elements_abs(const float *src, int n){
    int i = 0;float result = 0;
    __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 sum_v1 = _mm256_setzero_ps();
    __m256 sum_v2 = _mm256_setzero_ps();
    __m256 sum_v3 = _mm256_setzero_ps();
    __m256 sum_v4 = _mm256_setzero_ps();
    int total_size = n;
    for (; i <= total_size - 32; i += 32) {
        __m256 abs_data1 = _mm256_and_ps(_mm256_loadu_ps(src + i), abs_mask);
        __m256 abs_data2 = _mm256_and_ps(_mm256_loadu_ps(src + i + 8), abs_mask);
        __m256 abs_data3 = _mm256_and_ps(_mm256_loadu_ps(src + i + 16), abs_mask);
        __m256 abs_data4 = _mm256_and_ps(_mm256_loadu_ps(src + i + 24), abs_mask);
        sum_v1 = _mm256_add_ps(sum_v1, abs_data1);
        sum_v2 = _mm256_add_ps(sum_v2, abs_data2);
        sum_v3 = _mm256_add_ps(sum_v3, abs_data3);
        sum_v4 = _mm256_add_ps(sum_v4, abs_data4);
    }
	sum_v2 = _mm256_add_ps(_mm256_add_ps(sum_v1, sum_v2),_mm256_add_ps(sum_v3, sum_v4));
    for(; i < total_size - 8; i+=8) {
        sum_v1 = _mm256_and_ps(_mm256_loadu_ps(src + i), abs_mask);
        sum_v2 = _mm256_add_ps(sum_v2, sum_v1);
    }
    __m128 low  = _mm256_castps256_ps128(sum_v2);
    __m128 high = _mm256_extractf128_ps(sum_v2, 1);
    __m128 s = _mm_add_ps(low, high);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (; i < total_size; i++) {
        result += fabsf(src[i]);
    }
    return result;
}
float sum_elements_abs(const float *src1,const float *src2, int n){
    int i = 0;float result = 0;
    __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 sum_v1 = _mm256_setzero_ps();
    __m256 sum_v2 = _mm256_setzero_ps();
    __m256 sum_v3 = _mm256_setzero_ps();
    int total_size = n;
    for (; i <= total_size - 24; i += 24) {
        __m256 abs_data1 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i));
        __m256 abs_data2 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i + 8), _mm256_loadu_ps(src2 + i + 8));
        __m256 abs_data3 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i + 16), _mm256_loadu_ps(src2 + i + 16));
        abs_data1 = _mm256_and_ps(abs_data1, abs_mask);
        abs_data2 = _mm256_and_ps(abs_data2, abs_mask);
        abs_data3 = _mm256_and_ps(abs_data3, abs_mask);
        sum_v1 = _mm256_add_ps(sum_v1, abs_data1);
        sum_v2 = _mm256_add_ps(sum_v2, abs_data2);
        sum_v3 = _mm256_add_ps(sum_v3, abs_data3);
    }
	sum_v2 = _mm256_add_ps(_mm256_add_ps(sum_v1, sum_v2),sum_v3);
    for(; i < total_size - 8; i+=8) {
        sum_v3 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i), _mm256_loadu_ps(src2 + i));
        sum_v1 = _mm256_and_ps(sum_v3, abs_mask);
        sum_v2 = _mm256_add_ps(sum_v2, sum_v1);
    }
    __m128 low  = _mm256_castps256_ps128(sum_v2);
    __m128 high = _mm256_extractf128_ps(sum_v2, 1);
    __m128 s = _mm_add_ps(low, high);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (; i < total_size; i++) {
        result += fabsf(src1[i] - src2[i]);
    }
    return result;
}
float sum_elements(const float *src, int n) {
	int i = 0;float result = 0;
	__m256 sum1 = _mm256_setzero_ps();
	__m256 sum2 = _mm256_setzero_ps();
	__m256 sum3 = _mm256_setzero_ps();
	__m256 sum4 = _mm256_setzero_ps();
	for(;i<= n - 32;i+=32){
		sum1 = _mm256_add_ps(_mm256_loadu_ps(src + i),sum1);
		sum2 = _mm256_add_ps(_mm256_loadu_ps(src + i + 8),sum2);
		sum3 = _mm256_add_ps(_mm256_loadu_ps(src + i + 16),sum3);
		sum4 = _mm256_add_ps(_mm256_loadu_ps(src     + i + 24),sum4);
	}
	__m256 sum = _mm256_add_ps(_mm256_add_ps(sum1, sum2),_mm256_add_ps(sum3, sum4));
    for (; i <= n - 8; i += 8) {
        sum = _mm256_add_ps(_mm256_loadu_ps(src + i), sum);
    }
	__m128 low  = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 s = _mm_add_ps(low, high);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (;i<n;i++){
    	result += src[i];	
	}
    return result;
}
float dot(const float* src1, const float* src2, int n) {
    int i = 0;
    float result = 0;
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
	int total_size = n;
	for(;i<= total_size - 32;i+=32){
		sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + i),_mm256_loadu_ps(src2 + i),sum1);
		sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + i + 8),_mm256_loadu_ps(src2 + i + 8),sum2);
		sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + i + 16),_mm256_loadu_ps(src2 + i + 16),sum3);
		sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + i + 24),_mm256_loadu_ps(src2 + i + 24),sum4);
	}
	__m256 sum = _mm256_add_ps(_mm256_add_ps(sum1, sum2),_mm256_add_ps(sum3, sum4));
    for (; i <= total_size - 8; i += 8) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + i),_mm256_loadu_ps(src2 + i), sum);
    }
	__m128 low  = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 s = _mm_add_ps(low, high);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (;i<total_size;i++){
    	result += src1[i]*src2[i];	
	}
    return result;
}
float dist(const float* src1, const float* src2, int n){
    int i = 0;float result = 0;
	__m256 sum1 = _mm256_setzero_ps();
	__m256 sum2 = _mm256_setzero_ps();
	__m256 sum3 = _mm256_setzero_ps();
	__m256 sum4 = _mm256_setzero_ps();
	__m256 sub1, sub2, sub3, sub4;
	for(;i<= n - 32;i+=32){
		sub1 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i),_mm256_loadu_ps(src2 + i));
		sub2 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i + 8),_mm256_loadu_ps(src2 + i + 8));
		sub3 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i + 16),_mm256_loadu_ps(src2 + i + 16));
		sub4 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i + 24),_mm256_loadu_ps(src2 + i + 24));
		sum1 = _mm256_fmadd_ps(sub1,sub1,sum1);
		sum2 = _mm256_fmadd_ps(sub2,sub2,sum2);
		sum3 = _mm256_fmadd_ps(sub3,sub3,sum3);
		sum4 = _mm256_fmadd_ps(sub4,sub4,sum4);
	}
	sum1 = _mm256_add_ps(_mm256_add_ps(sum1, sum2),_mm256_add_ps(sum3, sum4));
    for (; i <= n - 8; i += 8) {
        sub1 = _mm256_sub_ps(_mm256_loadu_ps(src1 + i),_mm256_loadu_ps(src2 + i));
        sum1 = _mm256_fmadd_ps(sub1,sub1,sum1);
    }
	__m128 low  = _mm256_castps256_ps128(sum1);
    __m128 high = _mm256_extractf128_ps(sum1, 1);
    __m128 s = _mm_add_ps(low, high);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (;i<n;i++){
    	result += (src1[i]-src2[i])*(src1[i]-src2[i]);	
	}
    return result;
}
float dist(const float* src, const float scalar, int n){
    int i = 0;float result = 0;
	__m256 sum1 = _mm256_setzero_ps();
	__m256 sum2 = _mm256_setzero_ps();
	__m256 sum3 = _mm256_setzero_ps();
	__m256 sum4 = _mm256_setzero_ps();
	__m256 sub1, sub2, sub3, sub4;
	__m256 one = _mm256_set1_ps(scalar);
	for(;i<= n - 32;i+=32){
		sub1 = _mm256_sub_ps(_mm256_loadu_ps(src + i),one);
		sub2 = _mm256_sub_ps(_mm256_loadu_ps(src + i + 8),one);
		sub3 = _mm256_sub_ps(_mm256_loadu_ps(src + i + 16),one);
		sub4 = _mm256_sub_ps(_mm256_loadu_ps(src + i + 24),one);
		sum1 = _mm256_fmadd_ps(sub1,sub1,sum1);
		sum2 = _mm256_fmadd_ps(sub2,sub2,sum2);
		sum3 = _mm256_fmadd_ps(sub3,sub3,sum3);
		sum4 = _mm256_fmadd_ps(sub4,sub4,sum4);
	}
	sum1 = _mm256_add_ps(_mm256_add_ps(sum1, sum2),_mm256_add_ps(sum3, sum4));
    for (; i <= n - 8; i += 8) {
        sub1 = _mm256_sub_ps(_mm256_loadu_ps(src + i),one);
        sum1 = _mm256_fmadd_ps(sub1,sub1,sum1);
    }
	__m128 low  = _mm256_castps256_ps128(sum1);
    __m128 high = _mm256_extractf128_ps(sum1, 1);
    __m128 s = _mm_add_ps(low, high);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    result = _mm_cvtss_f32(s);
    for (;i<n;i++){
    	result += (src[i]-scalar)*(src[i]-scalar);	
	}
    return result;
}
float norm(const float* src, int n){return sqrtf(dot(src, src, n));}
float mean(const float* src, int n){return sum_elements(src, n) / n;}
float var(const float* src, const float* mean_vec, int n){
    return dist(src, mean_vec, n) / n;
}
void Sign_Matrix(const Mat& src, Mat& dst) {
    apply(src, dst, [](float x) { return (float)(x>0) - (float)(x<0); });
}
void UpdateParameters(const Mat& dsrc, const Mat& src, Mat& dst, float learning_rate, float parameter_scaling ) {
    apply(dsrc, src, dst, [learning_rate, parameter_scaling](float d, float s) { return s * parameter_scaling - learning_rate * d ; });
}
void ReLU(const Mat& src, Mat& dst){
    apply(src, dst, [](float x) { return x > 0.0f ? x : 0.0f; });
}
void dReLU(const Mat& cmp, const Mat& src, Mat& dst){
    apply(cmp, src, dst, [](float c, float s) { return (c > 0.0f) ? s : 0.0f;});
}
void Sigmoid(const Mat& src, Mat& dst){
    apply(src, dst, [](float x) { return 1.0f / (1.0f + expf(-x)); });
}
void Softmax(const Mat& src, Mat& dst){
    for (int i = 0; i < src.row; i++){
        float max_val = src(i, 0);
        float sum = 0.0f;
        for(int j = 1; j < src.col; j++)
            if(src(i, j) > max_val) max_val = src(i, j);
        for(int j = 0; j < src.col; j++){
            dst(i, j) = expf(src(i, j) - max_val);
            sum += dst(i, j);
        }
        float inv_sum = 1.0f / sum;
        for(int j = 0; j < src.col; j++)
            dst(i, j) *= inv_sum;
    }
}
void Hadamard(const Mat& src1, const Mat& src2, Mat& dst) {
    apply(src1, src2, dst, [](float a, float b) { return a * b; });
}
void Hadamard(const float* src1, const float* src2, float* dst,int vec_size) {
    const float* __restrict psrc1 = src1;
    const float* __restrict psrc2 = src2;
    float* __restrict pdst = dst;
    for (int i = 0; i < vec_size; ++i) {
        pdst[i] = psrc1[i] * psrc2[i];
    }
}
void Element_Wise_Div(const Mat& src1, const Mat& src2, Mat& dst) {
    apply(src1, src2, dst, [](float a, float b) { return a / b; });
}
void Hadamard_Broadcast_Row(const Mat& src1, const Mat& src2, Mat& dst) {
    apply_broadcast_row(src1, src2, dst, [](float a, float b) { return a * b; });
}
void Sum_Rows(const Mat& src, Mat& dst){
    // Bắt buộc phải reset các phần tử của dst về 0 trước khi cộng dồn
    dst.set_zeros(); 
    // Quét qua từng hàng (sample)
    for (int i = 0; i < src.row; i++) {
        vector_add(dst.data.data(),src.data.data() + i * src.col, dst.data.data(), src.col);
    }
}
void Sum_Cols(const Mat& src, Mat& dst){
    for (int i = 0; i < src.row ; i++){
        dst.data[i] = sum_elements(src.data.data() + i * src.col,src.col);
    } 
}
float Precision(float TP, float FP){
    if (TP + FP == 0) return 0.0f;
    else return (float)TP/(TP + FP);
}
float Recall(float TP, float FN){
    if (TP + FN == 0) return 0.0f;
    else return (float)TP/(TP + FN);
}
float F1_Score(float precision, float recall){
    if (precision + recall == 0) return 0.0f;
    else return 2 * precision * recall / (precision + recall);
}
float MSE(const Mat& Y, const Mat& Z) {
    return dist(Y.data.data(), Z.data.data(), Y.size()) / (2 * Y.row);
}
float MAE(const Mat& Y, const Mat& Z) {
    return sum_elements_abs(Y.data.data(), Z.data.data(), Y.size()) / Y.row;
}
float BCE(const Mat& Y, const Mat& P) {
    size_t size = Y.size();
    const float* y_ptr = Y.data.data();
    const float* p_ptr = P.data.data();
    double total_loss = 0.0;
    for (size_t i = 0; i < size; ++i) {
        total_loss += y_ptr[i] * logf(p_ptr[i] + 1e-7f) + (1 - y_ptr[i]) * logf(1 - p_ptr[i] + 1e-7f) ;
    }
    return static_cast<float>(total_loss / Y.row);
}
float CCE(const Mat& Y, const Mat& P){
    const float* y_ptr = Y.data.data();
    const float* p_ptr = P.data.data();
    double total_loss = 0.0;
    for (int i = 0; i < Y.row; i++) {
        for (int j = 0; j < Y.col; j++) {
            if (y_ptr[i * Y.col + j] > 0.0f)
                total_loss -= y_ptr[i * Y.col + j] * logf(p_ptr[i * P.col + j] + 1e-7f);
        }
    }
    return static_cast<float>(total_loss / Y.row);
}
float MSE(const Mat& Y, const Mat& Z, string regularization , float lambda, const Mat& W) {
    if (regularization == "L1") {
        return dist(Y.data.data(), Z.data.data(), Y.size()) / (2 * Y.row) + lambda * sum_elements_abs(W.data.data(), W.size()) / Y.row;
    }
    else if (regularization == "L2") {
        return dist(Y.data.data(), Z.data.data(), Y.size()) / (2 * Y.row) + 0.5f * lambda * dot(W.data.data(), W.data.data(), W.size()) / Y.row;
    }
    return dist(Y.data.data(), Z.data.data(), Y.size()) / (2 * Y.row);
}
float MAE(const Mat& Y, const Mat& Z, string regularization , float lambda, const Mat& W) {
    if (regularization == "L1") {
        return sum_elements_abs(Y.data.data(), Z.data.data(), Y.size()) / Y.row + lambda * sum_elements_abs(W.data.data(), W.size()) / Y.row;
    }
    else if (regularization == "L2") {
        return sum_elements_abs(Y.data.data(), Z.data.data(), Y.size()) / Y.row + 0.5f * lambda * dot(W.data.data(), W.data.data(), W.size()) / Y.row;
    }
    return sum_elements_abs(Y.data.data(), Z.data.data(), Y.size()) / Y.row;
}
float BCE(const Mat& Y, const Mat& P, string regularization , float lambda, const Mat& W) {
    size_t size = Y.size();
    const float* y_ptr = Y.data.data();
    const float* p_ptr = P.data.data();
    double total_loss = 0.0;
    for (size_t i = 0; i < size; ++i) {
        total_loss += y_ptr[i] * logf(p_ptr[i] + 1e-7f) + (1 - y_ptr[i]) * logf(1 - p_ptr[i] - 1e-7f) ;
    }
    return static_cast<float>(total_loss / Y.row);
    if (regularization == "L1") {
        total_loss += lambda * sum_elements_abs(W.data.data(), W.size()) / Y.row;
    }
    else if (regularization == "L2") {
        total_loss += 0.5f * lambda * dot(W.data.data(), W.data.data(), W.size()) / Y.row;
    }
    return static_cast<float>(total_loss / Y.row);
}
float CCE(const Mat& Y, const Mat& P, string regularization, float lambda, const Mat& W){
    const float* y_ptr = Y.data.data();
    const float* p_ptr = P.data.data();
    double total_loss = 0.0;
    for (int i = 0; i < Y.row; i++) {
        for (int j = 0; j < Y.col; j++) {
            if (y_ptr[i * Y.col + j] > 0.0f)
                total_loss -= y_ptr[i * Y.col + j] * logf(p_ptr[i * P.col + j] + 1e-7f);
        }
    }
    if (regularization == "L1") {
        total_loss += lambda * sum_elements_abs(W.data.data(), W.size()) / Y.row;
    }
    else if (regularization == "L2") {
        total_loss += 0.5f * lambda * dot(W.data.data(), W.data.data(), W.size()) / Y.row;
    }
    return static_cast<float>(total_loss / Y.row);
}
float Loss_Cal(const Mat& Y, const Mat& Y_hat, string loss_type){
    float val;
    if (loss_type == "MSE") val = MSE(Y, Y_hat);
    else if (loss_type == "MAE") val = MAE(Y, Y_hat);
    else if (loss_type == "BCE") val = BCE(Y, Y_hat);
    else if (loss_type == "CCE") val = CCE(Y, Y_hat);
    return val;
}
float VIF_Cal(const float* X_true,const float *X_pred ,float X_mean,int n){
    float RSS, TSS;
	RSS = dist(X_true,X_pred,n);
	TSS = dist(X_true,X_mean,n);
	return TSS/RSS;
}
void VIF(const Mat& X, Mat& VIF_mat){
    Mat X_scaled(X.row, X.col);
    Mat global_mean(1, X.col);
    Mat global_std(1, X.col);
    FeatureScaling(X, X_scaled, global_mean, global_std);
    Mat X_Mask(X.row, X.col - 1);
    Mat X_Vector(X.row, 1);
    Mat W(X.col - 1, 1);
    Mat B(1, 1);
    Mat X_pred(X.row, 1);
    for (int i = 0; i < VIF_mat.col; i++) {
        W.set_zeros();
        B.set_zeros();
        for (int j = 0; j < X.row; j++) {
            X_Vector(j, 0) = X_scaled(j, i); 
            int col = 0;
            for (int e = 0; e < X.col; e++) {
                if (e != i) {
                    X_Mask(j, col++) = X_scaled(j, e);
                }
            }
        }
        Train_LN(X_Mask, X_Vector, W, B, 0.25, 200, "MSE");
        mxm(X_Mask, W, X_pred);
        X_pred += B;
        VIF_mat(0, i) = VIF_Cal(X_Vector.data.data(), X_pred.data.data(), 0.0f, X_Vector.size());
    }
}
void ShowVIF(const Mat& X){
    Mat VIF_mat(1, X.col);
    VIF(X, VIF_mat);
    cout << "\nVIF values for each feature:\n";
    for (int i = 0; i < VIF_mat.col; i++) {
        cout << "Feature " << i << ": " << VIF_mat(0, i) << endl;
    }
}
void ModelEvaluation(const Mat& Y_true, const Mat& Y_pred, string loss_type, float threshold){
    if (loss_type == "BCE" || loss_type == "CCE"){
        float precision , recall , f1;
        float macro_precision = 0, macro_recall = 0, macro_f1 = 0;
        Mat Y_hat (Y_pred.row,Y_pred.col);
        Mat ConfusionMat(Y_pred.col, Y_pred.col);
        if (loss_type == "BCE") apply(Y_pred, Y_hat, [threshold](float x) {return (x > threshold) ? 1.0f : 0.0f;});
        else {
            for (int i = 0; i < Y_hat.row; i++) {
		        int pred_class = 0;
                for (int j = 1; j < Y_hat.col; j++) if (Y_pred(i, j) > Y_pred(i, pred_class)) pred_class = j;
                Y_hat(i,pred_class) = 1.0f; 
            }
            for (int i = 0;i < Y_true.row;i++){
                int idx_ytrue = -1, idx_yhat = -1;
                for (int j = 0;j < Y_true.col; j++){
                    if (Y_true(i,j)) idx_ytrue = j;
                    if (Y_hat(i,j)) idx_yhat = j;
                    if (idx_yhat != -1 && idx_ytrue != -1) break;
                }
                ConfusionMat(idx_ytrue , idx_yhat)++;
            }
            int total_sum = sum_elements(ConfusionMat.data.data(), ConfusionMat.size());
            Mat soc (1, ConfusionMat.row);Sum_Cols(ConfusionMat,soc);
            Mat sor (1, ConfusionMat.col);Sum_Rows(ConfusionMat,sor);
            for (int i = 0; i < ConfusionMat.row; i++){
                int tp = (int)ConfusionMat(i,i);
                int fp = sor.data[i] - tp;
                int fn = soc.data[i] - tp;
                int tn = total_sum - tp - fp - fn;
                precision = Precision(tp, fp); macro_precision += precision;
                recall = Recall(tp, fn); macro_recall += recall;
                f1 = F1_Score(precision, recall); macro_f1 += f1;
            }
            cout << "\nMacro Precision: " << (float)macro_precision / ConfusionMat.row << endl;
            cout << "Macro Recall: " << (float)macro_recall / ConfusionMat.row << endl;
            cout << "Macro F1-Score: " << (float)macro_f1 / ConfusionMat.row << endl;
        }
    }
    else if (loss_type == "MSE" || loss_type == "MAE"){
        float mse = MSE(Y_true, Y_pred);
        float mae = MAE(Y_true, Y_pred);
        cout << "\nRoot Mean Squared Error: " << sqrtf(2* mse) << endl;
        cout << "Mean Absolute Error: " << mae << endl;
    }
}
float ModelEvaluation(const Mat& Y_true, const Mat& Y_pred, string loss_type, string eval_type, float threshold){
    if (loss_type == "BCE" || loss_type == "CCE"){
        float precision , recall , f1;
        float macro_precision = 0, macro_recall = 0, macro_f1 = 0;
        Mat Y_hat (Y_pred.row,Y_pred.col);
        Mat ConfusionMat(Y_pred.col, Y_pred.col);
        if (loss_type == "BCE") apply(Y_pred, Y_hat, [threshold](float x) {return (x > threshold) ? 1.0f : 0.0f;});
        else {
            for (int i = 0; i < Y_hat.row; i++) {
		        int pred_class = 0;
                for (int j = 1; j < Y_hat.col; j++) if (Y_pred(i, j) > Y_pred(i, pred_class)) pred_class = j;
                Y_hat(i,pred_class) = 1.0f; 
            }
            for (int i = 0;i < Y_true.row;i++){
                int idx_ytrue = -1, idx_yhat = -1;
                for (int j = 0;j < Y_true.col; j++){
                    if (Y_true(i,j)) idx_ytrue = j;
                    if (Y_hat(i,j)) idx_yhat = j;
                    if (idx_yhat != -1 && idx_ytrue != -1) break;
                }
                ConfusionMat(idx_ytrue , idx_yhat)++;
            }
            int total_sum = sum_elements(ConfusionMat.data.data(), ConfusionMat.size());
            Mat soc (1, ConfusionMat.row);Sum_Cols(ConfusionMat,soc);
            Mat sor (1, ConfusionMat.col);Sum_Rows(ConfusionMat,sor);
            for (int i = 0; i < ConfusionMat.row; i++){
                int tp = (int)ConfusionMat(i,i);
                int fp = sor.data[i] - tp;
                int fn = soc.data[i] - tp;
                int tn = total_sum - tp - fp - fn;
                precision = Precision(tp, fp); macro_precision += precision;
                recall = Recall(tp, fn); macro_recall += recall;
                f1 = F1_Score(precision, recall); macro_f1 += f1;
            }
            if (eval_type == "precision" ) {
                cout << "\nMacro Precision: " << (float)macro_precision / ConfusionMat.row << endl; 
                return (float)macro_precision / ConfusionMat.row;
            }
            else if (eval_type == "recall" ) {
                cout << "\nMacro Recall: " << (float)macro_recall / ConfusionMat.row << endl;
                return (float)macro_recall / ConfusionMat.row;
            }
            else if (eval_type == "f1_score") {
                cout << "\nMacro F1-Score: " << (float)macro_f1 / ConfusionMat.row << endl;
                return (float)macro_f1 / ConfusionMat.row;
            }
            else return -1e3f;
        }
    }
    else if (loss_type == "RMSE") {
        float rmse = sqrtf(2* MSE(Y_true, Y_pred));
        cout << "\nRoot Mean Squared Error: " << rmse << endl;
        return rmse;
    }
    else if (loss_type == "MAE") {
        float mae = MAE(Y_true, Y_pred);
        cout << "Mean Absolute Error: " << mae << endl;
        return mae;
    }
}
void RandNormal(vector<float>& src, float mu, float sigma){
    static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    static std::normal_distribution<float> dis(mu, sigma);
    dis.param(std::normal_distribution<float>::param_type(mu, sigma));
    int total_size = src.size();
    for (int i = 0; i < total_size; ++i) {
        src[i] = dis(gen);
    }
}
void RandBer(vector<float>& dst, float p_keep){
    static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    float scale = 1.0f/p_keep;
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] = (dist(gen) < p_keep) ? scale : 0.0f;
    }
}
float RandUni(float a, float b){
    static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_real_distribution<float> dis(a, b);
    return dis(gen);
}
void StartTimer(){start_timer = high_resolution_clock::now();}
void StopTimer(){stop_timer = high_resolution_clock::now();} 
void PrintTimer() {
    auto duration = duration_cast<milliseconds>(stop_timer - start_timer);
    cout << "\nThoi gian thuc thi: " << duration.count() << " ms" << endl;
}
void ShowSoftmaxPredict(const Mat& Y_Pred, const Mat& Y_Test){
    cout << "\nSample  Predicted  Exact  Probabilities:\n";
	int cnt = 0;
	for (int i = 0; i < Y_Pred.row; i++) {
		int pred_class = 0;
		for (int j = 1; j < Y_Pred.col; j++)
			if (Y_Pred(i, j) > Y_Pred(i, pred_class)) pred_class = j;

		int true_class = 0;
		for (int j = 1; j < Y_Test.col; j++)
			if (Y_Test(i, j) > Y_Test(i, true_class)) true_class = j;
		if (pred_class == true_class) cnt++;
		cout << "  [" << i << "]  pred=" << pred_class << "  true=" << true_class << "  prob=[";
		for (int j = 0; j < Y_Pred.col; j++) {
			cout << fixed << setprecision(3) << Y_Pred(i, j);
			if (j < Y_Pred.col - 1) cout << ", ";
		}
		cout << "]" << (pred_class == true_class ? "  OK" : "  X") << "\n";
	}
	cout << "\nCorrect Predictions: " << cnt << "/" << Y_Test.row << " (" << 100.0f * cnt / Y_Test.row << "%)\n";
}
int EarlyStop(const Mat& X_Val, const Mat& Y_Val, vector <Mat>& W,vector <Mat>& B, string loss_type, int patience, int epoch){
    static int wait_count = 0;
    static float best_val_loss = 1e9f;
    static Mat Y_Pred;
    static vector <Mat> Best_W;
    static vector <Mat> Best_B;
    if (epoch == 0){
        wait_count = 0;
        best_val_loss = 1e9f;
        Y_Pred = Mat(Y_Val.row, Y_Val.col);
        Best_W.resize(W.size());
        Best_B.resize(B.size());
    }
    Forward_Pass_ReLU(X_Val, W, B, Y_Pred, loss_type);
    float current_val_loss = Loss_Cal(Y_Val, Y_Pred, loss_type);
    if (current_val_loss < best_val_loss) {
            // Lập kỷ lục mới -> Reset bộ đếm và backup trọng số
            best_val_loss = current_val_loss;
            wait_count = 0;
        for(size_t i = 0; i < W.size(); i++) {
            Best_W[i] = W[i];
            Best_B[i] = B[i];
        }
    }
    else {
        wait_count++;
        if (wait_count >= patience){
            cout << "\n[Early Stopping] at epoch " << epoch <<endl;
            for(size_t i = 0; i < W.size(); i++) {
                W[i] = Best_W[i];
                B[i] = Best_B[i];
            }
            return 1;        
        }
    }
    return 0;
}
void FeatureScaling(const Mat& src, Mat& dst, Mat& mean_mat, Mat& std_mat, string loss_type) {
    if (&src == &dst) {
        Mat Temp(src.col, src.row);
        src.transpose(Temp);
        dst = Temp;
    }
    else {
        std::swap(dst.row,dst.col);
        src.transpose(dst);
    }
    Mat mean_temp_mat(1, dst.col);
    for (int i = 0; i < dst.row; i++) {
        mean_mat(0, i) = mean(&dst(i, 0), dst.col);
        mean_temp_mat.set_ones(mean_mat(0, i));
        std_mat(0, i) = sqrt(var(&dst(i, 0), &mean_temp_mat(0, 0), dst.col));
        std_mat(0, i) = 1.0f / (std_mat(0, i) + 1e-8f);
    }
    dst.transpose();
    dst -= mean_mat;
    Hadamard_Broadcast_Row(dst, std_mat, dst);
}
void FeatureScaling(Mat& dst, Mat& mean_mat, Mat& inv_std_mat){
    dst -= mean_mat;
    Hadamard_Broadcast_Row(dst, inv_std_mat, dst);
}
void Rescale_Weight(Mat& W, Mat& B, Mat& mean_mat, Mat& inv_std_mat){
    W.transpose();
    Mat Temp(1, W.row);
    Hadamard(mean_mat.data.data(), inv_std_mat.data.data(), mean_mat.data.data(), W.col);
    for (int k = 0;k < W.row; k++) {
        Temp(0, k) = dot(mean_mat.data.data(), &W(k, 0), W.col);
    }
    Hadamard_Broadcast_Row(W, inv_std_mat, W);
    W.transpose();
    B -= Temp;
}
void Rescale_Y(Mat& Y_Pred, Mat& Y_mean, Mat& Y_inv_std){
    for (int i = 0; i < Y_Pred.row; i++) {
        for (int j = 0; j < Y_Pred.col; j++) {
            Y_Pred(i, j) = (Y_Pred(i, j) / Y_inv_std(0, j)) + Y_mean(0, j); 
        }
    }
}
void Train_LN(const Mat& X, const Mat& Y, Mat& W, Mat& B, float learning_rate, int epochs, Loss_History& history, string loss_type,
     string regularization, float lambda) {
    int x_row = X.row;
    Mat Y_pred(Y.row, Y.col);
    Mat dW(W.row, W.col);
    Mat dB(B.row, B.col);
    Mat ones = Mat::ones(Y.row, 1);
    float alpha = learning_rate / static_cast<float>(x_row);
    float parameter_scaling = 1.0f - lambda * alpha;
    if(loss_type == "MSE"){
        for (int epoch = 0; epoch < epochs; epoch++) {
            mxm(X, W, Y_pred);
            Y_pred += B;
            history.save(MSE(Y, Y_pred, regularization, lambda, W));
            Y_pred -= Y;
            mtxm(X, Y_pred, dW);
            mtxm(ones, Y_pred, dB);
            UpdateParameters(dW, W, W, alpha, parameter_scaling);
            UpdateParameters(dB, B, B, alpha);
        }
    } else if(loss_type == "MAE"){
        for (int epoch = 0; epoch < epochs; epoch++) {
            mxm(X, W, Y_pred);
            Y_pred += B;
            history.save(MAE(Y, Y_pred, regularization, lambda, W));
            Y_pred -= Y;
            Sign_Matrix(Y_pred, Y_pred);
            mtxm(X, Y_pred, dW);
            mtxm(ones, Y_pred, dB);
            UpdateParameters(dW, W, W, alpha, parameter_scaling);
            UpdateParameters(dB, B, B, alpha);
        }
    } else {
        cerr << "Error: Unsupported loss type. Use 'MSE' or 'MAE'." << endl;
    }
}
void Train_LN(const Mat& X, const Mat& Y, Mat& W, Mat& B, float learning_rate, int epochs, string loss_type) {
    int x_row = X.row;
    Mat Y_pred(Y.row, Y.col);
    Mat dW(W.row, W.col);
    Mat dB(B.row, B.col);
    Mat ones = Mat::ones(Y.row, 1);
    float alpha = learning_rate / static_cast<float>(x_row);
    if(loss_type == "MSE"){
        for (int epoch = 0; epoch < epochs; epoch++) {
            mxm(X, W, Y_pred);
            Y_pred += B;
            Y_pred -= Y;
            mtxm(X, Y_pred, dW);
            mtxm(ones, Y_pred, dB);
            UpdateParameters(dW, W, W, alpha);
            UpdateParameters(dB, B, B, alpha);
        }
    } else if(loss_type == "MAE"){
        for (int epoch = 0; epoch < epochs; epoch++) {
            mxm(X, W, Y_pred);
            Y_pred += B;
            Y_pred -= Y;
            Sign_Matrix(Y_pred, Y_pred);
            mtxm(X, Y_pred, dW);
            mtxm(ones, Y_pred, dB);
            UpdateParameters(dW, W, W, alpha);
            UpdateParameters(dB, B, B, alpha);
        }
    } else {
        cerr << "Error: Unsupported loss type. Use 'MSE' or 'MAE'." << endl;
    }
}
void Train_LG_BIN(const Mat& X, const Mat& Y, Mat& W, Mat& B, float learning_rate, int epochs, Loss_History& history,
     string regularization, float lambda){
    int x_row = X.row;
    Mat Z(Y.row, Y.col);
    Mat dW(W.row, W.col);
    Mat dB(B.row, B.col);
    Mat ones = Mat::ones(Y.row, 1);
    float alpha = learning_rate / static_cast<float>(x_row);
    float parameter_scaling = 1.0f - lambda * alpha;
    for (int epoch = 0; epoch < epochs; epoch++) {
            mxm(X, W, Z);
            Z += B;
            Sigmoid(Z, Z);
            history.save(BCE(Y, Z, regularization, lambda, W));
            Z-= Y;
            mtxm(X, Z, dW);
            mtxm(ones, Z, dB);
            UpdateParameters(dW, W, W, alpha, parameter_scaling);
            UpdateParameters(dB, B, B, alpha);
    }
}
void Train_MLP(const Mat& X, const Mat& Y, vector<Mat>& W, vector<Mat>& B, vector <int> hidden_nodes, float learning_rate, int epochs, Loss_History& history, string loss_type, string regularization, float lambda){
    // Init dW, dB truoc de W duoc cap nhat dung kich thuoc
    vector <Mat> dW(W.size());
    vector <Mat> dB(B.size());
    Init_Node(dW, dB, hidden_nodes, X.col, Y.col);
    // Init Z, A, dA sau khi W da co dung kich thuoc
    vector <Mat> Z(W.size());
    vector <Mat> A(W.size());
    vector <Mat> dA(W.size());
    for(size_t i = 0; i < W.size(); i++){
        Z[i]  = Mat(X.row, W[i].col);
        A[i]  = Mat(X.row, W[i].col);
        dA[i] = Mat(X.row, W[i].col);
    }
    Mat ones = Mat::ones(Y.row, 1);
    float alpha = learning_rate / static_cast<float>(X.row);
    float parameter_scaling = 1.0f - lambda * alpha;
    if (loss_type == "MSE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Forward_Pass_ReLU(X, W, B, Z, A, "MSE");
            history.save(MSE(Y, A.back(), regularization, lambda, W.back()));
            Backward_Pass_ReLU(X, Y, W, Z, A, dA, dW, dB, loss_type);
            for (size_t i = 0; i < W.size(); i++) {
                UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                UpdateParameters(dB[i], B[i], B[i], alpha);
            }
        }
    } else if (loss_type == "MAE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Forward_Pass_ReLU(X, W, B, Z, A, "MAE");
            history.save(MAE(Y, A.back(), regularization, lambda, W.back()));
            Backward_Pass_ReLU(X, Y, W, Z, A, dA, dW, dB, loss_type);
            for (size_t i = 0; i < W.size(); i++) {
                UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                UpdateParameters(dB[i], B[i], B[i], alpha);
            }
        }
    }
    else if (loss_type == "BCE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Forward_Pass_ReLU(X, W, B, Z, A, "BCE");
            history.save(BCE(Y, Z.back(), regularization, lambda, W.back()));
            Backward_Pass_ReLU(X, Y, W, Z, A, dA, dW, dB, loss_type);
            for (size_t i = 0; i < W.size(); i++) {
                UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                UpdateParameters(dB[i], B[i], B[i], alpha);
            }
        }
    }
    else if (loss_type == "CCE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Forward_Pass_ReLU(X, W, B, Z, A, "CCE");
            history.save(CCE(Y, A.back(), regularization, lambda, W.back()));
            Backward_Pass_ReLU(X, Y, W, Z, A, dA, dW, dB, loss_type);
            for (size_t i = 0; i < W.size(); i++) {
                UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                UpdateParameters(dB[i], B[i], B[i], alpha);
            }
        }
    }
}
void Train_MLP(const Mat& X, const Mat& Y, vector<Mat>& W, vector<Mat>& B, vector <int> hidden_nodes, float learning_rate, int epochs, Loss_History& history, string loss_type, float pkeep, int batch_size){
    // Init dW, dB truoc de W duoc cap nhat dung kich thuoc
    int X_mini_size = X.row/batch_size;
    vector <Mat> X_mini (X_mini_size);
    vector <Mat> Y_mini (X_mini_size);
    for (int b = 0 ; b < X_mini_size ;b++){
        X_mini[b] = Mat(batch_size,X.col);
        Y_mini[b] = Mat(batch_size,Y.col);
        Copy_Vec(X.data.data() + b * batch_size * X.col, X_mini[b].data.data(),batch_size * X.col);
        Copy_Vec(Y.data.data() + b * batch_size * Y.col, Y_mini[b].data.data(),batch_size * Y.col);
    }
    vector <Mat> dW(W.size());
    vector <Mat> dB(B.size());
    Init_Node(dW, dB, hidden_nodes, X.col, Y.col);
    vector <Mat> Z(W.size());
    vector <Mat> A(W.size());
    vector <Mat> dA(W.size());
    vector <Mat> Mask(W.size() - 1);
    for(size_t i = 0; i < W.size(); i++){
        Z[i]  = Mat(batch_size, W[i].col);
        A[i]  = Mat(batch_size, W[i].col);
        dA[i] = Mat(batch_size, W[i].col);
        if (i > 0) Mask[i - 1] = Mat(batch_size, W[i - 1].col);
    }
    Mat ones = Mat::ones(Y.row, 1);
    float alpha = learning_rate / static_cast<float>(batch_size);
    float parameter_scaling = 1.0f;
    if (loss_type == "MSE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0 ; b < X_mini_size ;b++){
                for(size_t i = 0; i < W.size() - 1; i++)RandBer(Mask[i].data, pkeep);
                Forward_Pass_ReLU(X_mini[b], W, B, Z, A, Mask, loss_type);
                Backward_Pass_ReLU(X_mini[b], Y_mini[b], W, Z, A, dA, dW, dB, Mask, loss_type);
                for (size_t i = 0; i < W.size(); i++) {
                    UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                    UpdateParameters(dB[i], B[i], B[i], alpha);
                }
            }
            history.save(MSE(Y_mini[X_mini_size - 1], A.back()));
        }
    } else if (loss_type == "MAE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0 ; b < X_mini_size ;b++){
                for(size_t i = 0; i < W.size() - 1; i++)RandBer(Mask[i].data, pkeep);
                Forward_Pass_ReLU(X_mini[b], W, B, Z, A, Mask, loss_type);
                Backward_Pass_ReLU(X_mini[b], Y_mini[b], W, Z, A, dA, dW, dB, Mask, loss_type);
                for (size_t i = 0; i < W.size(); i++) {
                    UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                    UpdateParameters(dB[i], B[i], B[i], alpha);
                }
            }
            history.save(MAE(Y_mini[X_mini_size - 1], A.back()));
        }
    }
    else if (loss_type == "BCE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0 ; b < X_mini_size ;b++){
                for(size_t i = 0; i < W.size() - 1; i++)RandBer(Mask[i].data, pkeep);
                Forward_Pass_ReLU(X_mini[b], W, B, Z, A, Mask, loss_type);
                Backward_Pass_ReLU(X_mini[b], Y_mini[b], W, Z, A, dA, dW, dB, Mask, loss_type);
                for (size_t i = 0; i < W.size(); i++) {
                    UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                    UpdateParameters(dB[i], B[i], B[i], alpha);
                }
            }
            history.save(BCE(Y_mini[X_mini_size - 1], A.back()));
        }
    }
    else if (loss_type == "CCE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0 ; b < X_mini_size ;b++){
                for(size_t i = 0; i < W.size() - 1; i++)RandBer(Mask[i].data, pkeep);
                Forward_Pass_ReLU(X_mini[b], W, B, Z, A, Mask, loss_type);
                Backward_Pass_ReLU(X_mini[b], Y_mini[b], W, Z, A, dA, dW, dB, Mask, loss_type);
                for (size_t i = 0; i < W.size(); i++) {
                    UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                    UpdateParameters(dB[i], B[i], B[i], alpha);
                }
            }
            history.save(CCE(Y_mini[X_mini_size - 1], A.back()));
        }
    }
}
void Train_MLP(const Mat& X, const Mat& Y, const Mat& X_Val, const Mat& Y_Val, vector<Mat>& W, vector<Mat>& B, vector <int> hidden_nodes, float learning_rate,
     int epochs, Loss_History& history, string loss_type, float pkeep, int batch_size){
    // Init dW, dB truoc de W duoc cap nhat dung kich thuoc
    int X_mini_size = X.row/batch_size;
    vector <Mat> X_mini (X_mini_size);
    vector <Mat> Y_mini (X_mini_size);
    for (int b = 0 ; b < X_mini_size ;b++){
        X_mini[b] = Mat(batch_size,X.col);
        Y_mini[b] = Mat(batch_size,Y.col);
        Copy_Vec(X.data.data() + b * batch_size * X.col, X_mini[b].data.data(),batch_size * X.col);
        Copy_Vec(Y.data.data() + b * batch_size * Y.col, Y_mini[b].data.data(),batch_size * Y.col);
    }
    vector <Mat> dW(W.size());
    vector <Mat> dB(B.size());
    Init_Node(dW, dB, hidden_nodes, X.col, Y.col);
    vector <Mat> Z(W.size());
    vector <Mat> A(W.size());
    vector <Mat> dA(W.size());
    vector <Mat> Mask(W.size() - 1);
    for(size_t i = 0; i < W.size(); i++){
        Z[i]  = Mat(batch_size, W[i].col);
        A[i]  = Mat(batch_size, W[i].col);
        dA[i] = Mat(batch_size, W[i].col);
        if (i > 0) Mask[i - 1] = Mat(batch_size, W[i - 1].col);
    }
    Mat ones = Mat::ones(Y.row, 1);
    float alpha = learning_rate / static_cast<float>(batch_size);
    float parameter_scaling = 1.0f;int patience = 15;
    if (loss_type == "MSE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0 ; b < X_mini_size ;b++){
                for(size_t i = 0; i < W.size() - 1; i++)RandBer(Mask[i].data, pkeep);
                Forward_Pass_ReLU(X_mini[b], W, B, Z, A, Mask, loss_type);
                Backward_Pass_ReLU(X_mini[b], Y_mini[b], W, Z, A, dA, dW, dB, Mask, loss_type);
                for (size_t i = 0; i < W.size(); i++) {
                    UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                    UpdateParameters(dB[i], B[i], B[i], alpha);
                }
            }
            history.save(MSE(Y_mini[X_mini_size - 1], A.back()));
            if (EarlyStop(X_Val, Y_Val, W, B, loss_type, patience, epoch)) {
                for (int k = 0; k < patience;k++)history.Loss.pop_back();
                break; 
            }
        }
    } else if (loss_type == "MAE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0 ; b < X_mini_size ;b++){
                for(size_t i = 0; i < W.size() - 1; i++)RandBer(Mask[i].data, pkeep);
                Forward_Pass_ReLU(X_mini[b], W, B, Z, A, Mask, loss_type);
                Backward_Pass_ReLU(X_mini[b], Y_mini[b], W, Z, A, dA, dW, dB, Mask, loss_type);
                for (size_t i = 0; i < W.size(); i++) {
                    UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                    UpdateParameters(dB[i], B[i], B[i], alpha);
                }
            }
            history.save(MAE(Y_mini[X_mini_size - 1], A.back()));
            if (EarlyStop(X_Val, Y_Val, W, B, loss_type, patience, epoch)) {
                for (int k = 0; k < patience;k++)history.Loss.pop_back();
                break; 
            }
        }
    }
    else if (loss_type == "BCE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0 ; b < X_mini_size ;b++){
                for(size_t i = 0; i < W.size() - 1; i++)RandBer(Mask[i].data, pkeep);
                Forward_Pass_ReLU(X_mini[b], W, B, Z, A, Mask, loss_type);
                Backward_Pass_ReLU(X_mini[b], Y_mini[b], W, Z, A, dA, dW, dB, Mask, loss_type);
                for (size_t i = 0; i < W.size(); i++) {
                    UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                    UpdateParameters(dB[i], B[i], B[i], alpha);
                }
            }
            history.save(BCE(Y_mini[X_mini_size - 1], A.back()));
            if (EarlyStop(X_Val, Y_Val, W, B, loss_type, patience, epoch)) {
                for (int k = 0; k < patience;k++)history.Loss.pop_back();
                break; 
            }
        }
    }
    else if (loss_type == "CCE") {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0 ; b < X_mini_size ;b++){
                for(size_t i = 0; i < W.size() - 1; i++)RandBer(Mask[i].data, pkeep);
                Forward_Pass_ReLU(X_mini[b], W, B, Z, A, Mask, loss_type);
                Backward_Pass_ReLU(X_mini[b], Y_mini[b], W, Z, A, dA, dW, dB, Mask, loss_type);
                for (size_t i = 0; i < W.size(); i++) {
                    UpdateParameters(dW[i], W[i], W[i], alpha, parameter_scaling);
                    UpdateParameters(dB[i], B[i], B[i], alpha);
                }
            }
            history.save(CCE(Y_mini[X_mini_size - 1], A.back()));
            if (EarlyStop(X_Val, Y_Val, W, B, loss_type, patience, epoch)) {
                for (int k = 0; k < patience;k++)history.Loss.pop_back();
                break; 
            }
        }
    }
}
void LinearRegression(Mat& X, Mat& Y, Mat& W, Mat& B, float learning_rate, int epochs, Loss_History& history, string loss_type,
     string regularization, float lambda) {
    W.rand(); B.rand();
    Mat mean_mat(1, X.col), std_mat(1, X.col);
    FeatureScaling(X, X, mean_mat, std_mat, "standard");
    Train_LN(X, Y, W, B, learning_rate, epochs, history, loss_type, regularization, lambda);
	Rescale_Weight(W, B, mean_mat, std_mat);
}
void LogisticRegression_Binary(Mat& X, Mat& Y, Mat& W, Mat& B, float learning_rate, int epochs, Loss_History& history,
     string regularization, float lambda) {
    W.rand(); B.rand();
    Mat mean_mat(1, X.col), std_mat(1, X.col);
    FeatureScaling(X, X, mean_mat, std_mat, "standard");
    Train_LG_BIN(X, Y, W, B, learning_rate, epochs, history, regularization, lambda);
    Rescale_Weight(W, B, mean_mat, std_mat);
}
void MLP(Mat& X, Mat& Y, vector<Mat>& W, vector<Mat>& B,vector <int> hidden_nodes,float learning_rate, int epochs, Loss_History& history,
     string loss_type, string regularization, float lambda){
    Init_Node(W, B, hidden_nodes, X.col, Y.col);
    for (size_t i = 0; i < W.size(); i++) {
        RandNormal(W[i].data, 0.0f, sqrtf(2.0f / W[i].row));
        B[i].set_zeros();
    }
    Mat X_mean_mat(1, X.col), X_std_mat(1, X.col);
    FeatureScaling(X, X, X_mean_mat, X_std_mat, "standard");
    Train_MLP(X, Y, W, B, hidden_nodes, learning_rate, epochs, history, loss_type, regularization, lambda);
    Rescale_Weight(W[0], B[0], X_mean_mat, X_std_mat);
}
void MLP(Mat& X, Mat& Y, vector<Mat>& W, vector<Mat>& B,vector <int> hidden_nodes,float learning_rate, int epochs, Loss_History& history,
     string loss_type, float pkeep, int batch_size){
    Init_Node(W, B, hidden_nodes, X.col, Y.col);
    for (size_t i = 0; i < W.size(); i++) {
        RandNormal(W[i].data, 0.0f, sqrtf(2.0f / W[i].row));
        B[i].set_zeros();
    }
    Mat X_mean_mat(1, X.col), X_std_mat(1, X.col);
    FeatureScaling(X, X, X_mean_mat, X_std_mat, "standard");
    Train_MLP(X, Y, W, B, hidden_nodes, learning_rate, epochs, history, loss_type, pkeep, batch_size);
    Rescale_Weight(W[0], B[0], X_mean_mat, X_std_mat);
}
void MLP(Mat& X, Mat& Y, Mat& X_Val, Mat& Y_Val, vector<Mat>& W, vector<Mat>& B, vector <int> hidden_nodes, float learning_rate, 
     int epochs, Loss_History& history, string loss_type , float pkeep, int batch_size){
    Init_Node(W, B, hidden_nodes, X.col, Y.col);
    for (size_t i = 0; i < W.size(); i++) {
        RandNormal(W[i].data, 0.0f, sqrtf(2.0f / W[i].row));
        B[i].set_zeros();
    }
    Mat X_mean_mat(1, X.col), X_std_mat(1, X.col);
    FeatureScaling(X, X, X_mean_mat, X_std_mat, "standard");
    FeatureScaling(X_Val, X_mean_mat, X_std_mat);
    Train_MLP(X, Y, X_Val, Y_Val, W, B, hidden_nodes, learning_rate, epochs, history, loss_type, pkeep, batch_size);
    Rescale_Weight(W[0], B[0], X_mean_mat, X_std_mat);        
}
void K_Fold_MLP(const Mat& X, const Mat& Y, int K, bool shuffle, vector <int> hidden_nodes, float learning_rate, int epoch, string loss_type , float pkeep, int batch_size){
    int fold_size = X.row / K;
    Mat X_Val(fold_size, X.col), Y_Val(fold_size, Y.col);
    Mat X_Val_Raw = X_Val;
    Mat X_Train(X.row - fold_size, X.col), Y_Train(Y.row - fold_size, Y.col);
    Mat Y_Mean_mat(1, Y.col), Y_Std_mat(1, Y.col);
    vector<Mat> W(hidden_nodes.size() + 1), B(hidden_nodes.size() + 1);
    Loss_History history;
    float avr_point = 0.0f;
    for (int k =0; k < K; k++){
        int start_idx;
        if (shuffle) {
            start_idx = rand() % X.row;
            if (start_idx + fold_size > X.row) start_idx = X.row - fold_size;
        } else {
            start_idx = k * fold_size;
        }
        X_Val.copyfrom(X, "row", start_idx, start_idx + fold_size);
        Y_Val.copyfrom(Y, "row", start_idx, start_idx + fold_size);
        X_Train.copyexcept(X,"row", start_idx, start_idx + fold_size);
        Y_Train.copyexcept(Y,"row", start_idx, start_idx + fold_size);
        if (loss_type == "MAE" || loss_type == "MSE") {
            FeatureScaling(Y_Train, Y_Train, Y_Mean_mat, Y_Std_mat);
            FeatureScaling(Y_Val, Y_Mean_mat, Y_Std_mat);
        }
        X_Val_Raw = X_Val;
        MLP(X_Train, Y_Train, X_Val, Y_Val, W, B, hidden_nodes, learning_rate, epoch, history, loss_type, pkeep, batch_size);
        Mat Y_Pred(Y_Val.row, Y_Val.col);
        Forward_Pass_ReLU(X_Val_Raw, W, B, Y_Pred, loss_type);
        if (loss_type == "MAE" || loss_type == "MSE") {
            Rescale_Y(Y_Pred, Y_Mean_mat, Y_Std_mat);
            Rescale_Y(Y_Val, Y_Mean_mat, Y_Std_mat);
        }
        cout << "\nFold " << k + 1 << endl;
        avr_point += ModelEvaluation(Y_Val, Y_Pred, loss_type, "f1_score");
    }
    cout << "\nAverage result: " << avr_point / K << endl;
}