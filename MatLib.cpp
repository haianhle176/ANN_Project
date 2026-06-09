#include "MatLib.h"
#include <immintrin.h>
#include <utility>
#include <fstream>
#include <cstring>
#include <omp.h>
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
void Loss_History::print_final(){if (Loss.size()==0) {cout <<"Loss history empty!";return;} cout<<"\nLoss final: "<<Loss[Loss.size() - 1] << " end at "<<Loss.size() <<" epoch\n";}
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
vector<int> indices_shuffle(int size, std::mt19937& gen){
    vector<int> indices(size);
    for (int i = 0; i < size; i++) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), gen);
    return indices;
}
float evaluateModel(const Mat& Y_true, const Mat& Y_pred, string eval_type, string type, int num_classes){
    if (type == "clf"){
        float precision , recall , f1;
        float macro_precision = 0, macro_recall = 0, macro_f1 = 0;
        int num_cl = num_classes > 0 ? num_classes : Y_pred.col;
        Mat ConfusionMat(num_cl, num_cl);
        for (int i = 0; i < Y_true.row; i++){
            int idx_ytrue, idx_yhat;
            if (Y_true.col == 1) {
                idx_ytrue = (int)std::round(Y_true(i, 0));
                idx_yhat  = (int)std::round(Y_pred(i, 0));
            }
            else {
                idx_ytrue = -1; idx_yhat = -1;
                for (int j = 0; j < Y_true.col; j++)
                    if (Y_true(i, j)) { idx_ytrue = j; break; }
                if (Y_pred.col == 1)
                    idx_yhat = (int)std::round(Y_pred(i, 0));
                else
                    for (int j = 0; j < Y_pred.col; j++)
                        if (Y_pred(i, j)) { idx_yhat = j; break; }
            }
            ConfusionMat(idx_ytrue, idx_yhat)++;
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
            return (float)macro_precision / ConfusionMat.row;
        }
        else if (eval_type == "recall" ) {
            return (float)macro_recall / ConfusionMat.row;
        }
        else if (eval_type == "f1_score") {
            return (float)macro_f1 / ConfusionMat.row;
        }
        else return -1e3f;
    }
    else if (type == "reg"){
        if (eval_type == "MAE") {
            float mae = MAE(Y_true, Y_pred);
            return mae;
        }
        else if (eval_type == "RMSE") {
        float rmse = sqrtf(2* MSE(Y_true, Y_pred));
        return rmse;
        }
    }
    return 0.0f;
}
void evaluateModel(const Mat& Y_true, const Mat& Y_pred, string type, int num_classes){
    if (type == "clf"){
        float precision , recall , f1;
        float macro_precision = 0, macro_recall = 0, macro_f1 = 0;
        int num_cl = num_classes > 0 ? num_classes : Y_pred.col;
        Mat ConfusionMat(num_cl, num_cl);
        for (int i = 0; i < Y_true.row; i++){
            int idx_ytrue, idx_yhat;
            if (Y_true.col == 1) {
                idx_ytrue = (int)std::round(Y_true(i, 0));
                idx_yhat  = (int)std::round(Y_pred(i, 0));
            }
            else {
                idx_ytrue = -1; idx_yhat = -1;
                for (int j = 0; j < Y_true.col; j++)
                    if (Y_true(i, j)) { idx_ytrue = j; break; }
                if (Y_pred.col == 1)
                    idx_yhat = (int)std::round(Y_pred(i, 0));
                else
                    for (int j = 0; j < Y_pred.col; j++)
                        if (Y_pred(i, j)) { idx_yhat = j; break; }
            }
            ConfusionMat(idx_ytrue, idx_yhat)++;
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
            cout << "\nMacro Recall: " << (float)macro_recall / ConfusionMat.row << endl;
            cout << "\nMacro F1-Score: " << (float)macro_f1 / ConfusionMat.row << endl;
    }
    else if (type == "reg"){
        float mse = MSE(Y_true, Y_pred);
        float mae = MAE(Y_true, Y_pred);
        cout << "\nRoot Mean Squared Error: " << sqrtf(2* mse) << endl;
        cout << "Mean Absolute Error: " << mae << endl;
    }
}
void K_Fold_Evaluate(Model* ML,const Mat& X, const Mat& Y, int K, bool shuffle, string type, string eval_type){
    int N = X.row;
    int fold_size = N / K;

    vector<int> indices(N);
    for (int i = 0; i < N; i++) indices[i] = i;

    if (shuffle) {
        static mt19937 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    Mat X_Val(fold_size, X.col), Y_Val(fold_size, Y.col);
    Mat X_Train((N - fold_size), X.col), Y_Train((N - fold_size), Y.col);
    float avr_score = 0.0f;
    float fold_score ;
    for (int k = 0; k < K; k++) {
        int val_start = k * fold_size;
        int val_end   = val_start + fold_size;

        for (int i = 0; i < fold_size; i++) {
            int src_row = indices[val_start + i];
            Copy_Vec(X.data.data() + src_row * X.col,
                     X_Val.data.data() + i * X.col, X.col);
            Copy_Vec(Y.data.data() + src_row * Y.col,
                     Y_Val.data.data() + i * Y.col, Y.col);
        }

        int dst_row = 0;
        for (int i = 0; i < N; i++) {
            if (i >= val_start && i < val_end) continue;
            int src_row = indices[i];
            Copy_Vec(X.data.data() + src_row * X.col,
                     X_Train.data.data() + dst_row * X.col, X.col);
            Copy_Vec(Y.data.data() + src_row * Y.col,
                     Y_Train.data.data() + dst_row * Y.col, Y.col);
            dst_row++;
        }

        ML->fit(X_Train, Y_Train);
        Mat Y_Pred = ML->predict(X_Val);
        cout << "\nFold " << k + 1 << endl;
        fold_score = ML->evaluate(Y_Val, Y_Pred, eval_type);
        cout << "Fold " << k + 1 << " Score: " << fold_score << " ("<< eval_type << ")" <<endl;
        avr_score += fold_score;
    }
    cout << "\nAverage score: " << avr_score / K << endl;
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
int Majority_Index(const float* dst, vector <int>& count, const vector <int>& indices){
    for (int i = 0;i < count.size();i++) count[i] = 0;
    if (indices.empty()){
        for (int i = 0;i< count.size();i++){
            count[dst[i]]++;
        }
    }
    else{
        for (int i : indices){
            count[dst[i]]++;
        }
    }
    int majority_class = 0;
    for (size_t i = 1; i < count.size(); i++) {
        if (count[i] > count[majority_class]) majority_class = i;
    }
    return majority_class;
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
void ShowPredict(const Mat& Y_Pred, const Mat& Y_Test, string type){
    if (type == "clf"){
        cout << "\nSample  Predicted  Exact  Probabilities:\n";
        int cnt = 0;
        for (int i = 0; i < Y_Pred.row; i++) {
            int pred_class = (Y_Pred.col == 1)
                ? (int)Y_Pred(i, 0)
                : [&]{ int c=0; for(int j=1;j<Y_Pred.col;j++) if(Y_Pred(i,j)>Y_Pred(i,c)) c=j; return c; }();

            int true_class = (Y_Test.col == 1)
                ? (int)Y_Test(i, 0)
                : [&]{ int c=0; for(int j=1;j<Y_Test.col;j++) if(Y_Test(i,j)>Y_Test(i,c)) c=j; return c; }();

            if (pred_class == true_class) cnt++;
            cout << "  [" << i + 1 << "]  pred=" << pred_class << "  true=" << true_class << "  prob=[";
            for (int j = 0; j < Y_Pred.col; j++) {
                cout << fixed << setprecision(3) << Y_Pred(i, j);
                if (j < Y_Pred.col - 1) cout << ", ";
            }
            cout << "]" << (pred_class == true_class ? "  OK" : "  X") << "\n";
        }
        cout << "\nCorrect Predictions: " << cnt << "/" << Y_Test.row << " (" << 100.0f * cnt / Y_Test.row << "%)\n";
    }
    else if (type == "reg") {
        cout << "\nSample  Predicted  Exact:\n";
        for (int i = 0; i < Y_Pred.row; i++) {
            for (int j = 0; j < Y_Pred.col; j++) {
                cout << "  [" << i + 1 << "]  pred=" << fixed << setprecision(3) << Y_Pred(i, j) << "  true=" << Y_Test(i, j) << "\n";
            }
        }
    }
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
void shuffleMatrixColumn(Mat& X, int colIndex, std::mt19937& gen) {
    int numRows = X.row;
    if (numRows <= 1) return;

    // Thuật toán Fisher-Yates áp dụng riêng cho chỉ số dòng của một cột
    for (int i = numRows - 1; i > 0; --i) {
        // Sinh một số nguyên ngẫu nhiên j trong khoảng [0, i]
        std::uniform_int_distribution<int> distrib(0, i);
        int j = distrib(gen);

        // Hoán đổi giá trị tại cột colIndex giữa dòng i và dòng j
        std::swap(X(i,colIndex), X(j, colIndex));
    }
}
void FeatureScaling(const Mat& src, Mat& dst,  Mat& mean_mat, Mat& std_mat, bool fit) {
    if (fit) {
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
    }
    dst -= mean_mat;
    Hadamard_Broadcast_Row(dst, std_mat, dst);
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

void MLP::k_fold(const Mat& X, const Mat& Y, int K, string eval_type, bool shuffle) {
    string type = (loss_type == "MSE" || loss_type == "MAE") ? "reg" : "clf";
    K_Fold_Evaluate(this, X, Y, K, shuffle, type,  eval_type);
}

MLP::MLP(const vector<int>& hidden_nodes, const string& loss_type, float learning_rate, int epochs,
     float pkeep, int batch_size, const string& regularization, float lambda)
    : hidden_nodes(hidden_nodes), loss_type(loss_type), learning_rate(learning_rate), epochs(epochs),
      regularization(regularization),
      lambda(lambda), pkeep(pkeep), batch_size(batch_size), X_mean(1, 0), X_std(1, 0),
      Y_mean(1, 0), Y_std(1, 0) {
    int layer_count = static_cast<int>(hidden_nodes.size()) + 1;
    W.resize(layer_count);
    B.resize(layer_count);
}

void MLP::initialize_weights(int input_dim, int output_dim) {
    int layer_count = static_cast<int>(hidden_nodes.size()) + 1;
    W.assign(layer_count, Mat());
    B.assign(layer_count, Mat());
    Init_Node(W, B, hidden_nodes, input_dim, output_dim);
    for (size_t i = 0; i < W.size(); i++) {
        RandNormal(W[i].data, 0.0f, sqrtf(2.0f / W[i].row));
        B[i].set_zeros();
    }
}

void MLP::scale_training_data(const Mat& X, const Mat& Y, Mat& X_scaled, Mat& Y_scaled) {
    X_scaled = X;
    X_mean = Mat(1, X.col);
    X_std = Mat(1, X.col);
    FeatureScaling(X, X_scaled, X_mean, X_std);

    Y_scaled = Y;
    if (loss_type == "MSE" || loss_type == "MAE") {
        Y_mean = Mat(1, Y.col);
        Y_std = Mat(1, Y.col);
        FeatureScaling(Y, Y_scaled, Y_mean, Y_std);
    } else {
        Y_mean = Mat(0, 0);
        Y_std = Mat(0, 0);
    }
}

void MLP::scale_validation_data(const Mat& X_val, const Mat& Y_val, Mat& X_val_scaled,
                                 Mat& Y_val_scaled) {
    X_val_scaled = X_val;
    if (X_mean.row == 1 && X_mean.col == X_val.col) {
        FeatureScaling(X_val, X_val_scaled, X_mean, X_std, false);
    }
    Y_val_scaled = Y_val;
    if ((loss_type == "MSE" || loss_type == "MAE") && Y_mean.row == 1 && Y_mean.col == Y_val.col) {
        FeatureScaling(Y_val, Y_val_scaled, Y_mean, Y_std, false);
    }
}

void MLP::fit(const Mat& X, const Mat& Y) {
    initialize_weights(X.col, Y.col);
    Mat X_scaled, Y_scaled;
    scale_training_data(X, Y, X_scaled, Y_scaled);

    int effective_batch = batch_size > 0 ? batch_size : X.row;
    if (batch_size > 0 || pkeep < 1.0f) {
        Train_MLP(X_scaled, Y_scaled, W, B, hidden_nodes, learning_rate, epochs, history,
                  loss_type, pkeep, effective_batch);
    } else {
        Train_MLP(X_scaled, Y_scaled, W, B, hidden_nodes, learning_rate, epochs, history,
                  loss_type, regularization, lambda);
    }
    Rescale_Weight(W[0] ,B[0], X_mean, X_std);
}

void MLP::fit_with_valid(const Mat& X, const Mat& Y, const Mat& X_val, const Mat& Y_val) {
    initialize_weights(X.col, Y.col);
    Mat X_scaled, Y_scaled;
    scale_training_data(X, Y, X_scaled, Y_scaled);

    Mat X_val_scaled, Y_val_scaled;
    scale_validation_data(X_val, Y_val, X_val_scaled, Y_val_scaled);

    int effective_batch = batch_size > 0 ? batch_size : X.row;
    Train_MLP(X_scaled, Y_scaled, X_val_scaled, Y_val_scaled, W, B, hidden_nodes,
              learning_rate, epochs, history, loss_type, pkeep, effective_batch);
}

void MLP::predict(const Mat& X, Mat& Y_pred) const {
    Mat X_scaled = X;
    MLP_Test(X_scaled, W, B, Y_pred, loss_type, const_cast<Mat&>(Y_mean), const_cast<Mat&>(Y_std));
}

Mat MLP::predict(const Mat& X) const {
    int output_dim = W.empty() ? 0 : W.back().col;
    Mat Y_pred(X.row, output_dim);
    predict(X, Y_pred);
    return Y_pred;
}

void MLP::evaluate(const Mat& Y_true, const Mat& Y_pred) const {
    if (loss_type == "BCE" || loss_type == "CCE"){
        Mat Y_hat (Y_pred.row,Y_pred.col);
        if (loss_type == "BCE") apply(Y_pred, Y_hat, [this](float x) {return (x > threshold) ? 1.0f : 0.0f;});
        else {
            for (int i = 0; i < Y_hat.row; i++) {
                int pred_class = 0;
                for (int j = 1; j < Y_hat.col; j++) if (Y_pred(i, j) > Y_pred(i, pred_class)) pred_class = j;
                Y_hat(i,pred_class) = 1.0f; 
            }
        }
        evaluateModel(Y_true, Y_hat, "clf");
    }
    else if (loss_type == "MSE" || loss_type == "MAE"){
        evaluateModel(Y_true, Y_pred, "reg");
    }
}
float MLP::evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const {
    if (loss_type == "BCE" || loss_type == "CCE"){
        Mat Y_hat (Y_pred.row,Y_pred.col);
        if (loss_type == "BCE") apply(Y_pred, Y_hat, [this](float x) {return (x > threshold) ? 1.0f : 0.0f;});
        else {
            for (int i = 0; i < Y_hat.row; i++) {
                int pred_class = 0;
                for (int j = 1; j < Y_hat.col; j++) if (Y_pred(i, j) > Y_pred(i, pred_class)) pred_class = j;
                Y_hat(i,pred_class) = 1.0f; 
            }
        }
        return evaluateModel(Y_true, Y_hat, eval_type, "clf");
    }
    else if (loss_type == "MSE" || loss_type == "MAE"){
        return evaluateModel(Y_true, Y_pred, eval_type, "reg");
    }
}
float DecisionTree::gini_cal(const vector<int>& count, int total_samples){
    if (total_samples == 0) return 0.0f;
    float gini = 1.0f;
    for (int cnt : count) {
        float prob = (float)cnt / total_samples;
        gini -= (prob * prob);
    }
    return gini;
}

float DecisionTree:: varience_cal(const Mat&Y, vector <int>& sample_indices){
    float y_mean= 0;
    for (int i : sample_indices){
        y_mean += Y(i, 0);
    }
    y_mean /= sample_indices.size();
    float temp = 0;
    for (int i : sample_indices){
        temp += (Y(i, 0) - y_mean) * (Y(i, 0) - y_mean);
    }
    return (temp < epsilon) ? 0 : temp;
}
float DecisionTree:: gini_cal(const Mat&Y, vector <int>& sample_indices){
    vector<int> count(num_class,0);
    for (int i : sample_indices) count[(int)Y(i,0)]++;
    float gini = 1.0f;
    float total_samples = sample_indices.size();
    for (int x : count) {
        float prob = x / total_samples;
        gini -= (prob * prob);
    }
    return gini;
}
void DecisionTree:: find_best_split_clf(const Mat& X, const Mat&Y, vector<int>& idx, vector <int>& count_left, vector <int>& count_right,
             int f, int& best_feature, float& best_threshold, float& best_gini, int& best_split_pos){
    int N = idx.size();
    std::sort(idx.begin(), idx.end(), [&X, f](int a, int b) {
        return X(a,f) < X(b,f);
    });
    for (int i : idx) {
        count_right[Y(i, 0)]++; 
    }
    int n_left = 0;
    int n_right = N;
    for (int i = 0; i < N - 1; i++) {
        int label = Y(idx[i], 0);

        // Lệnh cộng trừ lũy kế O(1) thần thánh: Di chuyển 1 phần tử từ Phải sang Trái
        count_right[label]--;
        count_left[label]++;
        
        n_left++;
        n_right--;

        // Bỏ qua nếu giá trị của phần tử tiếp theo trùng lặp (tránh cắt ở giữa 2 giá trị bằng nhau)
        if (X(idx[i],f) == X(idx[i + 1], f)) continue;

        // Tính nhanh Gini của nhát cắt hiện tại mà không cần split data
        float gini_left = gini_cal(count_left, n_left);
        float gini_right = gini_cal(count_right, n_right);
        
        float gini_split = ((float)n_left / N) * gini_left + ((float)n_right / N) * gini_right;

        // Cập nhật nếu tìm thấy nhát cắt ít vẩn đục hơn
        if (gini_split < best_gini) {
            best_gini = gini_split;
            best_feature = f;
            // Chọn ngưỡng nằm chính giữa phần tử hiện tại và phần tử tiếp theo
            best_threshold = (X(idx[i],f) + X(idx[i + 1],f)) / 2.0f; 
            best_split_pos = i;
        }
    }
}
void DecisionTree::find_best_split_reg(const Mat& X, const Mat& Y, vector<int>& idx, int f,
     int& best_feature, float& best_threshold, float& best_variance, int& best_split_pos) {
    int N = idx.size();
    std::sort(idx.begin(), idx.end(), [&X, f](int a, int b) { return X(a, f) < X(b, f); });

    // 2. Khởi tạo trạng thái ban đầu (Toàn bộ dữ liệu nằm bên PHẢI)
    int n_left = 0;
    int n_right = N;
    
    float sum_left = 0.0f;
    float sum_sq_left = 0.0f;
    
    float sum_right = 0.0f;
    float sum_sq_right = 0.0f;

    // Chạy vòng lặp 1 lần duy nhất để lấy tổng ban đầu cho nhánh Phải
    for (int i = 0; i < N; i++) {
        float y_val = Y(idx[i], 0);
        sum_right += y_val;
        sum_sq_right += (y_val * y_val);
    }

    // 3. Quét tịnh tiến O(N)
    for (int i = 0; i < N - 1; i++) {
        int current_idx = idx[i];
        float y_val = Y(current_idx, 0);

        // DỊCH CHUYỂN 1 PHẦN TỬ TỪ PHẢI SANG TRÁI (Chi phí O(1))
        n_left++;
        n_right--;
        
        sum_left += y_val;
        sum_right -= y_val;
        
        sum_sq_left += (y_val * y_val);
        sum_sq_right -= (y_val * y_val);

        // Bỏ qua nếu giá trị feature trùng nhau
        if (X(idx[i], f) == X(idx[i + 1], f)) continue;

        // Tính Variance 2 nhánh siêu tốc bằng công thức biến đổi
        float mean_left = sum_left / n_left;
        float var_left = (sum_sq_left / n_left) - (mean_left * mean_left);
        // Ngăn chặn sai số dấu phẩy động làm variance bị âm tí hon
        if (var_left < 0.0f) var_left = 0.0f; 

        float mean_right = sum_right / n_right;
        float var_right = (sum_sq_right / n_right) - (mean_right * mean_right);
        if (var_right < 0.0f) var_right = 0.0f;

        // Tính Tổng Phương sai có trọng số
        float variance_split = ((float)n_left / N) * var_left + ((float)n_right / N) * var_right;

        // Cập nhật kỷ lục
        if (variance_split < best_variance) {
            best_variance = variance_split;
            best_feature = f;
            best_threshold = (X(idx[i], f) + X(idx[i + 1], f)) / 2.0f;
            best_split_pos = i;
        }
    }
}
float DecisionTree:: get_majority(const Mat& Y, vector <int>& sample_indices){
if (type == "clf"){
        static vector<int> count(num_class);
        return Majority_Index(Y.data.data(), count, sample_indices);
    }
    else {
        float sum = 0.0f;
        for(int idx : sample_indices) sum += Y(idx, 0);
        return sum / sample_indices.size();
    }
}
float DecisionTree::predict_single(const float* X, TreeNode* node) const{
    if (node->is_leaf) {
        return node->leaf_value;
    }
    if (X[node->feature_idx] <= node->threshold) {
        return predict_single(X, node->left);
    } else {
        return predict_single(X, node->right);
    }
}

void DecisionTree::fit(const Mat& X, const Mat& Y){
    vector<int> first_sample_indices(Y.row);
    for (int i = 0; i < Y.row; i++) first_sample_indices[i] = i;
    vector<int> first_feature_indices(X.col);
    for (int i = 0; i < X.col; i++) first_feature_indices[i] = i;
    if (Y.col > 1 && type == "clf") {
        num_class = Y.col;
        Mat Y_idx(Y.row, 1);
        for (int i = 0;i < Y.row; i++){
            for (int j = 0; j < Y.col; j++) {
                if (Y(i,j) == 1.0f) {
                    Y_idx(i,0) = (float)j;
                    break;
                }
            }
        }
    root = build_tree(X, Y_idx, 0, first_sample_indices, first_feature_indices);
    }
    else{
        num_class = (int)*std::max_element(Y.data.begin(), Y.data.end()) + 1;
        root = build_tree(X, Y, 0, first_sample_indices, first_feature_indices);
    }
}
void DecisionTree::fit(const Mat& X, const Mat& Y, vector<int>& sample_indices, vector<int>& feature_indices) {
        if (Y.col > 1 && type == "clf") {
        num_class = Y.col;
        Mat Y_idx(Y.row, 1);
        for (int i = 0;i < Y.row; i++){
            for (int j = 0; j < Y.col; j++) {
                if (Y(i,j) == 1.0f) {
                    Y_idx(i,0) = (float)j;
                    break;
                }
            }
        }
    root = build_tree(X, Y_idx, 0, sample_indices, feature_indices);
    }
    else{
        num_class = (int)*std::max_element(Y.data.begin(), Y.data.end()) + 1;
        root = build_tree(X, Y, 0, sample_indices, feature_indices);
    }
}
Mat DecisionTree::predict(const Mat& X) const{
    Mat Y_pred(X.row, 1);
    for (int i = 0; i < X.row; i++) {
        Y_pred(i, 0) = predict_single(X.data.data() + i * X.col, root);
    }
    return Y_pred;
}
void DecisionTree:: predict(const Mat& X, Mat& Y_pred) const{
    for (int i = 0; i < X.row; i++) {
        Y_pred(i, 0) = predict_single(X.data.data() + i * X.col, root);
    }
}

float DecisionTree:: predict(const float* X) const{
    return predict_single(X, root);
}
float DecisionTree::evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const {
    return evaluateModel(Y_true, Y_pred, type, eval_type, num_class);
}
void DecisionTree::evaluate(const Mat& Y_true, const Mat& Y_pred) const {
    evaluateModel(Y_true, Y_pred, type, num_class);
}
void DecisionTree::k_fold(const Mat& X, const Mat& Y, int K, string eval_type, bool shuffle){
    K_Fold_Evaluate(this, X, Y, K, shuffle, type, eval_type);
}
TreeNode* DecisionTree:: build_tree(const Mat&X, const Mat& Y, int depth, vector<int>& sample_indices, vector<int>& feature_indices){
    int num_samples = sample_indices.size();
    TreeNode* node = new TreeNode();
    float current_stop_val;
    if (type == "clf") current_stop_val = gini_cal(Y, sample_indices);
    else if (type == "reg") current_stop_val = varience_cal(Y, sample_indices);
    if (depth >= max_depth || num_samples < min_samples_split || current_stop_val == 0.0f) {
        node->is_leaf = true;
        node->leaf_value = get_majority(Y, sample_indices);
        return node;
    }
    float best_gini = numeric_limits<float>::max();
    int best_feature = -1;
    float best_threshold = 0.0f;
    int best_split_pos = 0;
    if (type == "clf") {
        vector <int> count_left(num_class);
        vector <int> count_right(num_class);
        for (int f : feature_indices){
            for (int i = 0; i < num_class;i++){
                count_left[i] = 0;
                count_right[i] = 0;
            }
            find_best_split_clf(X, Y, sample_indices, count_left, count_right, f, best_feature, best_threshold,
            best_gini, best_split_pos);        
        }
    }
    else if (type == "reg"){
        for (int f : feature_indices){
            find_best_split_reg(X, Y, sample_indices, f, best_feature, best_threshold,
            best_gini, best_split_pos);        
        }
    }
    if ((best_feature == -1)||fabs(current_stop_val - best_gini) < min_impurity_decrease) {
        node->is_leaf = true;
        node->leaf_value = get_majority(Y, sample_indices);
        return node;
    }
    node->feature_idx = best_feature;
    node->threshold = best_threshold;
    auto it = std::stable_partition(sample_indices.begin(), sample_indices.end(), [&](int idx) {
        return X(idx, best_feature) <= best_threshold;
    });
    std::vector<int> left_indices(sample_indices.begin(), it);
    std::vector<int> right_indices(it, sample_indices.end());
    node->left = build_tree(X, Y, depth + 1, left_indices, feature_indices);
    node->right = build_tree(X, Y, depth + 1, right_indices, feature_indices);
    return node;
}
vector <int> RandomForest::_bootstrap(int num_samples){
    std::uniform_int_distribution<> dis(0, num_samples - 1);
    std::vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; i++) {
        indices[i] = dis(gen); // Lấy mẫu có hoàn lại
    }
    return indices;
}
float RandomForest::predict_single(const float* X) const{
    if (type == "reg") {
        // BÀI TOÁN HỒI QUY: Lấy Trung bình cộng câu trả lời của các cây
        float sum = 0.0f;
        for (int i = 0; i < num_trees; i++) {
            sum += forest[i].predict(X);
        }
        return sum / num_trees;
    } else {
        vector<int> votes(num_classes, 0);
        for (int i = 0; i < num_trees; i++) {
            int predicted_label = (int)forest[i].predict(X);
            votes[predicted_label]++;
        }
        return std::distance(votes.begin(), std::max_element(votes.begin(), votes.end()));
    }
}
Mat RandomForest:: predict(const Mat& X) const{
    Mat Y_Pred(X.row, 1);
    // Bạn cũng có thể song song hóa bước Inference này nếu tập Test có hàng triệu dòng
    #pragma omp parallel for
    for (int i = 0; i < X.row; i++) {
        Y_Pred(i,0) = predict_single(X.data.data() + i * X.col);
    }
    return Y_Pred;
}
void RandomForest:: predict(const Mat& X, Mat& Y_pred) const{
    // Bạn cũng có thể song song hóa bước Inference này nếu tập Test có hàng triệu dòng
    #pragma omp parallel for
    for (int i = 0; i < X.row; i++) {
        Y_pred(i,0) = predict_single(X.data.data() + i * X.col);
    }
}
void RandomForest::fit(const Mat& X, const Mat& Y){
    if (type == "reg") {num_features_split = std::max(1, X.col / 3);num_classes = 1;}
    else if (type == "clf") {num_features_split = std::max(1, (int)std::sqrt(X.col));
    if (Y.col > 1) num_classes = Y.col;
    else num_classes = (int)*std::max_element(Y.data.begin(), Y.data.end()) + 1;}
    forest.resize(num_trees, DecisionTree(type, max_depth, min_samples_split, num_classes));
    if (Y.col > 1) {
        Mat Y_idx(Y.row, 1);
        for (int i = 0;i < Y.row; i++){
            for (int j = 0; j < Y.col; j++) {
                if (Y(i,j) == 1.0f) {
                    Y_idx(i,0) = (float)j;
                    break;
                }
            }
        }
        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 thread_gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ omp_get_thread_num());
            #pragma omp for
            for (int i = 0; i < num_trees; i++) {
                // Bước 1: Tạo mảng chỉ số dòng ngẫu nhiên cho cây thứ i
                std::vector<int> sample_indices = _bootstrap(X.row);
                std::vector<int> feature_indices = indices_shuffle(X.col, gen);
                forest[i].fit(X, Y_idx, sample_indices, feature_indices);
            }
        }
    }
    else{
        #pragma omp parallel
        {
            gen = mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ omp_get_thread_num());
            #pragma omp for
            for (int i = 0; i < num_trees; i++) {
                // Bước 1: Tạo mảng chỉ số dòng ngẫu nhiên cho cây thứ i
                std::vector<int> sample_indices = _bootstrap(X.row);
                std::vector<int> feature_indices = indices_shuffle(X.col, gen);
                forest[i].fit(X, Y, sample_indices, feature_indices);
            }
        }
    }
}
void RandomForest::evaluate(const Mat& Y_true, const Mat& Y_pred) const{
    evaluateModel(Y_true, Y_pred, type, num_classes);
}
float RandomForest::evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const {
    return evaluateModel(Y_true, Y_pred, eval_type, type, num_classes);
}
void RandomForest::k_fold(const Mat& X, const Mat& Y, int K, string eval_type, bool shuffle){
    if (type == "reg") {num_features_split = std::max(1, X.col / 3);num_classes = 1;}
    else if (type == "clf") {num_features_split = std::max(1, (int)std::sqrt(X.col));
    if (Y.col > 1) num_classes = Y.col;
    else num_classes = (int)*std::max_element(Y.data.begin(), Y.data.end()) + 1;}
    forest.resize(num_trees, DecisionTree(type, max_depth, min_samples_split, num_classes));
    K_Fold_Evaluate(this, X, Y, K, shuffle, type, eval_type);
}
void RandomForest::feature_importance(Mat& X, const Mat& Y_true){
    Mat X_temp(X.row, X.col);
    Mat Y_pred(Y_true.row,Y_true.col);
    Y_pred = this->predict(X);
    string eval_type = (type == "clf") ? "f1_score" : "MAE";
    float base_line = evaluate(Y_true, Y_pred, eval_type);
    int best_idx = 0, worst_idx = 0;
    double best_val = 0, worst_val = 1e6;
    for (int i = 0; i < X.col; i ++){
        double sum = 0;
        X_temp = X;
        for (int j = 0; j < 10; j ++){
            shuffleMatrixColumn(X_temp, i, gen);
            Y_pred = this->predict(X_temp);
            sum += evaluate(Y_true, Y_pred, eval_type);
        }
        sum/=10.0f;
        sum = fabs(sum - base_line);
        if (best_val < sum) {
            best_idx = i;
            best_val = sum;
        }
        if (worst_val > sum){
            worst_idx = i;
            worst_val = sum;
        }
        cout << "feature " << i + 1 << " score gap : " << sum <<endl;
    }
    cout << "Most important feature is feature " << best_idx + 1 << " with score gap = " << fixed << setprecision(5) << best_val << endl;
    cout << "Least important feature is feature " << worst_idx + 1 << " with score gap = " << fixed << setprecision(5) << worst_val << endl; 
}
LinearRegression::LinearRegression(const string& loss_type,float learning_rate, int epochs,int batch_size, const string& regularization,
                                   float lambda)
    : loss_type(loss_type), learning_rate(learning_rate), epochs(epochs), regularization(regularization), lambda(lambda), batch_size(batch_size),
      X_mean(1, 0), X_std(1, 0) {}

void LinearRegression::fit(const Mat& X, const Mat& Y) {
    W = Mat(X.col, Y.col);
    B = Mat(1, Y.col);
    W.rand();
    B.rand();
    X_mean = Mat(1, X.col);
    X_std = Mat(1, X.col);
    Mat X_scaled = X;
    FeatureScaling(X, X_scaled, X_mean, X_std);

    if (batch_size > 0 && batch_size < X.row) {
        int n_batches = (X.row + batch_size - 1) / batch_size;
        vector<Mat> X_mini(n_batches);
        vector<Mat> Y_mini(n_batches);
        for (int b = 0; b < n_batches; ++b) {
            int start = b * batch_size;
            int end = min(X.row, start + batch_size);
            int current_rows = end - start;
            X_mini[b] = Mat(current_rows, X.col);
            Y_mini[b] = Mat(current_rows, Y.col);
            for (int i = 0; i < current_rows; ++i) {
                Copy_Vec(X_scaled.data.data() + (start + i) * X.col, X_mini[b].data.data() + i * X.col, X.col);
                Copy_Vec(Y.data.data() + (start + i) * Y.col, Y_mini[b].data.data() + i * Y.col, Y.col);
            }
        }
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int b = 0; b < n_batches; ++b) {
                Train_LN(X_mini[b], Y_mini[b], W, B, learning_rate, 1, history, loss_type, regularization, lambda);
            }
        }
    } else {
        Train_LN(X_scaled, Y, W, B, learning_rate, epochs, history, loss_type, regularization, lambda);
    }
}

void LinearRegression::fit_with_valid(const Mat& X, const Mat& Y, const Mat& X_val, const Mat& Y_val) {
    W = Mat(X.col, Y.col);
    B = Mat(1, Y.col);
    W.rand();
    B.rand();

    Mat X_scaled = X;
    FeatureScaling(X, X_scaled, X_mean, X_std);
    Mat X_val_scaled = X_val;
    if (X_mean.row == 1 && X_mean.col == X_val.col) {
        FeatureScaling(X_val, X_val_scaled, X_mean, X_std);
    }

    int patience = 15;
    int wait = 0;
    float best_val = numeric_limits<float>::infinity();
    Mat best_W = W;
    Mat best_B = B;

    int n_batches = 0;
    vector<Mat> X_mini;
    vector<Mat> Y_mini;
    if (batch_size > 0 && batch_size < X.row) {
        n_batches = (X.row + batch_size - 1) / batch_size;
        X_mini.resize(n_batches);
        Y_mini.resize(n_batches);
        for (int b = 0; b < n_batches; ++b) {
            int start = b * batch_size;
            int end = min(X.row, start + batch_size);
            int current_rows = end - start;
            X_mini[b] = Mat(current_rows, X.col);
            Y_mini[b] = Mat(current_rows, Y.col);
            for (int i = 0; i < current_rows; ++i) {
                Copy_Vec(X_scaled.data.data() + (start + i) * X.col, X_mini[b].data.data() + i * X.col, X.col);
                Copy_Vec(Y.data.data() + (start + i) * Y.col, Y_mini[b].data.data() + i * Y.col, Y.col);
            }
        }
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (n_batches > 0) {
            for (int b = 0; b < n_batches; ++b) {
                Train_LN(X_mini[b], Y_mini[b], W, B, learning_rate, 1, history, loss_type, regularization, lambda);
            }
        } else {
            Train_LN(X_scaled, Y, W, B, learning_rate, 1, history, loss_type, regularization, lambda);
        }

        Mat Y_pred_val;
        predict(X_val, Y_pred_val);
        float val_loss = Loss_Cal(Y_val, Y_pred_val, loss_type);
        if (val_loss < best_val) {
            best_val = val_loss;
            wait = 0;
            best_W = W;
            best_B = B;
        } else {
            wait++;
            if (wait >= patience) {
                W = best_W;
                B = best_B;
                break;
            }
        }
    }
}

void LinearRegression::predict(const Mat& X, Mat& Y_pred) const {
    Mat X_scaled = X;
    if (X_mean.row == 1 && X_mean.col == X.col) {
        FeatureScaling(X, X_scaled, const_cast<Mat&>(X_mean), const_cast<Mat&>(X_std));
    }
    mxm(X_scaled, W, Y_pred);
    Y_pred += B;
}

Mat LinearRegression::predict(const Mat& X) const {
    Mat Y_pred(X.row, W.col);
    predict(X, Y_pred);
    return Y_pred;
}

void LinearRegression::evaluate(const Mat& Y_true, const Mat& Y_pred) const {
    evaluateModel(Y_true, Y_pred, "reg");
}

float LinearRegression::evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const {
    return evaluateModel(Y_true, Y_pred, eval_type, "reg");
}

LogisticRegression::LogisticRegression(const string& loss_type, float learning_rate, int epochs, const string& regularization,
                                       float lambda, int batch_size)
    : loss_type(loss_type), learning_rate(learning_rate), epochs(epochs), regularization(regularization), lambda(lambda), batch_size(batch_size),
      X_mean(1, 0), X_std(1, 0){}

void LogisticRegression::fit(const Mat& X, const Mat& Y) {
    W = Mat(X.col, Y.col);
    B = Mat(1, Y.col);
    W.rand();
    B.rand();
    X_mean = Mat(1, X.col);
    X_std = Mat(1, X.col);
    Mat X_scaled = X;
    FeatureScaling(X, X_scaled, X_mean, X_std);

    Mat Y_scaled = Y;

    if (batch_size > 0 && batch_size < X.row) {
        int n_batches = (X.row + batch_size - 1) / batch_size;
        vector<Mat> X_mini(n_batches);
        vector<Mat> Y_mini(n_batches);
        for (int b = 0; b < n_batches; ++b) {
            int start = b * batch_size;
            int end = min(X.row, start + batch_size);
            int current_rows = end - start;
            X_mini[b] = Mat(current_rows, X.col);
            Y_mini[b] = Mat(current_rows, Y.col);
            for (int i = 0; i < current_rows; ++i) {
                Copy_Vec(X_scaled.data.data() + (start + i) * X.col, X_mini[b].data.data() + i * X.col, X.col);
                Copy_Vec(Y_scaled.data.data() + (start + i) * Y.col, Y_mini[b].data.data() + i * Y.col, Y.col);
            }
        }
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int b = 0; b < n_batches; ++b) {
                Train_LG_BIN(X_mini[b], Y_mini[b], W, B, learning_rate, 1, history, regularization, lambda);
            }
        }
    } else {
        Train_LG_BIN(X_scaled, Y_scaled, W, B, learning_rate, epochs, history, regularization, lambda);
    }
}

void LogisticRegression::fit_with_valid(const Mat& X, const Mat& Y, const Mat& X_val, const Mat& Y_val) {
    W = Mat(X.col, Y.col);
    B = Mat(1, Y.col);
    W.rand();
    B.rand();

    Mat X_scaled = X;
    FeatureScaling(X, X_scaled, X_mean, X_std);

    Mat Y_scaled = Y;

    Mat X_val_scaled = X_val;
    if (X_mean.row == 1 && X_mean.col == X_val.col) {
        FeatureScaling(X_val, X_val_scaled, X_mean, X_std);
    }

    Mat Y_val_scaled = Y_val;

    int patience = 15;
    int wait = 0;
    float best_val = numeric_limits<float>::infinity();
    Mat best_W = W;
    Mat best_B = B;

    int n_batches = 0;
    vector<Mat> X_mini;
    vector<Mat> Y_mini;
    if (batch_size > 0 && batch_size < X.row) {
        n_batches = (X.row + batch_size - 1) / batch_size;
        X_mini.resize(n_batches);
        Y_mini.resize(n_batches);
        for (int b = 0; b < n_batches; ++b) {
            int start = b * batch_size;
            int end = min(X.row, start + batch_size);
            int current_rows = end - start;
            X_mini[b] = Mat(current_rows, X.col);
            Y_mini[b] = Mat(current_rows, Y.col);
            for (int i = 0; i < current_rows; ++i) {
                Copy_Vec(X_scaled.data.data() + (start + i) * X.col, X_mini[b].data.data() + i * X.col, X.col);
                Copy_Vec(Y_scaled.data.data() + (start + i) * Y.col, Y_mini[b].data.data() + i * Y.col, Y.col);
            }
        }
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (n_batches > 0) {
            for (int b = 0; b < n_batches; ++b) {
                Train_LG_BIN(X_mini[b], Y_mini[b], W, B, learning_rate, 1, history, regularization, lambda);
            }
        } else {
            Train_LG_BIN(X_scaled, Y_scaled, W, B, learning_rate, 1, history, regularization, lambda);
        }

        Mat Y_pred_val;
        predict(X_val, Y_pred_val);
        float val_loss = Loss_Cal(Y_val, Y_pred_val, loss_type);
        if (val_loss < best_val) {
            best_val = val_loss;
            wait = 0;
            best_W = W;
            best_B = B;
        } else {
            wait++;
            if (wait >= patience) {
                W = best_W;
                B = best_B;
                break;
            }
        }
    }
}

void LogisticRegression::predict(const Mat& X, Mat& Y_pred) const {
    Mat X_scaled = X;
    if (X_mean.row == 1 && X_mean.col == X.col) {
        FeatureScaling(X, X_scaled, const_cast<Mat&>(X_mean), const_cast<Mat&>(X_std));
    }
    mxm(X_scaled, W, Y_pred);
    Y_pred += B;
    Sigmoid(Y_pred, Y_pred);
}

Mat LogisticRegression::predict(const Mat& X) const {
    Mat Y_pred(X.row, W.col);
    predict(X, Y_pred);
    return Y_pred;
}

void LogisticRegression::evaluate(const Mat& Y_true, const Mat& Y_pred) const {
    if (loss_type == "BCE") {
        Mat Y_hat (Y_pred.row, Y_pred.col);
        apply(Y_pred, Y_hat, [this](float x) {return (x > threshold) ? 1.0f : 0.0f;});
        evaluateModel(Y_true, Y_hat, "clf");
    }
}

float LogisticRegression::evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const {
    if (loss_type == "BCE") {
        Mat Y_hat (Y_pred.row, Y_pred.col);
        apply(Y_pred, Y_hat, [this](float x) {return (x > threshold) ? 1.0f : 0.0f;});
        return evaluateModel(Y_true, Y_hat, "clf", eval_type);
    }
    return 0.0f;
}

void KNN:: fit(const Mat& X, const Mat& Y) {
    X_train = Mat(X.row, X.col);
    X_mean = Mat(1, X.col);
    X_std = Mat(1, X.col);
    FeatureScaling(X, X_train, X_mean, X_std);
    if (type == "reg"){
        num_classes = 1;
        Y_train = Y; 
        return;
    }
    if (Y.col > 1 && type == "clf") {
        Mat Y_idx(Y.row, 1);
        num_classes = Y.col;
        for (int i = 0;i < Y.row; i++){
            for (int j = 0; j < Y.col; j++) {
                if (Y(i,j) == 1.0f) {
                    Y_idx(i,0) = (float)j;
                    break;
                }
            }
        }
        Y_train = Y_idx;
    }
    else {
        num_classes = (int)*std::max_element(Y.data.begin(), Y.data.end()) + 1;
        Y_train = Y;
    }
}
Mat KNN:: predict(const Mat& X) const {
    Mat Y_pred (X.row, 1);
    Mat X_scaled = X;
    FeatureScaling(X, X_scaled, const_cast<Mat&> (X_mean), const_cast<Mat&> (X_std), false);
    vector <pair<float, float>> distance(X_train.row);
    if (type == "clf"){
        for (int i = 0;i < X.row;i++){
            vector <int> count (num_classes, 0);
            for (int j = 0; j < X_train.row; j++) {
                distance[j].first = dist(X_scaled.data.data() + i * X_scaled.col, X_train.data.data() + j * X_train.col, X_scaled.col);
                distance[j].second = Y_train(j,0);
            }
            std::nth_element(distance.begin(), distance.begin() + K, distance.end(),
                        [](const pair<float, float>& a, const pair<float, float>& b) {
                            return a.first < b.first; 
                        });
            for (int k = 0; k < K;k++){
                count[distance[k].second]++;
            }
            auto max_it = std::max_element(count.begin(), count.end());
            int best_label = std::distance(count.begin(), max_it);
            Y_pred(i, 0) = (float)best_label;
        }
    }
    else if (type == "reg"){
        for (int i = 0;i < X.row;i++){
            double sum = 0;
            for (int j = 0; j < X_train.row; j++) {
                distance[j].first = dist(X_scaled.data.data() + i * X_scaled.col, X_train.data.data() + j * X_train.col, X_scaled.col);
                distance[j].second = Y_train(j,0);
            }
            std::nth_element(distance.begin(), distance.begin() + K, distance.end(),
                        [](const pair<float, float>& a, const pair<float, float>& b) {
                            return a.first < b.first; 
                        });
            for (int k = 0; k < K; k++) sum += distance[k].second;
            sum/=K;
            Y_pred(i, 0) = sum;
        }
    }
    return Y_pred;
}
void KNN::evaluate(const Mat& Y_true, const Mat& Y_pred) const {
    evaluateModel(Y_true, Y_pred, type, num_classes);
}
float KNN::evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const {
    return evaluateModel(Y_true, Y_pred, eval_type, type, num_classes);
}