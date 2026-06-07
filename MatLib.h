#ifndef MatLib_H 
#define MatLib_H
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>
#include <random>
using namespace std;
using namespace std::chrono;

struct Mat {
    int row,col;
    vector <float> data;
    Mat(int row = 0, int col = 0);
    float& operator()(int r, int c);
    const float& operator()(int r, int c) const;
    void set_identity();
    void set_zeros();
    void set_ones(float value = 1.0f);
    static Mat identity(int size);
    static Mat zeros(int row, int col);
    static Mat ones(int row, int col, float value = 1.0f);
    void transpose(Mat& dst) const;
    void append_row(const vector<float>& new_row);
    void append_col(const vector<float>& new_col);
    void append_row(float value);
    void append_col(float value);
    void row_col_swap();
    const int size() const;
    void rand(float min_val = -1.0f, float max_val = 1.0f);
    void print() const;
    void savetxt(const string& filename) const;
    Mat& copyfrom(const Mat& src, string direction, int start, int end);
    Mat& copyexcept(const Mat& src, string direction, int start_except, int end_except);
    Mat& loadmat(const string& filename, int start = 0, int end = 0);
    Mat& operator=(const Mat& other);
    Mat& operator+=(const Mat& other);
    Mat& operator-=(const Mat& other);
    Mat& operator*=(float scalar);
    Mat& transpose();
    friend Mat operator*(const Mat& src1, const Mat& src2);
};
struct Loss_History {
	vector <float> Loss;
    void save(float loss);
    void print();
    void print_final();
};
struct TreeNode{
    int feature_idx;       // Chỉ số của đặc trưng (cột) được chọn để chia
    float threshold;       // Ngưỡng chia (điểm cắt)
    TreeNode* left;        // Nhánh trái (<= threshold)
    TreeNode* right;       // Nhánh phải (> threshold)
    float leaf_value;        // Giá trị dự đoán nếu là nút lá
    bool is_leaf;
    TreeNode() : feature_idx(-1), threshold(0.0f), left(nullptr), right(nullptr), leaf_value(-1.0f), is_leaf(false) {}
};

class Model{
    public:
    virtual ~Model() = default;
    virtual void fit(const Mat& X,const Mat& y) = 0;
    virtual Mat predict(const Mat& X) const = 0;
    virtual float evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const;;        
};
class MLP : public Model {
private: 
    vector<Mat> W;
    vector<Mat> B;
    vector<int> hidden_nodes;
    string loss_type;
    string regularization;
    float learning_rate;
    int epochs;
    float lambda;
    float pkeep;
    int batch_size;
    Mat X_mean, X_std;
    Mat Y_mean, Y_std;
public:
    float threshold = 0.5f;
    Loss_History history;
    MLP(const vector<int>& hidden_nodes = {}, const string& loss_type = "MSE", float learning_rate = 0.001f, int epochs = 1000, 
        float pkeep = 1.0f, int batch_size = 0, const string& regularization = "", float lambda = 0.0f);

    void fit (const Mat& X, const Mat& Y) override;
    void fit_with_valid(const Mat& X, const Mat& Y, const Mat& X_val, const Mat& Y_val);
    void predict(const Mat& X, Mat& Y_pred) const;
    Mat predict(const Mat& X) const override;
    void k_fold(const Mat& X, const Mat& Y, int K, string eval_type, bool shuffle);
    void evaluate(const Mat& Y_true, const Mat& Y_pred) const;
    float evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const override;

private:
    void initialize_weights(int input_dim, int output_dim);
    void scale_training_data(const Mat& X, const Mat& Y, Mat& X_scaled, Mat& Y_scaled);
    void scale_validation_data(const Mat& X_val, const Mat& Y_val, Mat& X_val_scaled,
                               Mat& Y_val_scaled);
};

class DecisionTree : public Model {
    private:
        int max_depth;
        int min_samples_split;
        string type;
        TreeNode* root;
        float epsilon, min_impurity_decrease;
        int num_class;
        vector<int> sample_indices;
        vector<int> feature_indices;
        float gini_cal(const Mat&Y);
        float varience_cal(const Mat&Y);
        float gini_cal(const Mat&Y, vector <int>& sample_indices);
        float varience_cal(const Mat&Y, vector <int>& sample_indices);
        float gini_cal(const vector<int>& count, int total_samples);
        void split_dataset(const Mat& X, const Mat& Y, int split_pos, int feature_idx,
             float threshold, Mat& X_left, Mat& Y_left, Mat& X_right, Mat& Y_right);
        void find_best_split_clf(const Mat& X, const Mat&Y, vector<int>& idx, vector <int>& count_left, vector <int>& count_right,
             int f, int& best_feature, float& best_threshold, float& best_gini, int& best_split_pos);
        void find_best_split_reg(const Mat& X, const Mat& Y, vector<int>& idx, int f, int& best_feature, float& best_threshold,
             float& best_variance, int& best_split_pos);
        float get_majority(const Mat& Y);
        float get_majority(const Mat& Y, vector <int>& sample_indices);
        float predict_single(const float* X, TreeNode* node) const;
        TreeNode* build_tree(const Mat&X, const Mat& Y, int depth);
        TreeNode* build_forest(const Mat&X, const Mat& Y, int depth, vector<int>& sample_indices, vector<int>& feature_indices);
    public:
        DecisionTree(string type, int max_depth = 5, int min_samples_split = 3, int num_class = 0)
        : type(type), max_depth(max_depth), min_samples_split(min_samples_split), root(nullptr),
         epsilon(1e-3), min_impurity_decrease(1e-4), num_class(num_class){}
        void fit(const Mat& X, const Mat& Y) override ;
        Mat predict(const Mat& X) const override;
        void predict(const Mat& X, Mat& Y_pred) const;
        float predict(const float* X) const;
        void evaluate(const Mat& Y_true, const Mat& Y_pred) const;
        float evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const override;
        void k_fold(const Mat& X, const Mat& Y, int K, string eval_type, bool shuffle);
        void fit_forest(const Mat& X, const Mat& Y, vector<int>& sample_indices, vector<int>& feature_indices);
};

class RandomForest : public Model{
    private:
        int num_trees;
        int max_depth;
        int min_samples_split;
        int num_features_split;
        string type;
        int num_classes;
        vector<DecisionTree> forest;
        std::mt19937 gen;
        vector<int> _bootstrap(int num_samples);
    public:
        RandomForest(string type, int num_trees = 100, int max_depth = 10, int min_samples_split = 2)
        :type(type), num_trees(num_trees), max_depth(max_depth), min_samples_split(min_samples_split){}
        void fit(const Mat& X, const Mat& Y) override;
        Mat predict(const Mat& X) const override;
        void predict(const Mat& X, Mat& Y_pred) const;
        float predict_single(const float* X) const;
        void evaluate(const Mat& Y_true, const Mat& Y_pred) const;
        float evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const override;
        void k_fold(const Mat& X, const Mat& Y, int K, string eval_type, bool shuffle);
        void feature_importance(Mat& X, const Mat& Y_true);
};
class LinearRegression : public Model {
private:
    Mat W;
    Mat B;
    string loss_type;
    string regularization;
    float lambda;
    int batch_size;
    Mat X_mean, X_std;
    float learning_rate;
    int epochs;
public:
    Loss_History history;
    LinearRegression(const string& loss_type = "MSE", float learning_rate = 0.01f, int epochs = 1000, const string& regularization = "",
                     float lambda = 0.0f, int batch_size = 0);

    void fit(const Mat& X, const Mat& Y) override;
    void fit_with_valid(const Mat& X, const Mat& Y, const Mat& X_val, const Mat& Y_val);
    void predict(const Mat& X, Mat& Y_pred) const;
    Mat predict(const Mat& X) const override;
    void evaluate(const Mat& Y_true, const Mat& Y_pred) const;
    float evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const override;
};

class LogisticRegression : public Model{
private:
    Mat W;
    Mat B;
    string loss_type;
    string regularization;
    float lambda;
    int batch_size;
    Mat X_mean, X_std;
    float learning_rate;
    int epochs;
public:
    Loss_History history;
    float threshold = 0.5f;
    LogisticRegression(const string& loss_type = "BCE", float learning_rate = 0.1f, int epochs = 1000, const string& regularization = "",
                       float lambda = 0.0f, int batch_size = 0);

    void fit(const Mat& X, const Mat& Y) override;
    void fit_with_valid(const Mat& X, const Mat& Y, const Mat& X_val, const Mat& Y_val);
    void predict(const Mat& X, Mat& Y_pred) const ;
    Mat predict(const Mat& X) const override;
    void evaluate(const Mat& Y_true, const Mat& Y_pred) const;
    float evaluate(const Mat& Y_true, const Mat& Y_pred, string eval_type) const override;
};


void mxm(const Mat& src1, const Mat& src2, Mat& dst);
void mxm_block_tilt(const Mat& src1, const Mat& src2, Mat& dst);
void mtxm(const Mat& src1, const Mat& src2, Mat& dst);
void mxmt_naive(const Mat& src1, const Mat& src2, Mat& dst);
void mxmt(const Mat& src1, const Mat& src2, Mat& dst);
void vector_add(const float* src1, const float* src2, float* dst, int n);
void vector_sub(const float* src1, const float* src2, float* dst, int n);
float sum_elements_abs(const float *src, int n);
float sum_elements_abs(const float *src1,const float *src2, int n);
float sum_elements(const float* src, int n);
float dot(const float* src1, const float* src2, int n);
float dist(const float* src1, const float* src2, int n);
float dist(const float* src, const float scalar, int n);
float norm(const float* src, int n);
float mean(const float* src, int n);
float var(const float* src, const float* mean_vec, int n);
template <typename Func>
void apply(const Mat& src, Mat& dst, Func f){
    float* __restrict pdst = dst.data.data();
    const float* __restrict psrc = src.data.data();
    int total_size = dst.size();
    for (int i = 0; i < total_size; ++i) {
        pdst[i] = f(psrc[i]);
    }
}
template <typename Func>
void apply(const Mat& src1, const Mat& src2, Mat& dst, Func f){
    if (src1.row != src2.row || src1.col != src2.col || src1.row != dst.row || src1.col != dst.col) {
        cerr << "Error: All matrices must have the same dimensions for element-wise operation." << endl;
        return;
    }
    const float* __restrict psrc1 = src1.data.data();
    const float* __restrict psrc2 = src2.data.data();
    float* __restrict pdst = dst.data.data();
    int total_size = src1.size();
    for (int i = 0; i < total_size; ++i) {
        pdst[i] = f(psrc1[i], psrc2[i]);
    }
}
template <typename Func>
void apply_broadcast_row(const Mat& lhs, const Mat& rhs, Mat& dst, Func f) {
    int lhs_row = lhs.row;int lhs_col = lhs.col;
    for (int i = 0; i < lhs_row; ++i) {
        const float* __restrict plhs = lhs.data.data() + i * lhs_col;
        const float* __restrict prhs = rhs.data.data();
        float* pDst = dst.data.data() + i * dst.col;
        for (int j = 0; j < lhs_col; ++j) {
            pDst[j] = f(plhs[j], prhs[j]);
        }   
    }
}
void Sign_Matrix(const Mat& src, Mat& dst);
void UpdateParameters(const Mat& dsrc, const Mat& src, Mat& dst, float learning_rate, float parameter_scaling = 1.0f);
void ReLU(const Mat& src, Mat& dst);
void dReLU(const Mat& cmp, const Mat& src, Mat& dst);
void Sigmoid(const Mat& src, Mat& dst);
void Softmax(const Mat& src, Mat& dst);
void Hadamard(const Mat& src1, const Mat& src2, Mat& dst);
void Hadamard(const float* src1, const float* src2, float* dst,int n);
void Element_Wise_Div(const Mat& src1, const Mat& src2, Mat& dst);
void Hadamard_Broadcast_Row(const Mat& src1, const Mat& src2, Mat& dst);
void Sum_Rows(const Mat& src, Mat& dst);
void Sum_Cols(const Mat& src, Mat& dst);
inline void Copy_Vec(const float* src, float* dst,int n) {std::copy(src, src + n, dst);}
vector<int> indices_shuffle(int size, std::mt19937& gen);
float evaluateModel(const Mat& Y_true, const Mat& Y_pred, string eval_type, string type);
void evaluateModel(const Mat& Y_true, const Mat& Y_pred, string type);
void K_Fold_Evaluate(Model* ML,const Mat& X, const Mat& Y, int K, bool shuffle, string type, string eval_type);

float MSE(const Mat& Y, const Mat& Z);
float MAE(const Mat& Y, const Mat& Z);
float BCE(const Mat& Y, const Mat& P);
float CCE(const Mat& Y, const Mat& P);
float MSE(const Mat& Y, const Mat& Z, string regularization, float lambda, const Mat& W);
float MAE(const Mat& Y, const Mat& Z, string regularization, float lambda, const Mat& W);
float BCE(const Mat& Y, const Mat& P, string regularization, float lambda, const Mat& W);
float Loss_Cal(const Mat& Y, const Mat& Y_hat, string loss_type);

float CCE(const Mat& Y, const Mat& P, string regularization, float lambda, const Mat& W);
void FeatureScaling(const Mat& src, Mat& dst, Mat& mean_mat, Mat& std_mat, bool fit = true);
void Rescale_Weight(Mat& W, Mat& B, Mat& mean_mat, Mat& inv_std_mat);
void Rescale_Y(Mat& Y_Pred, Mat& Y_mean, Mat& Y_inv_std);

inline void Forward_Pass_ReLU(const Mat& X, const vector<Mat>& W, const vector<Mat>& B, vector<Mat>& Z, vector<Mat>& A, string type){
    for (size_t l = 0; l < W.size(); l++) {
        if (l == 0) {
            mxm(X, W[0], Z[0]);
        } else {
            mxm(A[l - 1], W[l], Z[l]);
        }
        Z[l] += B[l];
        
        // Layer cuối cùng không dùng ReLU
        if (l == W.size() - 1) {
            if (type == "BCE") Sigmoid(Z[l], A[l]);
            else if (type == "CCE") Softmax(Z[l],  A[l]);
            else A[l] = Z[l]; // Giữ nguyên tuyến tính cho MSE/MAE
        } else {
            ReLU(Z[l], A[l]);
        }
    }
}
inline void Forward_Pass_ReLU(const Mat& X, const vector<Mat>& W, const vector<Mat>& B, vector<Mat>& Z, vector<Mat>& A, vector<Mat>& Mask, string type){
    for (size_t l = 0; l < W.size(); l++) {
        if (l == 0) {
            mxm(X, W[0], Z[0]);
        } else {
            mxm(A[l - 1], W[l], Z[l]);
        }
        Z[l] += B[l];
        
        // Layer cuối cùng không dùng ReLU
        if (l == W.size() - 1) {
            if (type == "BCE") Sigmoid(Z[l], A[l]);
            else if (type == "CCE") Softmax(Z[l],  A[l]);
            else A[l] = Z[l]; // Giữ nguyên tuyến tính cho MSE/MAE
        } else {
            ReLU(Z[l], A[l]);
            Hadamard(A[l], Mask[l], A[l]);
        }
    }
}
inline void Forward_Pass_ReLU(const Mat& X, const vector<Mat>& W, const vector<Mat>& B, Mat& Y_Pred, string type){
    vector<Mat> A(W.size());
    vector<Mat> Z(W.size());

    for (size_t l = 0; l < W.size(); l++) {
        Z[l] = Mat(X.row, W[l].col);
        A[l] = Mat(X.row, W[l].col);
        if (l == 0) mxm(X, W[0], Z[0]);
        else mxm(A[l - 1], W[l], Z[l]);
        
        Z[l] += B[l];
        
        if (l == W.size() - 1) {
            if (type == "BCE") Sigmoid(Z[l], Y_Pred);
            else if (type == "CCE") Softmax(Z[l], Y_Pred);
            else Y_Pred = Z[l];
        } else {
            ReLU(Z[l], A[l]);
        }
    }
}
inline void MLP_Test(const Mat& X, const vector<Mat>& W, const vector<Mat>& B, Mat& Y_Pred, string loss_type, Mat &Y_mean, Mat& Y_inv_std){
    vector<Mat> A(W.size());
    vector<Mat> Z(W.size());
    for (size_t l = 0; l < W.size(); l++) {
        Z[l] = Mat(X.row, W[l].col);
        A[l] = Mat(X.row, W[l].col);
        if (l == 0) mxm(X, W[0], Z[0]);
        else mxm(A[l - 1], W[l], Z[l]);
        
        Z[l] += B[l];
        
        if (l == W.size() - 1) {
            if (loss_type == "BCE") Sigmoid(Z[l], Y_Pred);
            else if (loss_type == "CCE") Softmax(Z[l], Y_Pred);
            else Y_Pred = Z[l];
        } else {
            ReLU(Z[l], A[l]);
        }
    }
    if (loss_type == "MSE" || loss_type == "MAE") Rescale_Y(Y_Pred, Y_mean, Y_inv_std);
}
inline void Backward_Pass_ReLU(const Mat& X, const Mat& Y, vector<Mat>& W, vector<Mat>& Z, vector<Mat>& A,vector <Mat>& dA, vector<Mat>& dW, vector<Mat>& dB, string loss_type){
    if (loss_type=="MAE"){
        apply(A.back(), Y, dA.back(), [](float a, float y) { return (float)(a - y > 0) - (float)(a - y < 0);});
    }
    else apply(A.back(), Y, dA.back(), [](float a, float y) { return a - y;});
    if (W.size() >= 2) mtxm(A[W.size()-2], dA.back(), dW[W.size()-1]);
    else mtxm(X, dA.back(), dW[W.size()-1]);
    Sum_Rows(dA.back(), dB.back());
    for (int l = W.size() - 2; l >= 0; l--) {
        mxmt(dA[l + 1], W[l + 1], dA[l]);
        dReLU(Z[l], dA[l], dA[l]);
        if (l > 0) {
            mtxm(A[l - 1], dA[l], dW[l]);
        } else {
            mtxm(X, dA[l], dW[l]);
        }
        Sum_Rows(dA[l], dB[l]);
    }
}
inline void Backward_Pass_ReLU(const Mat& X, const Mat& Y,const vector<Mat>& W,const vector<Mat>& Z,const vector<Mat>& A,vector <Mat>& dA,
     vector<Mat>& dW, vector<Mat>& dB, vector<Mat>& Mask, string loss_type){
    if (loss_type=="MAE"){
        apply(A.back(), Y, dA.back(), [](float a, float y) { return (float)(a - y > 0) - (float)(a - y < 0);});
    }
    else apply(A.back(), Y, dA.back(), [](float a, float y) { return a - y;});
    if (W.size() >= 2) mtxm(A[W.size()-2], dA.back(), dW[W.size()-1]);
    else mtxm(X, dA.back(), dW[W.size()-1]);
    Sum_Rows(dA.back(), dB.back());
    for (int l = W.size() - 2; l >= 0; l--) {
        mxmt(dA[l + 1], W[l + 1], dA[l]);
        Hadamard(dA[l], Mask[l], dA[l]);
        dReLU(Z[l], dA[l], dA[l]);
        if (l > 0) {
            mtxm(A[l - 1], dA[l], dW[l]);
        } else {
            mtxm(X, dA[l], dW[l]);
        }
        Sum_Rows(dA[l], dB[l]);
    }
}
void RandNormal(vector<float>& dst, float mu, float sigma);
float RandUni(float a,float b);
void RandBer(vector<float>& dst, float p_keep);
static time_point<high_resolution_clock> start_timer, stop_timer;
void StartTimer();
void StopTimer();
void PrintTimer();
void ShowPredict(const Mat& Y_Pred, const Mat& Y_Test, string type);
int EarlyStop(const Mat& X_Val, const Mat& Y_Val, vector <Mat>& W,vector <Mat>& B, string loss_type , int patience, int epoch);
void shuffleMatrixColumn(Mat& X, int colIndex, std::mt19937& gen);
float Precision(float TP, float FP);
float Recall(float TP, float FN);
float F1_Score(float precision, float recall);
int Majority_Index(const vector<float>& dst,vector <int>& count, const vector <int>& indices = {});


void Train_LN(const Mat& X, const Mat& Y, Mat& W, Mat& B, float learning_rate, int epochs, Loss_History& history, string loss_type,
     string regularization = "none", float lambda = 0.0f);
void Train_LN(const Mat& X, const Mat& Y, Mat& W, Mat& B, float learning_rate, int epochs, string loss_type);
void Train_LG_BIN(const Mat& X, const Mat& Y, Mat& W, Mat& B, float learning_rate, int epochs, Loss_History* history,
     string regularization = "none", float lambda = 0.0f);
void Train_MLP(const Mat& X, const Mat& Y, vector<Mat>& W, vector<Mat>& B, vector <int> hidden_nodes, float learning_rate, int epochs, Loss_History& history,
     string loss_type , string regularization = "none", float lambda = 0.0f);
void Train_MLP(const Mat& X, const Mat& Y, vector<Mat>& W, vector<Mat>& B, vector <int> hidden_nodes, float learning_rate, int epochs, Loss_History& history,
     string loss_type, float pkeep, int batch_size);
void Train_MLP(const Mat& X, const Mat& Y, const Mat& X_Val, const Mat& Y_Val, vector<Mat>& W, vector<Mat>& B, vector <int> hidden_nodes, float learning_rate,
     int epochs, Loss_History& history, string loss_type, float pkeep, int batch_size);





#endif