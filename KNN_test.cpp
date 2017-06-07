#include <falconn/lsh_nn_table.h>
#include <flann/flann.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdio>

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::max;
using std::mt19937_64;
using std::runtime_error;
using std::string;
using std::uniform_int_distribution;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::construct_table;
using falconn::compute_number_of_hash_functions;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::StorageHashTable;
using falconn::get_default_parameters;

using flann::Matrix;
using flann::L2;
using flann::Index;

typedef DenseVector<float> Point;

const string SIFT_BASE = "datasets/sift/sift_base.fvecs";
const string SIFT_GROUNDTRUTH = "datasets/sift/sift_groundtruth.fvecs";
const string SIFT_query = "datasets/sift/sift_query.fvecs";
const int NUM_DATA = 100000;
const int NUM_QUERIES = 1000;
const int SEED = 4057218;
const int NUM_HASH_TABLES = 60;
const int NUM_HASH_BITS = 18;
const int NUM_ROTATIONS = 1;
const int NUM_PROBS=60;
const int NUM_CHECK =1024*8;

bool read_point(FILE *file, Point *point, int &d) {
    if (fread(&d, sizeof(int), 1, file) != 1) {
        return false;
    }
    float *buf = new float[d];
    if (fread(buf, sizeof(float), d, file) != (size_t)d) {
        throw runtime_error("can't read a point");
    }
    point->resize(d);
    for (int i = 0; i < d; ++i) {
        (*point)[i] = buf[i];
    }
    delete[] buf;
    return true;
}


void read_dataset(string file_name, vector<Point> *dataset, int &d, int num) {
    FILE *file = fopen(file_name.c_str(), "rb");
    if (!file) {
        throw runtime_error("can't open the file with the dataset");
    }
    Point p;
    dataset->clear();
    while (read_point(file, &p, d) && dataset->size()<num) {
        dataset->push_back(p);
    }
    if (fclose(file)) {
        throw runtime_error("fclose() error");
    }
}

Matrix<float> vector_to_matrix(vector<Point> &dataset, int num, int d) {
    Matrix<float> M_dataset;
    vector<float> cache;
    for(auto it = dataset.begin(); it != dataset.end(); it++){
        Point point = *it;
        for(int it2 = 0; it2 < d; it2++){
            cache.push_back(point[it2]);
        }
    }
    M_dataset = Matrix<float>(&cache[0], num, d);
    return M_dataset;
}

Matrix<int> int_vector_to_matrix(vector<vector<int>> &results, int num, int nn) {
    Matrix<int> M_dataset;
    vector<int> cache;
    for(auto it = results.begin(); it != results.end(); it++){
        vector<int> point_index = *it;
        for(int it2 = 0; it2 < nn; it2++){
            cache.push_back(point_index[it2]);
        }
    }
    M_dataset = Matrix<int>(&cache[0], num, nn);
    return M_dataset;
}

float compute_precision(Matrix<int> answer, Matrix<int> indices, int num, int nn) {
    int count=0;
    float precision;
    for(int i=0; i<num; i++){
        for(int j=0; j<nn; j++){
            for(int k=0; k<nn; k++){
                if(indices[i][j] == answer[i][k]) {
                    count++;
                }
            }
        }
    }
    precision = (float)count/(num*nn);
    return precision;
}


int main() {
    try {
        vector<Point> dataset(NUM_DATA), queries(NUM_QUERIES);
        int d;
        int nn=10;

        // read the dataset
        cout << "reading points" << endl;
        read_dataset(SIFT_BASE, &dataset, d, NUM_DATA);
        cout << dataset.size() << " points read" << endl;
        cout << "data dimensions: " << d << endl;

        // read the queries
        cout << "reading queries" << endl;
        read_dataset(SIFT_BASE, &queries, d, NUM_QUERIES);
        cout << queries.size() << " points read" << endl;
        cout << "queries dimensions: " << d << endl;

        Matrix<float> M_dataset(new float[NUM_DATA], NUM_DATA, d);
        Matrix<float> M_query(new float[NUM_QUERIES], NUM_QUERIES, d);;

        M_dataset = vector_to_matrix(dataset, NUM_DATA, d);
        M_query = vector_to_matrix(queries, NUM_QUERIES, d);
        cout << "vector to matrix" << endl;

        //linear search
        Matrix<int> answer(new int[M_query.rows*nn], M_query.rows, nn);
        Matrix<float> exact_dists(new float[M_query.rows*nn], M_query.rows, nn);
        Index<L2<float>> linear_index(M_dataset, flann::LinearIndexParams());
        linear_index.buildIndex();

        cout << "running linear scan" << endl;
        auto t1 = high_resolution_clock::now();
        linear_index.knnSearch(M_query, answer, exact_dists, nn, flann::SearchParams(1024));
        auto t2 = high_resolution_clock::now();
        double linear_time = duration_cast<duration<double>>(t2 - t1).count();
        cout << "done" << endl;
        cout << "linear search time: " << linear_time  << endl;

        //kd-tree search
        vector<double> kd_search_times;
        vector<float> kd_precisions;

        Matrix<int> indices(new int[M_query.rows*nn], M_query.rows, nn);
        Matrix<float> dists(new float[M_query.rows*nn], M_query.rows, nn);

        // construct an randomized kd-tree index using 16 kd-trees
        Index<L2<float>> kd_tree_index(M_dataset, flann::KDTreeIndexParams(16));
        cout << "building kd-tree index" << endl;
        t1 = high_resolution_clock::now();
        kd_tree_index.buildIndex();
        t2 = high_resolution_clock::now();
        auto construct_time = duration_cast<duration<double>>(t2 - t1).count();
        cout << "done" << endl;
        cout << "construction time of kd-tree: " << construct_time << endl;

        for(int num_check=32; num_check<=NUM_CHECK; num_check*=2) {
            // do a knn search, using num_check checks
            cout << "using kd-tree to do a knn search" << endl;
            t1 = high_resolution_clock::now();
            kd_tree_index.knnSearch(M_query, indices, dists, nn, flann::SearchParams(num_check));
            t2 = high_resolution_clock::now();
            double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
            cout << "done" << endl;
            cout << "search time of randomized kd-tree with " << num_check << " checks: " << elapsed_time << endl;
            kd_search_times.push_back(elapsed_time);

            float kd_tree_presicion = compute_precision(answer, indices, NUM_QUERIES, nn);
            cout << "precision of randomized kd-tree with " << num_check << " checks: " << kd_tree_presicion << endl;
            kd_precisions.push_back(kd_tree_presicion);
        }

        //LSH search
        // setting parameters and constructing the table
        vector<double> LSH_construct_times;
        vector<double> LSH_search_times;
        vector<float> LSH_precisions;
        for(int num_hash=10; num_hash<=NUM_HASH_TABLES; num_hash+=10) {
            LSHConstructionParameters params;
            params.dimension = d;
            params.lsh_family = LSHFamily::Hyperplane;
            params.l = num_hash;
            params.distance_function = DistanceFunction::EuclideanSquared;
            compute_number_of_hash_functions<Point>(NUM_HASH_BITS, &params);
            params.num_rotations = NUM_ROTATIONS;
            // we want to use all the available threads to set up
            params.num_setup_threads = 0;
            params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;

            cout << "building the index based on the Hyperplane LSH" << endl;
            cout << "num_hash: " << num_hash << endl;
            t1 = high_resolution_clock::now();
            auto table = construct_table<Point>(dataset, params);
            t2 = high_resolution_clock::now();
            double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
            cout << "done" << endl;
            cout << "construction time: " << elapsed_time << endl;
            LSH_construct_times.push_back(elapsed_time);

            vector<vector<int>> results;
            vector<int> one_result;
            cout << "using LSH to do a knn search" << endl;
            t1 = high_resolution_clock::now();
            for (auto it = queries.begin(); it != queries.end(); it++) {
                table->find_k_nearest_neighbors(*it, nn, &one_result);
                results.push_back(one_result);
            }
            t2 = high_resolution_clock::now();
            elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
            cout << "search time of LSH: " << elapsed_time << endl;
            LSH_search_times.push_back(elapsed_time);

            Matrix<int> M_result = int_vector_to_matrix(results, NUM_QUERIES, nn);
            float LSH_precision = compute_precision(answer, M_result, NUM_QUERIES, nn);
            cout << "precision of LSH: " << LSH_precision << endl;
            LSH_precisions.push_back(LSH_precision);
        }

        cout << "statistic:" << endl;
        cout << "linear search time: " << linear_time  << endl;
        cout << "randomized kd-tree: " << endl;
        cout << "construction time: " << construct_time << endl;
        cout << "number check:  ";
        for(int num_check=32; num_check<=NUM_CHECK; num_check*=2) {
            cout << num_check << "  ";
        }
        cout << endl;
        cout << "randomized_kd_search_times:";
        for(auto it = kd_search_times.begin(); it != kd_search_times.end(); it++) {
            cout << *it << "  ";
        }
        cout << endl;
        cout << "randomized_kd_precisions:";
        for(auto it = kd_precisions.begin(); it != kd_precisions.end(); it++) {
            cout << *it << "  ";
        }
        cout << endl;
        cout << "LSH: " << endl;
        cout << "num_hash: ";
        for(int num_hash=10; num_hash<=NUM_HASH_TABLES; num_hash+=10) {
           cout << num_hash << "  ";
        }
        cout << endl;
        cout << "LSH_construct_times:";
        for(auto it = LSH_construct_times.begin(); it != LSH_construct_times.end(); it++) {
            cout << *it << "  ";
        }
        cout << endl;
        cout << "LSH_search_times:";
        for(auto it = LSH_search_times.begin(); it != LSH_search_times.end(); it++) {
            cout << *it << "  ";
        }
        cout << endl;
        cout << "LSH_precisions:";
        for(auto it = LSH_precisions.begin(); it != LSH_precisions.end(); it++) {
            cout << *it << "  ";
        }
        cout << endl;

    } catch (runtime_error &e) {
        cerr << "Runtime error: " << e.what() << endl;
        return 1;
    } catch (exception &e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "ERROR" << endl;
        return 1;
    }
    return 0;
}

