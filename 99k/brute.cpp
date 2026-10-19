#include <bits/stdc++.h>
#include <chrono>
using namespace std;

vector<float> load_fbin(const std::string& path, size_t expected_count) {
    ifstream in(path, ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    vector<float> data(expected_count);
    in.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(float));
    if (!in) throw runtime_error("read failed or file too short: " + path);
    return data;
}

vector<int> load_ibin(const string& path, size_t expected_count) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("cannot open " + path);

    vector<int> data(expected_count);
    in.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(int));
    if (!in) throw runtime_error("read failed or file too short: " + path);
    return data;
}

float squared_distance(const float* a, const float* b, size_t dim) {
    float dist = 0;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

vector<int> top_k_indices(const vector<float>& dataset, int n, int dim, const float* query, size_t k) {
    priority_queue<pair<float, int>> maxheap;

    for (int i = 0; i < n; ++i) {
        float dist = squared_distance(&dataset[(size_t)i * dim], query, dim);

        if (maxheap.size() < k) {
            maxheap.emplace(dist, i);
        } else if (dist < maxheap.top().first) {
            maxheap.pop();
            maxheap.emplace(dist, i);
        }
    }

    vector<int> indices;

    while (!maxheap.empty()) {
        indices.push_back(maxheap.top().second);
        maxheap.pop();
    }

    reverse(indices.begin(), indices.end());
    return indices;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    vector<float> dataset, query;
    vector<int> match;

    try {
        dataset = load_fbin("dataset.fbin", (size_t)99000 * 128);
        query   = load_fbin("query.fbin",   (size_t)1000 * 128);
        match   = load_ibin("match.ibin",  1000);
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    int dim = 128, n = 99000, nq = 1000, k = 1;

    // Time only the searches; no printing inside the timed region.
    vector<int> found(nq, -1);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < nq; ++i) {
        auto indices = top_k_indices(dataset, n, dim, &query[(size_t)i * dim], k);
        if (!indices.empty()) found[i] = indices[0];
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    int correct = 0;
    for (int i = 0; i < nq; ++i) {
        if (found[i] < 0) continue;
        if (found[i] == match[i]) correct++;
    }

    cout << "Recall@1: " << (double)correct / nq << "\n";
    cout << "Total time: " << ms << " ms, avg per query: " << ms / nq << " ms\n";

    return 0;
}