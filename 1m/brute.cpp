#include <bits/stdc++.h>
#include <chrono>
using namespace std;

vector<float> load_fbin(const string& path, size_t expected_count) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("cannot open " + path);
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
    int dim = 128, n = 1000000, nq = 1000, k = 1;
    vector<float> dataset, query;
    vector<int> match;
    try {
        dataset = load_fbin("dataset.fbin", (size_t)n * dim);
        query = load_fbin("query.fbin", (size_t)nq * dim);
        match = load_ibin("match_top1.ibin", (size_t)nq);
        cout << "Loaded dataset: " << dataset.size() << " floats\n";
        cout << "Loaded query: " << query.size() << " floats\n";
        cout << "Loaded match: " << match.size() << " ints\n";
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    int exact = 0, tie = 0, identical = 0, brute_closer = 0, truth_closer = 0;
    vector<array<int, 4>> ties;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < nq; ++i) {
        const float* q = &query[(size_t)i * dim];
        auto indices = top_k_indices(dataset, n, dim, q, k);
        int found = indices[0];
        if (found == match[i]) {
            exact++;
        } else {
            const float* pf = &dataset[(size_t)found * dim];
            const float* pt = &dataset[(size_t)match[i] * dim];
            float d_found = squared_distance(pf, q, dim);
            float d_truth = squared_distance(pt, q, dim);
            if (d_found == d_truth) {
                tie++;
                bool same = memcmp(pf, pt, dim * sizeof(float)) == 0;
                if (same) identical++;
                ties.push_back({i, found, match[i], same ? 1 : 0});
            } else if (d_found < d_truth) {
                brute_closer++;
            } else {
                truth_closer++;
            }
        }
    }
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count();
    cout << "Recall@1 exact-id: " << (double)exact / nq << "\n";
    cout << "Recall@1 tie-aware: " << (double)(exact + tie) / nq << "\n";
    cout << "mismatches: ties " << tie << ", identical vectors " << identical
         << ", brute closer than truth " << brute_closer
         << ", truth closer than brute " << truth_closer << "\n";
    for (auto& t : ties)
        cout << "tie q=" << t[0] << " found=" << t[1] << " truth=" << t[2]
             << " diff=" << abs(t[1] - t[2]) << " identical=" << t[3] << "\n";
    cout << "Total time: " << ms << " ms, avg per query: " << ms / nq << " ms\n";
    return 0;
}