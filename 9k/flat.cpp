#include <bits/stdc++.h>
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

double probe_ms(const vector<float>& dataset, const vector<float>& query, int n, int dim, int probe_queries) {
    volatile int sink = 0;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < probe_queries; ++i) {
        auto idx = top_k_indices(dataset, n, dim, &query[(size_t)i * dim], 1);
        sink = idx[0];
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count() / probe_queries;
}

vector<int> search_layer(const vector<float>& dataset, const vector<vector<int>>& neighbors, int dim, const float* q, int entry, int ef) {
    priority_queue<pair<float, int>> max_heap;
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> min_heap;
    vector<bool> visited(neighbors.size(), false);

    float dist = squared_distance(&dataset[(size_t)entry * dim], q, dim);
    max_heap.emplace(dist, entry);
    min_heap.emplace(dist, entry);
    visited[entry] = true;

    while (!min_heap.empty()) {
        auto [curr_dist, curr_node] = min_heap.top();
        if ((int)max_heap.size() == ef && curr_dist > max_heap.top().first) break;
        min_heap.pop();

        for (int neighbor : neighbors[curr_node]) {
            if (visited[neighbor]) continue;
            visited[neighbor] = true;

            float neighbor_dist = squared_distance(&dataset[(size_t)neighbor * dim], q, dim);

            if ((int)max_heap.size() < ef || neighbor_dist < max_heap.top().first) {
                max_heap.emplace(neighbor_dist, neighbor);
                min_heap.emplace(neighbor_dist, neighbor);
                if ((int)max_heap.size() > ef) max_heap.pop();
            }
        }
    }

    vector<int> result;
    while (!max_heap.empty()) {
        result.push_back(max_heap.top().second);
        max_heap.pop();
    }
    reverse(result.begin(), result.end());
    return result;
}

void shrink_neighbors(vector<int>& list, const vector<float>& dataset, int dim, int x, int Mmax) {
    if ((int)list.size() <= Mmax) return;
    vector<pair<float, int>> scored;
    scored.reserve(list.size());
    for (int id : list) scored.emplace_back(squared_distance(&dataset[(size_t)id * dim], &dataset[(size_t)x * dim], dim), id);
    sort(scored.begin(), scored.end());
    list.clear();
    for (int i = 0; i < Mmax; ++i) list.push_back(scored[i].second);
}

vector<vector<int>> build_graph_insert(const vector<float>& dataset, int n, int dim, int M, int ef_construction, int Mmax) {
    vector<vector<int>> neighbors(n);
    for (int p = 1; p < n; ++p) {
        auto res = search_layer(dataset, neighbors, dim, &dataset[(size_t)p * dim], 0, ef_construction);
        for (int j = 0; j < M && j < (int)res.size(); ++j) {
            neighbors[p].push_back(res[j]);
            neighbors[res[j]].push_back(p);
            if ((int)neighbors[res[j]].size() > Mmax) shrink_neighbors(neighbors[res[j]], dataset, dim, res[j], Mmax);
        }
    }
    return neighbors;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(0);
    int dim = 128, n = 9000, nq = 1000, gt_k = 10, M = 16;
    int num_sweeps = argc > 1 ? atoi(argv[1]) : 5;
    int probe_queries = 100;
    const double probe_ref = 2.097, probe_limit = 1.08;
    vector<float> dataset, query;
    vector<int> match;
    try {
        dataset = load_fbin("dataset.fbin", (size_t)n * dim);
        query = load_fbin("query.fbin", (size_t)nq * dim);
        match = load_ibin("match.ibin", (size_t)nq * gt_k);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    double probe_pre = probe_ms(dataset, query, n, dim, probe_queries);
    cout << "probe pre-build: " << probe_pre << " ms/query\n";

    auto start = chrono::high_resolution_clock::now();
    auto neighbors = build_graph_insert(dataset, n, dim, M, 100, 32);
    auto end = chrono::high_resolution_clock::now();
    size_t max_deg = 0;
    long long total_deg = 0;
    for (auto& v : neighbors) { max_deg = max(max_deg, v.size()); total_deg += v.size(); }
    cout << "Max degree: " << max_deg << "  avg degree: " << (double)total_deg / n << "\n";
    cout << "Graph build: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

    vector<int> efs = {10, 20, 50, 100, 200};
    double before = probe_ms(dataset, query, n, dim, probe_queries);
    cout << "probe start: " << before << " ms/query\n";

    for (int s = 0; s < num_sweeps; ++s) {
        vector<vector<int>> found(efs.size(), vector<int>(nq, -1));
        vector<double> times(efs.size());
        for (size_t e = 0; e < efs.size(); ++e) {
            start = chrono::high_resolution_clock::now();
            for (int i = 0; i < nq; ++i) {
                auto res = search_layer(dataset, neighbors, dim, &query[(size_t)i * dim], 0, efs[e]);
                if (!res.empty()) found[e][i] = res[0];
            }
            end = chrono::high_resolution_clock::now();
            times[e] = chrono::duration<double, milli>(end - start).count() / nq;
        }
        double after = probe_ms(dataset, query, n, dim, probe_queries);
        bool ok = before <= probe_ref * probe_limit && after <= probe_ref * probe_limit;
        cout << "sweep " << s << "  probe_before " << before << "  probe_after " << after
             << "  status " << (ok ? "OK" : "SLOW") << "\n";
        for (size_t e = 0; e < efs.size(); ++e) {
            int correct = 0;
            for (int i = 0; i < nq; ++i)
                if (found[e][i] == match[(size_t)i * gt_k]) correct++;
            cout << "  ef=" << efs[e] << "  recall@1: " << (double)correct / nq
                 << "  avg per query: " << times[e] << " ms\n";
        }
        before = after;
    }
    return 0;
}