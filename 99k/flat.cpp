#include <bits/stdc++.h>
using namespace std;

vector<float> load_fbin(const string& path, size_t expected_count) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("cannot open " + path);

    vector<float> data(expected_count);

    in.read(
        reinterpret_cast<char*>(data.data()),
        expected_count * sizeof(float)
    );

    if (!in)
        throw runtime_error("read failed or file too short: " + path);

    return data;
}

vector<int> load_ibin(const string& path, size_t expected_count) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("cannot open " + path);

    vector<int> data(expected_count);

    in.read(
        reinterpret_cast<char*>(data.data()),
        expected_count * sizeof(int)
    );

    if (!in)
        throw runtime_error("read failed or file too short: " + path);

    return data;
}

float squared_distance(
    const float* a,
    const float* b,
    size_t dim
) {
    float dist = 0;

    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }

    return dist;
}

// Exact-match recall: counts how many of the top-k results appear in the
// ground-truth set for query i.
int count_correct(const vector<int>& res, const vector<int>& match, int i, int k) {
    int c = 0;
    for (int j = 0; j < k && j < (int)res.size(); ++j)
        for (int t = 0; t < k; ++t)
            if (res[j] == match[(size_t)i * k + t]) { c++; break; }
    return c;
}

vector<int> search_layer(
    const vector<float>& dataset,
    const vector<vector<int>>& neighbors,
    int dim,
    const float* q,
    int entry,
    int ef
) {
    priority_queue<pair<float, int>> max_heap;

    priority_queue<
        pair<float, int>,
        vector<pair<float, int>>,
        greater<pair<float, int>>
    > min_heap;

    vector<bool> visited(neighbors.size(), false);

    float dist = squared_distance(
        &dataset[(size_t)entry * dim],
        q,
        dim
    );

    max_heap.emplace(dist, entry);
    min_heap.emplace(dist, entry);

    visited[entry] = true;

    while (!min_heap.empty()) {

        auto [curr_dist, curr_node] = min_heap.top();

        if ((int)max_heap.size() >= ef &&
            curr_dist > max_heap.top().first) {
            break;
        }

        min_heap.pop();

        for (int neighbor : neighbors[curr_node]) {

            if (visited[neighbor])
                continue;

            visited[neighbor] = true;

            float neighbor_dist = squared_distance(
                &dataset[(size_t)neighbor * dim],
                q,
                dim
            );

            if ((int)max_heap.size() < ef ||
                neighbor_dist < max_heap.top().first) {

                max_heap.emplace(neighbor_dist, neighbor);
                min_heap.emplace(neighbor_dist, neighbor);

                if ((int)max_heap.size() > ef)
                    max_heap.pop();
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

void shrink_neighbors(
    vector<int>& list,
    const vector<float>& dataset,
    int dim,
    int x,
    int Mmax
) {
    if ((int)list.size() <= Mmax)
        return;

    vector<pair<float, int>> scored;

    scored.reserve(list.size());

    for (int id : list) {

        float dist = squared_distance(
            &dataset[(size_t)id * dim],
            &dataset[(size_t)x * dim],
            dim
        );

        scored.emplace_back(dist, id);
    }

    sort(scored.begin(), scored.end());

    list.clear();

    for (int i = 0; i < Mmax; ++i)
        list.push_back(scored[i].second);
}

vector<vector<int>> build_graph_insert(
    const vector<float>& dataset,
    int n,
    int dim,
    int M,
    int ef_construction,
    int Mmax
) {
    vector<vector<int>> neighbors(n);

    for (int p = 1; p < n; ++p) {

        auto res = search_layer(
            dataset,
            neighbors,
            dim,
            &dataset[(size_t)p * dim],
            0,
            ef_construction
        );

        for (int j = 0;
             j < M && j < (int)res.size();
             ++j) {

            neighbors[p].push_back(res[j]);

            neighbors[res[j]].push_back(p);

            if ((int)neighbors[res[j]].size() > Mmax) {

                shrink_neighbors(
                    neighbors[res[j]],
                    dataset,
                    dim,
                    res[j],
                    Mmax
                );
            }
        }
    }

    return neighbors;
}

#include <chrono>
#include <cstdio>

double canary_ms() {
    auto t0 = std::chrono::steady_clock::now();
    volatile double acc = 0.0;
    for (long i = 0; i < 200000000L; i++) acc = acc + i * 0.5;
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Usage: test [brute_ms_per_query]
// If you pass the brute-force ms/query from your separate brute run, speedup is printed.
int main(int argc, char** argv) {

    ios::sync_with_stdio(false);
    cin.tie(0);

    // printf("canary: %.1f ms\n", canary_ms());

    double brute_ms = (argc > 1) ? atof(argv[1]) : 0.0;

    vector<float> dataset, query;
    vector<int> match;

    try {

        // D2 dataset
        dataset = load_fbin(
            "dataset.fbin",
            (size_t)99000 * 128
        );

        // 1000 queries
        query = load_fbin(
            "query.fbin",
            (size_t)1000 * 128
        );

        // One ground-truth neighbor per query
        match = load_ibin(
            "match.ibin",
            1000
        );

        cout << "Loaded dataset: "
             << dataset.size()
             << " floats\n";

        cout << "Loaded query: "
             << query.size()
             << " floats\n";

        cout << "Loaded match: "
             << match.size()
             << " ints\n";

    }
    catch (const exception& e) {

        cerr << "Error: "
             << e.what()
             << "\n";

        return 1;
    }

    int dim = 128;
    int n = 99000;
    int nq = 1000;
    int k = 1;

    int M = 16;
    int ef_construction = 100;
    int Mmax = 32;



    // ==========================================
    // GRAPH BUILD
    // ==========================================

    cout << "\nBuilding graph...\n";

    auto start =
        chrono::high_resolution_clock::now();

    auto neighbors = build_graph_insert(
        dataset,
        n,
        dim,
        M,
        ef_construction,
        Mmax
    );

    auto end =
        chrono::high_resolution_clock::now();


    // Graph statistics

    size_t max_deg = 0;
    long long total_deg = 0;

    for (auto& v : neighbors) {

        max_deg = max(
            max_deg,
            v.size()
        );

        total_deg += v.size();
    }

    cout << "Max degree: "
         << max_deg
         << "\n";

    cout << "Avg degree: "
         << (double)total_deg / n
         << "\n";

    double build_ms =
        chrono::duration<double, milli>(
            end - start
        ).count();

    cout << "Graph build: "
         << build_ms
         << " ms\n";


    // ==========================================
    // GRAPH SEARCH
    // ==========================================

    vector<vector<int>> results(nq);

    for (int ef : {10, 20, 50, 100, 200}) {

        start =
            chrono::high_resolution_clock::now();

        for (int i = 0; i < nq; ++i) {

            results[i] = search_layer(
                dataset,
                neighbors,
                dim,
                &query[(size_t)i * dim],
                0,
                ef
            );
        }

        end =
            chrono::high_resolution_clock::now();

        double graph_ms =
            chrono::duration<double, milli>(
                end - start
            ).count() / nq;

        long long correct = 0;

        for (int i = 0; i < nq; ++i) {
            correct += count_correct(results[i], match, i, k);
        }

        double recall =
            (double)correct / ((double)nq * k);

        cout << "ef=" << ef
             << "  Recall@" << k << ": "
             << recall
             << "  avg per query: "
             << graph_ms
             << " ms";

        if (brute_ms > 0)
            cout << "  speedup: " << brute_ms / graph_ms << "x";

        cout << "\n";
    }

    return 0;
}