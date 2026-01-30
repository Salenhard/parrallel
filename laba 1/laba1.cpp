#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <fstream>

double average_omp(const double* vector, size_t n) {
    double sum = 0.0;

    #pragma omp parallel
    {
        double local_sum = 0.0;
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();

        for (size_t i = t; i < n; i += T) {
            local_sum += vector[i];
        }

        #pragma omp atomic
        sum += local_sum;
    }

    return sum / n;
}

double avg(const double* vector, size_t n){
    double sum = 0.0;
    for(size_t i = 0; i < n; i++) {
        sum += vector[i];
    }
    return sum / n;
}

int main() {
    size_t N = 200000000;
    // std::cout << "Input N: ";
    // std::cin >> N;

    std::vector<double> data(N);
    for (size_t i = 0; i < N; i++)
        data[i] = i;

    auto start = std::chrono::high_resolution_clock::now();
    double ref = avg(data.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    double time_1 = std::chrono::duration<double>(end - start).count();

    std::cout << "Baseline avg = " << ref << ", time = " << time_1 << " s\n";

    std::ofstream csv("results.csv");
    csv << "T,time,acceleration\n";

    int max_threads = omp_get_max_threads();

    for (int T = 1; T <= max_threads; T++) {
        omp_set_num_threads(T);

        auto t_start = std::chrono::high_resolution_clock::now();
        double r = average_omp(data.data(), N);
        auto t_end = std::chrono::high_resolution_clock::now();

        double time_T = std::chrono::duration<double>(t_end - t_start).count();
        double acceleration = time_1 / time_T;

        csv << T << "," << time_T << "," << acceleration << "\n";

        std::cout << "T=" << T
                  << " time=" << time_T
                  << " accel=" << acceleration << std::endl;
    }

    csv.close();
    std::cout << "Saved to results.csv\n";

    return 0;
}

