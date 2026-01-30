#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

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
    std::cout << "Input N:";
    std::cin >> N;
    std::vector<double> data(N);

    for (size_t i = 0; i < N; i++)
        data[i] = i;

    std::cout << "N = " << N << std::endl;

    auto start_sync = std::chrono::high_resolution_clock::now();
    double r1 = avg(data.data(), N);
    auto end_sync = std::chrono::high_resolution_clock::now();
    double time_sync = std::chrono::duration<double>(end_sync - start_sync).count();

    auto start_omp = std::chrono::high_resolution_clock::now();
    double r2 = average_omp(data.data(), N);
    auto end_omp = std::chrono::high_resolution_clock::now();
    double time_omp = std::chrono::duration<double>(end_omp - start_omp).count();

    std::cout << "avg() result          = " << r1 << std::endl;
    std::cout << "average_omp() result  = " << r2 << std::endl;

    std::cout << "Sync time: "    << time_sync << " s" << std::endl;
    std::cout << "OMP time:  "    << time_omp << " s" << std::endl;

    return 0;
}
