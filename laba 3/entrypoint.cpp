#include "vector_mod.h"
#include "test.h"
#include "performance.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include "num_threads.h"

int main(int argc, char** argv)
{
    std::cout << "==Correctness tests. ";
    for (std::size_t iTest = 0; iTest < test_data_count; ++iTest)
    {
        if (test_data[iTest].result != vector_mod(test_data[iTest].dividend,
                                                   test_data[iTest].dividend_size,
                                                   test_data[iTest].divisor))
        {
            std::cout << "FAILURE==\n";
            return -1;
        }
    }
    std::cout << "ok.==\n";

    std::cout << "==Performance tests. ";
    auto measurements = run_experiments();
    std::cout << "Done==\n";

    // ---------- CSV ----------
    std::ofstream csv("results.csv");
    csv << "T,time,acceleration\n";

    std::cout << std::setfill(' ') << std::setw(2) << "T:" << " |"
              << std::setw(3 + 2 * sizeof(IntegerWord)) << "Value:" << " | "
              << std::setw(14) << "Duration, ms:" << " | Acceleration:\n";

    const double base_time = static_cast<double>(measurements[0].time.count());

    for (std::size_t T = 1; T <= measurements.size(); ++T)
    {
        const auto& m = measurements[T - 1];
        double acceleration = base_time / m.time.count();

        std::cout << std::setw(2) << T << " | 0x"
                  << std::setw(2 * sizeof(IntegerWord)) << std::setfill('0')
                  << std::hex << m.result;
        std::cout << " | " << std::setfill(' ')
                  << std::setw(14) << std::dec << m.time.count();
        std::cout << " | " << acceleration << "\n";

        csv << T << ","
            << m.time.count() << ","
            << acceleration << "\n";
    }

    csv.close();
    std::cout << "CSV saved to results.csv\n";

    return 0;
}
