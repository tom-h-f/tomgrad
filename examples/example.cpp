#include <tomgrad/tomgrad.hpp>
#include <iostream>

int main() {
    using namespace tomgrad;

    std::cout << "tomgrad - Header-only C++23 Library\n\n";

    auto t1 = zeros<double>({2, 3});
    std::cout << "Zeros tensor (2x3):\n";
    for (const auto& val : t1.data()) {
        std::cout << val << " ";
    }
    std::cout << "\n\n";

    auto t2 = ones<float>({2, 2});
    std::cout << "Ones tensor (2x2):\n";
    for (const auto& val : t2.data()) {
        std::cout << val << " ";
    }
    std::cout << "\n\n";

    auto t3 = fill<int>({3}, 42);
    std::cout << "Filled tensor (3x42):\n";
    for (const auto& val : t3.data()) {
        std::cout << val << " ";
    }
    std::cout << "\n\n";

    std::cout << "Example complete!\n";
    return 0;
}
