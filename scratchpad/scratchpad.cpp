#include <tomgrad/tomgrad.hpp>
#include <print>

int main() {
    using namespace tomgrad;

    auto t = zeros<double>({3, 3});
    std::println("Size: {} elements", t.data().size());
    t.print();

    return 0;
}
