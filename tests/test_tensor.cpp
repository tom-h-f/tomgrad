#include "catch2/catch_test_macros.hpp"
#include <cstddef>
#include <stdexcept>
#include <vector>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <tomgrad/tomgrad.hpp>

using namespace tomgrad;

TEST_CASE("Tensor construction", "[tensor]") {
    SECTION("Default construction") {
        Tensor<double> t;
        REQUIRE(t.empty());
        REQUIRE(t.size() == 0);
    }

    SECTION("Construction with shape and default value") {
        Tensor<int> t({2, 3});
        REQUIRE(t.shape().size() == 2);
        REQUIRE(t.shape()[0] == 2);
        REQUIRE(t.shape()[1] == 3);
        REQUIRE(t.size() == 6);
    }

    SECTION("Construction with shape and value") {
        Tensor<double> t({2, 2}, 3.14);
        REQUIRE(t.size() == 4);
        for (std::size_t i = 0; i < t.size(); ++i) {
            REQUIRE(t[i] == 3.14);
        }
    }

    SECTION("Construction with shape and data") {
        std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
        Tensor<double> t({2, 2}, data);
        REQUIRE(t.size() == 4);
        for (std::size_t i = 0; i < t.size(); ++i) {
            REQUIRE(t[i] == data[i]);
        }
    }

    SECTION("Invalid data size throws") {
        std::vector<double> data = {1.0, 2.0};
        REQUIRE_THROWS_AS(Tensor<double>({2, 2}, data), std::invalid_argument);
    }
}

TEST_CASE("Tensor utility functions", "[tensor]") {
    SECTION("zeros") {
        auto t = zeros<double>({3, 3});
        REQUIRE(t.size() == 9);
        for (std::size_t i = 0; i < t.size(); ++i) {
            REQUIRE(t[i] == 0.0);
        }
    }

    SECTION("ones") {
        auto t = ones<float>({2, 4});
        REQUIRE(t.size() == 8);
        for (std::size_t i = 0; i < t.size(); ++i) {
            REQUIRE(t[i] == 1.0f);
        }
    }

    SECTION("fill") {
        auto t = fill<int>({2, 2}, 42);
        REQUIRE(t.size() == 4);
        for (std::size_t i = 0; i < t.size(); ++i) {
            REQUIRE(t[i] == 42);
        }
    }
}

TEST_CASE("Tensor element access", "[tensor]") {
    Tensor<int> t({3}, std::vector<int>{10, 20, 30});
    REQUIRE(t[0] == 10);
    REQUIRE(t[1] == 20);
    REQUIRE(t[2] == 30);
}
