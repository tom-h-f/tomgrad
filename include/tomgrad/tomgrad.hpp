#pragma once

#include <cstddef>
#include <print>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tomgrad {

template <typename T>
concept Scalar = std::is_arithmetic_v<T>;

template <typename T> class Tensor {
public:
  using value_type = T;
  using reference = T &;
  using const_reference = const T &;

  Tensor() = default;

  explicit Tensor(std::vector<std::size_t> shape, T value = T{})
      : shape_(std::move(shape)), data_(compute_size(shape_), value) {}

  explicit Tensor(std::vector<std::size_t> shape, std::vector<T> data)
      : shape_(std::move(shape)), data_(std::move(data)) {
    if (data_.size() != compute_size(shape_)) {
      throw std::invalid_argument("Data size does not match shape");
    }
  }

  [[nodiscard]] const auto &shape() const noexcept { return shape_; }
  [[nodiscard]] const auto &data() const noexcept { return data_; }
  [[nodiscard]] auto &data() noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
  [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

  [[nodiscard]] const_reference operator[](std::size_t idx) const {
    return data_[idx];
  }
  [[nodiscard]] reference operator[](std::size_t idx) { return data_[idx]; }

  void print() const {
    std::println("Tensor:");
    std::println("\tShape: {}", shape_);
    std::print("\tData: \n\t");
    for (std::size_t i = 0; i < data_.size(); i++) {
      std::print("[{:.03f}] ", data_[i]);
      if ((i + 1) % 6 == 0) {
        std::println();
        std::print("\t");
      }
    }
    std::println();
  }

private:
  [[nodiscard]] static std::size_t
  compute_size(const std::vector<std::size_t> &shape) {
    if (shape.empty())
      return 0;
    std::size_t size = 1;
    for (auto dim : shape) {
      size *= dim;
    }
    return size;
  }

  std::vector<std::size_t> shape_;
  std::vector<T> data_;
};

template <Scalar T> Tensor<T> zeros(std::vector<std::size_t> shape) {
  return Tensor<T>{std::move(shape), T{0}};
}

template <Scalar T> Tensor<T> ones(std::vector<std::size_t> shape) {
  return Tensor<T>{std::move(shape), T{1}};
}

template <Scalar T> Tensor<T> fill(std::vector<std::size_t> shape, T value) {
  return Tensor<T>{std::move(shape), value};
}

} // namespace tomgrad
