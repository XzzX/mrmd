#pragma once


// convenience casting functions for unsigned ints
    template <typename T>
    inline constexpr unsigned int uint_c(const T &val)
    {
        return static_cast<unsigned int>(val);
    }

    template <typename T>
    inline constexpr uint8_t uint8_c(const T &val)
    {
        return static_cast<uint8_t>(val);
    }

    template <typename T>
    inline constexpr uint16_t uint16_c(const T &val)
    {
        return static_cast<uint16_t>(val);
    }

    template <typename T>
    inline constexpr uint32_t uint32_c(const T &val)
    {
        return static_cast<uint32_t>(val);
    }

    template <typename T>
    inline constexpr uint64_t uint64_c(const T &val)
    {
        return static_cast<uint64_t>(val);
    }

// convenience casting functions for signed ints
    template <typename T>
    inline constexpr int int_c(const T &val)
    {
        return static_cast<int>(val);
    }

    template <typename T>
    inline constexpr int8_t int8_c(const T &val)
    {
        return static_cast<int8_t>(val);
    }

    template <typename T>
    inline constexpr int16_t int16_c(const T &val)
    {
        return static_cast<int16_t>(val);
    }

    template <typename T>
    inline constexpr int32_t int32_c(const T &val)
    {
        return static_cast<int32_t>(val);
    }

    template <typename T>
    inline constexpr int64_t int64_c(const T &val)
    {
        return static_cast<int64_t>(val);
    }

    using idx_t = int64_t;
    template <typename T>
    inline constexpr idx_t idx_c(const T &val)
    {
        return static_cast<idx_t>(val);
    }

// convenience casting functions for floating point
    using real_t = double;

    inline constexpr real_t operator"" _r(const long double val) { return static_cast<real_t>(val); }

    inline constexpr real_t operator"" _r(const unsigned long long val)
    {
        return static_cast<real_t>(val);
    }

    template <typename T>
    inline constexpr real_t real_c(T t)
    {
        return static_cast<real_t>(t);
    }

    template <typename T>
    inline constexpr double double_c(const T &val)
    {
        return static_cast<double>(val);
    }

    template <typename T>
    inline constexpr float float_c(const T &val)
    {
        return static_cast<float>(val);
    }
