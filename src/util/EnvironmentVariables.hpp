#pragma once

#include "datatypes.hpp"

namespace mrmd
{
namespace util
{
inline std::string getEnvironmentVariable(const std::string variable)
{
    auto value =
        std::getenv(variable) != nullptr ? std::string(std::getenv(variable)) : std::string("");
    return value;
}

}  // namespace util
}  // namespace mrmd
