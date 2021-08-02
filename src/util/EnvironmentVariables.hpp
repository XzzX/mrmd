#pragma once

#include "datatypes.hpp"

namespace mrmd
{
namespace util
{
inline std::string getEnvironmentVariable(const std::string& variable)
{
    auto value = std::getenv(variable.c_str()) != nullptr
                     ? std::string(std::getenv(variable.c_str()))
                     : std::string("");
    return value;
}

}  // namespace util
}  // namespace mrmd
