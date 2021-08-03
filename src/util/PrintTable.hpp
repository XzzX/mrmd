#pragma once

#include <iomanip>
#include <iostream>

#include "datatypes.hpp"

namespace mrmd
{
namespace util
{
template <typename HEAD>
void printTable(HEAD head)
{
    std::cout << " │ " << std::setw(10) << std::setprecision(2) << std::fixed << head << " │ "
              << std::endl;
}
template <typename HEAD, typename... TAIL>
void printTable(HEAD head, TAIL... tail)
{
    std::cout << " │ " << std::setw(10) << std::setprecision(2) << std::fixed << head;
    printTable(tail...);
}

template <typename HEAD>
void printTableSep(HEAD /*head*/)
{
    std::cout << "─┼─" << std::setw(10) << std::setprecision(2) << std::fixed << "──────────"
              << "─┼─" << std::endl;
}
template <typename HEAD, typename... TAIL>
void printTableSep(HEAD /*head*/, TAIL... tail)
{
    std::cout << "─┼─" << std::setw(10) << std::setprecision(2) << std::fixed << "──────────";
    printTableSep(tail...);
}

}  // namespace util
}  // namespace mrmd
