///
/// Copyright (c) 2020 Sebastian Eibl
///
/// Permission is hereby granted, free of charge, to any person obtaining
/// a copy of this software and associated documentation files (the
/// "Software"), to deal in the Software without restriction, including
/// without limitation the rights to use, copy, modify, merge, publish,
/// distribute, sublicense, and/or sell copies of the Software, and to
/// permit persons to whom the Software is furnished to do so, subject to
/// the following conditions:
///
/// The above copyright notice and this permission notice shall be
/// included in all copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
/// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
/// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
/// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
/// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
/// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
/// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
///

#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace assumption
{
inline void log(const std::string& msg) { std::cout << msg << std::endl; }

template <typename T>
std::string generateAssertionMessage(const T& lhs,
                                     const std::string& lhsExpression,
                                     const std::string& opString,
                                     const std::string& filename,
                                     const int64_t lineno,
                                     const std::string& /*function*/)
{
    std::stringstream ss;
    int length = static_cast<int>(lhsExpression.length());
    ss << "========================================================================\n"
       << "ASSUMPTION FAILED! (" << filename << ":" << lineno << ")\n\n"
       << "Expression: " << opString << "( " << lhsExpression << " )\n"
       << "Value:     " << std::setw(length) << std::setfill(' ') << lhsExpression << " = " << lhs;
    return ss.str();
}

template <typename T, typename U>
std::string generateAssertionMessage(const T& lhs,
                                     const U& rhs,
                                     const std::string& lhsExpression,
                                     const std::string& rhsExpression,
                                     const std::string& opString,
                                     const std::string& filename,
                                     const int64_t lineno,
                                     const std::string& /*function*/)
{
    std::stringstream ss;
    int length = static_cast<int>(std::max(lhsExpression.length(), rhsExpression.length()));
    ss << "========================================================================\n"
       << "ASSUMPTION FAILED! (" << filename << ":" << lineno << ")\n\n"
       << "Expression: " << lhsExpression << opString << rhsExpression << "\n"
       << "Values:     " << std::setw(length) << std::setfill(' ') << lhsExpression << " = " << lhs
       << "\n"
       << "            " << std::setw(length) << std::setfill(' ') << rhsExpression << " = " << rhs;
    return ss.str();
}

#define LOG(msg)                   \
    {                              \
        std::stringstream ss;      \
        ss << msg;                 \
        assumption::log(ss.str()); \
    }

template <typename FLOAT>
bool isFloatEqual(const FLOAT& lhs, const FLOAT& rhs)
{
    return std::abs(lhs - rhs) < FLOAT(1e-10);
}

// macro overloading (-> https://stackoverflow.com/a/24028231)

#define GLUE(x, y) x y

#define RETURN_ARG_COUNT(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, _10_, count, ...) count
#define EXPAND_ARGS(args) RETURN_ARG_COUNT args
#define COUNT_ARGS_MAX10(...) EXPAND_ARGS((__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

#define OVERLOAD_MACRO2(name, count) name##count
#define OVERLOAD_MACRO1(name, count) OVERLOAD_MACRO2(name, count)
#define OVERLOAD_MACRO(name, count) OVERLOAD_MACRO1(name, count)

#define MACRO_OVERLOAD(name, ...) \
    GLUE(OVERLOAD_MACRO(name, COUNT_ARGS_MAX10(__VA_ARGS__)), (__VA_ARGS__))

#define MRMD_VERBOSE_CHECK_1(X)                                  \
    {                                                            \
        do                                                       \
        {                                                        \
            if (!((X)))                                          \
            {                                                    \
                LOG(assumption::generateAssertionMessage(        \
                    (X), #X, "", __FILE__, __LINE__, __func__)); \
                assert(false);                                   \
            }                                                    \
        } while (0);                                             \
    }
#define MRMD_VERBOSE_CHECK_2(X, MSG)                               \
    {                                                              \
        do                                                         \
        {                                                          \
            if (!((X)))                                            \
            {                                                      \
                LOG(assumption::generateAssertionMessage(          \
                        (X), #X, "", __FILE__, __LINE__, __func__) \
                    << "\n"                                        \
                    << MSG);                                       \
                assert(false);                                     \
            }                                                      \
        } while (0);                                               \
    }

#define MRMD_VERBOSE_CHECK_NULLPTR_1(X)                                 \
    {                                                                   \
        do                                                              \
        {                                                               \
            if (!((X) == nullptr))                                      \
            {                                                           \
                LOG(assumption::generateAssertionMessage(               \
                    (X), #X, "nullptr", __FILE__, __LINE__, __func__)); \
                assert(false);                                          \
            }                                                           \
        } while (0);                                                    \
    }
#define MRMD_VERBOSE_CHECK_NULLPTR_2(X, MSG)                              \
    {                                                                     \
        do                                                                \
        {                                                                 \
            if (!((X) == nullptr))                                        \
            {                                                             \
                LOG(assumption::generateAssertionMessage(                 \
                        (X), #X, "nullptr", __FILE__, __LINE__, __func__) \
                    << "\n"                                               \
                    << MSG);                                              \
                assert(false);                                            \
            }                                                             \
        } while (0);                                                      \
    }

#define MRMD_VERBOSE_CHECK_NOT_NULLPTR_1(X)                              \
    {                                                                    \
        do                                                               \
        {                                                                \
            if (!((X) != nullptr))                                       \
            {                                                            \
                LOG(assumption::generateAssertionMessage(                \
                    (X), #X, "!nullptr", __FILE__, __LINE__, __func__)); \
                assert(false);                                           \
            }                                                            \
        } while (0);                                                     \
    }
#define MRMD_VERBOSE_CHECK_NOT_NULLPTR_2(X, MSG)                           \
    {                                                                      \
        do                                                                 \
        {                                                                  \
            if (!((X) != nullptr))                                         \
            {                                                              \
                LOG(assumption::generateAssertionMessage(                  \
                        (X), #X, "!nullptr", __FILE__, __LINE__, __func__) \
                    << "\n"                                                \
                    << MSG);                                               \
                assert(false);                                             \
            }                                                              \
        } while (0);                                                       \
    }

#define MRMD_VERBOSE_CHECK_EQUAL_2(X, Y)                                      \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!((X) == (Y)))                                                \
            {                                                                 \
                LOG(assumption::generateAssertionMessage(                     \
                    (X), (Y), #X, #Y, " == ", __FILE__, __LINE__, __func__)); \
                assert(false);                                                \
            }                                                                 \
        } while (0);                                                          \
    }
#define MRMD_VERBOSE_CHECK_EQUAL_3(X, Y, MSG)                                   \
    {                                                                           \
        do                                                                      \
        {                                                                       \
            if (!((X) == (Y)))                                                  \
            {                                                                   \
                LOG(assumption::generateAssertionMessage(                       \
                        (X), (Y), #X, #Y, " == ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                     \
                    << MSG);                                                    \
                assert(false);                                                  \
            }                                                                   \
        } while (0);                                                            \
    }

#define MRMD_VERBOSE_CHECK_FLOAT_EQUAL_2(X, Y)                                \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!(isFloatEqual((X), (Y))))                                    \
            {                                                                 \
                LOG(assumption::generateAssertionMessage(                     \
                    (X), (Y), #X, #Y, " == ", __FILE__, __LINE__, __func__)); \
                assert(false);                                                \
            }                                                                 \
        } while (0);                                                          \
    }
#define MRMD_VERBOSE_CHECK_FLOAT_EQUAL_3(X, Y, MSG)                             \
    {                                                                           \
        do                                                                      \
        {                                                                       \
            if (!(isFloatEqual((X), (Y))))                                      \
            {                                                                   \
                LOG(assumption::generateAssertionMessage(                       \
                        (X), (Y), #X, #Y, " == ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                     \
                    << MSG);                                                    \
                assert(false);                                                  \
            }                                                                   \
        } while (0);                                                            \
    }

#define MRMD_VERBOSE_CHECK_NOT_EQUAL_2(X, Y)                                  \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!((X) != (Y)))                                                \
            {                                                                 \
                LOG(assumption::generateAssertionMessage(                     \
                    (X), (Y), #X, #Y, " != ", __FILE__, __LINE__, __func__)); \
                assert(false);                                                \
            }                                                                 \
        } while (0);                                                          \
    }
#define MRMD_VERBOSE_CHECK_NOT_EQUAL_3(X, Y, MSG)                               \
    {                                                                           \
        do                                                                      \
        {                                                                       \
            if (!((X) != (Y)))                                                  \
            {                                                                   \
                LOG(assumption::generateAssertionMessage(                       \
                        (X), (Y), #X, #Y, " != ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                     \
                    << MSG);                                                    \
                assert(false);                                                  \
            }                                                                   \
        } while (0);                                                            \
    }

#define MRMD_VERBOSE_CHECK_GREATER_2(X, Y)                                   \
    {                                                                        \
        do                                                                   \
        {                                                                    \
            if (!((X) > (Y)))                                                \
            {                                                                \
                LOG(assumption::generateAssertionMessage(                    \
                    (X), (Y), #X, #Y, " > ", __FILE__, __LINE__, __func__)); \
                assert(false);                                               \
            }                                                                \
        } while (0);                                                         \
    }
#define MRMD_VERBOSE_CHECK_GREATER_3(X, Y, MSG)                                \
    {                                                                          \
        do                                                                     \
        {                                                                      \
            if (!((X) > (Y)))                                                  \
            {                                                                  \
                LOG(assumption::generateAssertionMessage(                      \
                        (X), (Y), #X, #Y, " > ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                    \
                    << MSG);                                                   \
                assert(false);                                                 \
            }                                                                  \
        } while (0);                                                           \
    }

#define MRMD_VERBOSE_CHECK_LESS_2(X, Y)                                      \
    {                                                                        \
        do                                                                   \
        {                                                                    \
            if (!((X) < (Y)))                                                \
            {                                                                \
                LOG(assumption::generateAssertionMessage(                    \
                    (X), (Y), #X, #Y, " < ", __FILE__, __LINE__, __func__)); \
                assert(false);                                               \
            }                                                                \
        } while (0);                                                         \
    }
#define MRMD_VERBOSE_CHECK_LESS_3(X, Y, MSG)                                   \
    {                                                                          \
        do                                                                     \
        {                                                                      \
            if (!((X) < (Y)))                                                  \
            {                                                                  \
                LOG(assumption::generateAssertionMessage(                      \
                        (X), (Y), #X, #Y, " < ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                    \
                    << MSG);                                                   \
                assert(false);                                                 \
            }                                                                  \
        } while (0);                                                           \
    }

#define MRMD_VERBOSE_CHECK_GREATEREQUAL_2(X, Y)                               \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!((X) >= (Y)))                                                \
            {                                                                 \
                LOG(assumption::generateAssertionMessage(                     \
                    (X), (Y), #X, #Y, " >= ", __FILE__, __LINE__, __func__)); \
                assert(false);                                                \
            }                                                                 \
        } while (0);                                                          \
    }
#define MRMD_VERBOSE_CHECK_GREATEREQUAL_3(X, Y, MSG)                            \
    {                                                                           \
        do                                                                      \
        {                                                                       \
            if (!((X) >= (Y)))                                                  \
            {                                                                   \
                LOG(assumption::generateAssertionMessage(                       \
                        (X), (Y), #X, #Y, " >= ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                     \
                    << MSG);                                                    \
                assert(false);                                                  \
            }                                                                   \
        } while (0);                                                            \
    }

#define MRMD_VERBOSE_CHECK_LESSEQUAL_2(X, Y)                                  \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!((X) <= (Y)))                                                \
            {                                                                 \
                LOG(assumption::generateAssertionMessage(                     \
                    (X), (Y), #X, #Y, " <= ", __FILE__, __LINE__, __func__)); \
                assert(false);                                                \
            }                                                                 \
        } while (0);                                                          \
    }
#define MRMD_VERBOSE_CHECK_LESSEQUAL_3(X, Y, MSG)                               \
    {                                                                           \
        do                                                                      \
        {                                                                       \
            if (!((X) <= (Y)))                                                  \
            {                                                                   \
                LOG(assumption::generateAssertionMessage(                       \
                        (X), (Y), #X, #Y, " <= ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                     \
                    << MSG);                                                    \
                assert(false);                                                  \
            }                                                                   \
        } while (0);                                                            \
    }

#define MRMD_VERBOSE_CHECK(...) MACRO_OVERLOAD(MRMD_VERBOSE_CHECK_, __VA_ARGS__)
#define MRMD_VERBOSE_CHECK_NULLPTR(...) MACRO_OVERLOAD(MRMD_VERBOSE_CHECK_EQUAL_, __VA_ARGS__)
#define MRMD_VERBOSE_CHECK_NOT_NULLPTR(...) \
    MACRO_OVERLOAD(MRMD_VERBOSE_CHECK_NOT_NULLPTR_, __VA_ARGS__)
#define MRMD_VERBOSE_CHECK_EQUAL(...) MACRO_OVERLOAD(MRMD_VERBOSE_CHECK_EQUAL_, __VA_ARGS__)
#define MRMD_VERBOSE_CHECK_FLOAT_EQUAL(...) MACRO_OVERLOAD(MRMD_VERBOSE_CHECK_EQUAL_, __VA_ARGS__)
#define MRMD_VERBOSE_CHECK_NOT_EQUAL(...) MACRO_OVERLOAD(MRMD_VERBOSE_CHECK_NOT_EQUAL_, __VA_ARGS__)
#define MRMD_VERBOSE_CHECK_GREATER(...) MACRO_OVERLOAD(MRMD_VERBOSE_CHECK_GREATER_, __VA_ARGS__)
#define MRMD_VERBOSE_CHECK_LESS(...) MACRO_OVERLOAD(MRMD_VERBOSE_CHECK_LESS_, __VA_ARGS__)
#define MRMD_VERBOSE_CHECK_GREATEREQUAL(...) \
    MACRO_OVERLOAD(MRMD_VERBOSE_CHECK_GREATEREQUAL_, __VA_ARGS__)
#define MRMD_VERBOSE_CHECK_LESSEQUAL(...) MACRO_OVERLOAD(MRMD_VERBOSE_CHECK_LESSEQUAL_, __VA_ARGS__)

}  // namespace assumption