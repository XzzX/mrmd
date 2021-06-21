/**
\file
Minimal working example:
\code
#include "assumption.h"

int main()
{
    ASSERT_EQUAL(4*5, 4*6, "simple " << "maths");
}
\endcode

Output:
\verbatim
ASSUMPTION FAILED! (example.cpp:5)

Expression: 4*5 == 4*6
Values:     4*5 = 20
            4*6 = 24

simple maths
\endverbatim
**/

#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace ecab
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
       << "Expression: " << opString << lhsExpression << "\n"
       << "Value:      " << std::setw(length) << std::setfill(' ') << lhsExpression << " = " << lhs;
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

#define LOG(msg)              \
    {                         \
        std::stringstream ss; \
        ss << msg;            \
        ecab::log(ss.str());  \
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

#define CHECK_TRUE_1(X)                                                                         \
    {                                                                                           \
        do                                                                                      \
        {                                                                                       \
            if (!((X)))                                                                         \
            {                                                                                   \
                LOG(ecab::generateAssertionMessage((X), #X, "", __FILE__, __LINE__, __func__)); \
                exit(EXIT_FAILURE);                                                             \
            }                                                                                   \
        } while (0);                                                                            \
    }
#define CHECK_TRUE_2(X, MSG)                                                                  \
    {                                                                                         \
        do                                                                                    \
        {                                                                                     \
            if (!((X)))                                                                       \
            {                                                                                 \
                LOG(ecab::generateAssertionMessage((X), #X, "", __FILE__, __LINE__, __func__) \
                    << "\n"                                                                   \
                    << MSG);                                                                  \
                exit(EXIT_FAILURE);                                                           \
            }                                                                                 \
        } while (0);                                                                          \
    }

#define CHECK_FALSE_1(X)                                                                         \
    {                                                                                            \
        do                                                                                       \
        {                                                                                        \
            if (!(!(X)))                                                                         \
            {                                                                                    \
                LOG(ecab::generateAssertionMessage((X), #X, "!", __FILE__, __LINE__, __func__)); \
                exit(EXIT_FAILURE);                                                              \
            }                                                                                    \
        } while (0);                                                                             \
    }
#define CHECK_FALSE_2(X, MSG)                                                                  \
    {                                                                                          \
        do                                                                                     \
        {                                                                                      \
            if (!(!(X)))                                                                       \
            {                                                                                  \
                LOG(ecab::generateAssertionMessage((X), #X, "!", __FILE__, __LINE__, __func__) \
                    << "\n"                                                                    \
                    << MSG);                                                                   \
                exit(EXIT_FAILURE);                                                            \
            }                                                                                  \
        } while (0);                                                                           \
    }

#define CHECK_NULLPTR_1(X)                                                  \
    {                                                                       \
        do                                                                  \
        {                                                                   \
            if (!((X) == nullptr))                                          \
            {                                                               \
                LOG(ecab::generateAssertionMessage(                         \
                    (X), #X, "nullptr == ", __FILE__, __LINE__, __func__)); \
                exit(EXIT_FAILURE);                                         \
            }                                                               \
        } while (0);                                                        \
    }
#define CHECK_NULLPTR_2(X, MSG)                                               \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!((X) == nullptr))                                            \
            {                                                                 \
                LOG(ecab::generateAssertionMessage(                           \
                        (X), #X, "nullptr == ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                   \
                    << MSG);                                                  \
                exit(EXIT_FAILURE);                                           \
            }                                                                 \
        } while (0);                                                          \
    }

#define CHECK_NOT_NULLPTR_1(X)                                              \
    {                                                                       \
        do                                                                  \
        {                                                                   \
            if (!((X) != nullptr))                                          \
            {                                                               \
                LOG(ecab::generateAssertionMessage(                         \
                    (X), #X, "nullptr != ", __FILE__, __LINE__, __func__)); \
                exit(EXIT_FAILURE);                                         \
            }                                                               \
        } while (0);                                                        \
    }
#define CHECK_NOT_NULLPTR_2(X, MSG)                                           \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!((X) != nullptr))                                            \
            {                                                                 \
                LOG(ecab::generateAssertionMessage(                           \
                        (X), #X, "nullptr != ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                   \
                    << MSG);                                                  \
                exit(EXIT_FAILURE);                                           \
            }                                                                 \
        } while (0);                                                          \
    }

#define CHECK_EQUAL_2(X, Y)                                                   \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!((X) == (Y)))                                                \
            {                                                                 \
                LOG(ecab::generateAssertionMessage(                           \
                    (X), (Y), #X, #Y, " == ", __FILE__, __LINE__, __func__)); \
                exit(EXIT_FAILURE);                                           \
            }                                                                 \
        } while (0);                                                          \
    }
#define CHECK_EQUAL_3(X, Y, MSG)                                                \
    {                                                                           \
        do                                                                      \
        {                                                                       \
            if (!((X) == (Y)))                                                  \
            {                                                                   \
                LOG(ecab::generateAssertionMessage(                             \
                        (X), (Y), #X, #Y, " == ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                     \
                    << MSG);                                                    \
                exit(EXIT_FAILURE);                                             \
            }                                                                   \
        } while (0);                                                            \
    }

#define CHECK_NOT_EQUAL_2(X, Y)                                               \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!((X) != (Y)))                                                \
            {                                                                 \
                LOG(ecab::generateAssertionMessage(                           \
                    (X), (Y), #X, #Y, " != ", __FILE__, __LINE__, __func__)); \
                exit(EXIT_FAILURE);                                           \
            }                                                                 \
        } while (0);                                                          \
    }
#define CHECK_NOT_EQUAL_3(X, Y, MSG)                                            \
    {                                                                           \
        do                                                                      \
        {                                                                       \
            if (!((X) != (Y)))                                                  \
            {                                                                   \
                LOG(ecab::generateAssertionMessage(                             \
                        (X), (Y), #X, #Y, " != ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                     \
                    << MSG);                                                    \
                exit(EXIT_FAILURE);                                             \
            }                                                                   \
        } while (0);                                                            \
    }

#define CHECK_GREATER_2(X, Y)                                                \
    {                                                                        \
        do                                                                   \
        {                                                                    \
            if (!((X) > (Y)))                                                \
            {                                                                \
                LOG(ecab::generateAssertionMessage(                          \
                    (X), (Y), #X, #Y, " > ", __FILE__, __LINE__, __func__)); \
                exit(EXIT_FAILURE);                                          \
            }                                                                \
        } while (0);                                                         \
    }
#define CHECK_GREATER_3(X, Y, MSG)                                             \
    {                                                                          \
        do                                                                     \
        {                                                                      \
            if (!((X) > (Y)))                                                  \
            {                                                                  \
                LOG(ecab::generateAssertionMessage(                            \
                        (X), (Y), #X, #Y, " > ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                    \
                    << MSG);                                                   \
                exit(EXIT_FAILURE);                                            \
            }                                                                  \
        } while (0);                                                           \
    }

#define CHECK_LESS_2(X, Y)                                                   \
    {                                                                        \
        do                                                                   \
        {                                                                    \
            if (!((X) < (Y)))                                                \
            {                                                                \
                LOG(ecab::generateAssertionMessage(                          \
                    (X), (Y), #X, #Y, " < ", __FILE__, __LINE__, __func__)); \
                exit(EXIT_FAILURE);                                          \
            }                                                                \
        } while (0);                                                         \
    }
#define CHECK_LESS_3(X, Y, MSG)                                                \
    {                                                                          \
        do                                                                     \
        {                                                                      \
            if (!((X) < (Y)))                                                  \
            {                                                                  \
                LOG(ecab::generateAssertionMessage(                            \
                        (X), (Y), #X, #Y, " < ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                    \
                    << MSG);                                                   \
                exit(EXIT_FAILURE);                                            \
            }                                                                  \
        } while (0);                                                           \
    }

#define CHECK_GREATER_EQUAL_2(X, Y)                                           \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!((X) >= (Y)))                                                \
            {                                                                 \
                LOG(ecab::generateAssertionMessage(                           \
                    (X), (Y), #X, #Y, " >= ", __FILE__, __LINE__, __func__)); \
                exit(EXIT_FAILURE);                                           \
            }                                                                 \
        } while (0);                                                          \
    }
#define CHECK_GREATER_EQUAL_3(X, Y, MSG)                                        \
    {                                                                           \
        do                                                                      \
        {                                                                       \
            if (!((X) >= (Y)))                                                  \
            {                                                                   \
                LOG(ecab::generateAssertionMessage(                             \
                        (X), (Y), #X, #Y, " >= ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                     \
                    << MSG);                                                    \
                exit(EXIT_FAILURE);                                             \
            }                                                                   \
        } while (0);                                                            \
    }

#define CHECK_LESS_EQUAL_2(X, Y)                                              \
    {                                                                         \
        do                                                                    \
        {                                                                     \
            if (!((X) <= (Y)))                                                \
            {                                                                 \
                LOG(ecab::generateAssertionMessage(                           \
                    (X), (Y), #X, #Y, " <= ", __FILE__, __LINE__, __func__)); \
                exit(EXIT_FAILURE);                                           \
            }                                                                 \
        } while (0);                                                          \
    }
#define CHECK_LESS_EQUAL_3(X, Y, MSG)                                           \
    {                                                                           \
        do                                                                      \
        {                                                                       \
            if (!((X) <= (Y)))                                                  \
            {                                                                   \
                LOG(ecab::generateAssertionMessage(                             \
                        (X), (Y), #X, #Y, " <= ", __FILE__, __LINE__, __func__) \
                    << "\n"                                                     \
                    << MSG);                                                    \
                exit(EXIT_FAILURE);                                             \
            }                                                                   \
        } while (0);                                                            \
    }

#define CHECK_TRUE(...) MACRO_OVERLOAD(CHECK_TRUE_, __VA_ARGS__) void(0)
#define CHECK_FALSE(...) MACRO_OVERLOAD(CHECK_FALSE_, __VA_ARGS__) void(0)
#define CHECK_NULLPTR(...) MACRO_OVERLOAD(CHECK_EQUAL_, __VA_ARGS__) void(0)
#define CHECK_NOT_NULLPTR(...) MACRO_OVERLOAD(CHECK_NOT_NULLPTR_, __VA_ARGS__) void(0)
#define CHECK_EQUAL(...) MACRO_OVERLOAD(CHECK_EQUAL_, __VA_ARGS__) void(0)
#define CHECK_NOT_EQUAL(...) MACRO_OVERLOAD(CHECK_NOT_EQUAL_, __VA_ARGS__) void(0)
#define CHECK_GREATER(...) MACRO_OVERLOAD(CHECK_GREATER_, __VA_ARGS__) void(0)
#define CHECK_LESS(...) MACRO_OVERLOAD(CHECK_LESS_, __VA_ARGS__) void(0)
#define CHECK_GREATER_EQUAL(...) MACRO_OVERLOAD(CHECK_GREATER_EQUAL_, __VA_ARGS__) void(0)
#define CHECK_LESS_EQUAL(...) MACRO_OVERLOAD(CHECK_LESS_EQUAL_, __VA_ARGS__) void(0)

#ifndef NDEBUG

#define ASSERT_TRUE_1(X) \
    {                    \
        CHECK_TRUE_1(X)  \
    }
#define ASSERT_TRUE_2(X, MSG) \
    {                         \
        CHECK_TRUE_2(X, MSG)  \
    }

#define ASSERT_FALSE_1(X) \
    {                     \
        CHECK_FALSE_1(X)  \
    }
#define ASSERT_FALSE_2(X, MSG) \
    {                          \
        CHECK_FALSE_2(X, MSG)  \
    }

#define ASSERT_NULLPTR_1(X) \
    {                       \
        CHECK_NULLPTR_1(X)  \
    }
#define ASSERT_NULLPTR_2(X, MSG) \
    {                            \
        CHECK_NULLPTR_2(X, MSG)  \
    }

#define ASSERT_NOT_NULLPTR_1(X) \
    {                           \
        CHECK_NOT_NULLPTR_1(X)  \
    }
#define ASSERT_NOT_NULLPTR_2(X, MSG) \
    {                                \
        CHECK_NOT_NULLPTR_2(X, MSG)  \
    }

#define ASSERT_EQUAL_2(X, Y) \
    {                        \
        CHECK_EQUAL_2(X, Y)  \
    }
#define ASSERT_EQUAL_3(X, Y, MSG) \
    {                             \
        CHECK_EQUAL_3(X, Y, MSG)  \
    }

#define ASSERT_NOT_EQUAL_2(X, Y) \
    {                            \
        CHECK_NOT_EQUAL_2(X, Y)  \
    }
#define ASSERT_NOT_EQUAL_3(X, Y, MSG) \
    {                                 \
        CHECK_NOT_EQUAL_3(X, Y, MSG)  \
    }

#define ASSERT_GREATER_2(X, Y) \
    {                          \
        CHECK_GREATER_2(X, Y)  \
    }
#define ASSERT_GREATER_3(X, Y, MSG) \
    {                               \
        CHECK_GREATER_3(X, Y, MSG)  \
    }

#define ASSERT_LESS_2(X, Y) \
    {                       \
        CHECK_LESS_2(X, Y)  \
    }
#define ASSERT_LESS_3(X, Y, MSG) \
    {                            \
        CHECK_LESS_3(X, Y, MSG)  \
    }

#define ASSERT_GREATER_EQUAL_2(X, Y) \
    {                                \
        CHECK_GREATER_EQUAL_2(X, Y)  \
    }
#define ASSERT_GREATER_EQUAL_3(X, Y, MSG) \
    {                                     \
        CHECK_GREATER_EQUAL_3(X, Y, MSG)  \
    }

#define ASSERT_LESS_EQUAL_2(X, Y) \
    {                             \
        CHECK_LESS_EQUAL_2(X, Y)  \
    }
#define ASSERT_LESS_EQUAL_3(X, Y, MSG) \
    {                                  \
        CHECK_LESS_EQUAL_3(X, Y, MSG)  \
    }

#else

#define ASSERT_TRUE_1(X) \
    {                    \
    }
#define ASSERT_TRUE_2(X, MSG) \
    {                         \
    }

#define ASSERT_FALSE_1(X) \
    {                     \
    }
#define ASSERT_FALSE_2(X, MSG) \
    {                          \
    }

#define ASSERT_NULLPTR_1(X) \
    {                       \
    }
#define ASSERT_NULLPTR_2(X, MSG) \
    {                            \
    }

#define ASSERT_NOT_NULLPTR_1(X) \
    {                           \
    }
#define ASSERT_NOT_NULLPTR_2(X, MSG) \
    {                                \
    }

#define ASSERT_EQUAL_2(X, Y) \
    {                        \
    }
#define ASSERT_EQUAL_3(X, Y, MSG) \
    {                             \
    }

#define ASSERT_NOT_EQUAL_2(X, Y) \
    {                            \
    }
#define ASSERT_NOT_EQUAL_3(X, Y, MSG) \
    {                                 \
    }

#define ASSERT_GREATER_2(X, Y) \
    {                          \
    }
#define ASSERT_GREATER_3(X, Y, MSG) \
    {                               \
    }

#define ASSERT_LESS_2(X, Y) \
    {                       \
    }
#define ASSERT_LESS_3(X, Y, MSG) \
    {                            \
    }

#define ASSERT_GREATER_EQUAL_2(X, Y) \
    {                                \
    }
#define ASSERT_GREATER_EQUAL_3(X, Y, MSG) \
    {                                     \
    }

#define ASSERT_LESS_EQUAL_2(X, Y) \
    {                             \
    }
#define ASSERT_LESS_EQUAL_3(X, Y, MSG) \
    {                                  \
    }

#endif

#define ASSERT_TRUE(...) MACRO_OVERLOAD(ASSERT_TRUE_, __VA_ARGS__) void(0)
#define ASSERT_FALSE(...) MACRO_OVERLOAD(ASSERT_FALSE_, __VA_ARGS__) void(0)
#define ASSERT_NULLPTR(...) MACRO_OVERLOAD(ASSERT_EQUAL_, __VA_ARGS__) void(0)
#define ASSERT_NOT_NULLPTR(...) MACRO_OVERLOAD(ASSERT_NOT_NULLPTR_, __VA_ARGS__) void(0)
#define ASSERT_EQUAL(...) MACRO_OVERLOAD(ASSERT_EQUAL_, __VA_ARGS__) void(0)
#define ASSERT_NOT_EQUAL(...) MACRO_OVERLOAD(ASSERT_NOT_EQUAL_, __VA_ARGS__) void(0)
#define ASSERT_GREATER(...) MACRO_OVERLOAD(ASSERT_GREATER_, __VA_ARGS__) void(0)
#define ASSERT_LESS(...) MACRO_OVERLOAD(ASSERT_LESS_, __VA_ARGS__) void(0)
#define ASSERT_GREATEREQUAL(...) MACRO_OVERLOAD(ASSERT_GREATER_EQUAL_, __VA_ARGS__) void(0)
#define ASSERT_LESSEQUAL(...) MACRO_OVERLOAD(ASSERT_LESS_EQUAL_, __VA_ARGS__) void(0)

}  // namespace ecab
