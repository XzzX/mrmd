// Copyright 2024 Sebastian Eibl
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "cmake.hpp"
#include "normal.hpp"
#include "verbose.hpp"

#ifdef VERBOSE_ASSERT
#define MRMD_DEVICE_CHECK_1(X)  \
    {                           \
        MRMD_VERBOSE_CHECK_1(X) \
    }
#define MRMD_DEVICE_CHECK_2(X, MSG)  \
    {                                \
        MRMD_VERBOSE_CHECK_2(X, MSG) \
    }

#define MRMD_DEVICE_CHECK_NULLPTR_1(X)  \
    {                                   \
        MRMD_VERBOSE_CHECK_NULLPTR_1(X) \
    }
#define MRMD_DEVICE_CHECK_NULLPTR_2(X, MSG)  \
    {                                        \
        MRMD_VERBOSE_CHECK_NULLPTR_2(X, MSG) \
    }

#define MRMD_DEVICE_CHECK_NOT_NULLPTR_1(X)  \
    {                                       \
        MRMD_VERBOSE_CHECK_NOT_NULLPTR_1(X) \
    }
#define MRMD_DEVICE_CHECK_NOT_NULLPTR_2(X, MSG)  \
    {                                            \
        MRMD_VERBOSE_CHECK_NOT_NULLPTR_2(X, MSG) \
    }

#define MRMD_DEVICE_CHECK_EQUAL_2(X, Y)  \
    {                                    \
        MRMD_VERBOSE_CHECK_EQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_EQUAL_3(X, Y, MSG)  \
    {                                         \
        MRMD_VERBOSE_CHECK_EQUAL_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_CHECK_FLOAT_EQUAL_2(X, Y) \
    {                                         \
        MRMD_VERBOSE_CHECK_EQUAL_2(X, Y)      \
    }
#define MRMD_DEVICE_CHECK_FLOAT_EQUAL_3(X, Y, MSG) \
    {                                              \
        MRMD_VERBOSE_CHECK_EQUAL_3(X, Y, MSG)      \
    }

#define MRMD_DEVICE_CHECK_NOT_EQUAL_2(X, Y)  \
    {                                        \
        MRMD_VERBOSE_CHECK_NOT_EQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_NOT_EQUAL_3(X, Y, MSG)  \
    {                                             \
        MRMD_VERBOSE_CHECK_NOT_EQUAL_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_CHECK_GREATER_2(X, Y)  \
    {                                      \
        MRMD_VERBOSE_CHECK_GREATER_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_GREATER_3(X, Y, MSG)  \
    {                                           \
        MRMD_VERBOSE_CHECK_GREATER_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_CHECK_LESS_2(X, Y)  \
    {                                   \
        MRMD_VERBOSE_CHECK_LESS_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_LESS_3(X, Y, MSG)  \
    {                                        \
        MRMD_VERBOSE_CHECK_LESS_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_CHECK_GREATEREQUAL_2(X, Y)  \
    {                                           \
        MRMD_VERBOSE_CHECK_GREATEREQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_GREATEREQUAL_3(X, Y, MSG)  \
    {                                                \
        MRMD_VERBOSE_CHECK_GREATEREQUAL_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_CHECK_LESSEQUAL_2(X, Y)  \
    {                                        \
        MRMD_VERBOSE_CHECK_LESSEQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_LESSEQUAL_3(X, Y, MSG)  \
    {                                             \
        MRMD_VERBOSE_CHECK_LESSEQUAL_3(X, Y, MSG) \
    }
#else
#define MRMD_DEVICE_CHECK_1(X) \
    {                          \
        MRMD_NORMAL_CHECK_1(X) \
    }
#define MRMD_DEVICE_CHECK_2(X, MSG) \
    {                               \
        MRMD_NORMAL_CHECK_2(X, MSG) \
    }

#define MRMD_DEVICE_CHECK_NULLPTR_1(X) \
    {                                  \
        MRMD_NORMAL_CHECK_NULLPTR_1(X) \
    }
#define MRMD_DEVICE_CHECK_NULLPTR_2(X, MSG) \
    {                                       \
        MRMD_NORMAL_CHECK_NULLPTR_2(X, MSG) \
    }

#define MRMD_DEVICE_CHECK_NOT_NULLPTR_1(X) \
    {                                      \
        MRMD_NORMAL_CHECK_NOT_NULLPTR_1(X) \
    }
#define MRMD_DEVICE_CHECK_NOT_NULLPTR_2(X, MSG) \
    {                                           \
        MRMD_NORMAL_CHECK_NOT_NULLPTR_2(X, MSG) \
    }

#define MRMD_DEVICE_CHECK_EQUAL_2(X, Y) \
    {                                   \
        MRMD_NORMAL_CHECK_EQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_EQUAL_3(X, Y, MSG) \
    {                                        \
        MRMD_NORMAL_CHECK_EQUAL_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_CHECK_FLOAT_EQUAL_2(X, Y) \
    {                                         \
        MRMD_NORMAL_CHECK_EQUAL_2(X, Y)       \
    }
#define MRMD_DEVICE_CHECK_FLOAT_EQUAL_3(X, Y, MSG) \
    {                                              \
        MRMD_NORMAL_CHECK_EQUAL_3(X, Y, MSG)       \
    }

#define MRMD_DEVICE_CHECK_NOT_EQUAL_2(X, Y) \
    {                                       \
        MRMD_NORMAL_CHECK_NOT_EQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_NOT_EQUAL_3(X, Y, MSG) \
    {                                            \
        MRMD_NORMAL_CHECK_NOT_EQUAL_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_CHECK_GREATER_2(X, Y) \
    {                                     \
        MRMD_NORMAL_CHECK_GREATER_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_GREATER_3(X, Y, MSG) \
    {                                          \
        MRMD_NORMAL_CHECK_GREATER_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_CHECK_LESS_2(X, Y) \
    {                                  \
        MRMD_NORMAL_CHECK_LESS_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_LESS_3(X, Y, MSG) \
    {                                       \
        MRMD_NORMAL_CHECK_LESS_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_CHECK_GREATEREQUAL_2(X, Y) \
    {                                          \
        MRMD_NORMAL_CHECK_GREATEREQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_GREATEREQUAL_3(X, Y, MSG) \
    {                                               \
        MRMD_NORMAL_CHECK_GREATEREQUAL_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_CHECK_LESSEQUAL_2(X, Y) \
    {                                       \
        MRMD_NORMAL_CHECK_LESSEQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_CHECK_LESSEQUAL_3(X, Y, MSG) \
    {                                            \
        MRMD_NORMAL_CHECK_LESSEQUAL_3(X, Y, MSG) \
    }
#endif

#define MRMD_DEVICE_CHECK(...) MACRO_OVERLOAD(MRMD_DEVICE_CHECK_, __VA_ARGS__)
#define MRMD_DEVICE_CHECK_NULLPTR(...) MACRO_OVERLOAD(MRMD_DEVICE_CHECK_EQUAL_, __VA_ARGS__)
#define MRMD_DEVICE_CHECK_NOT_NULLPTR(...) \
    MACRO_OVERLOAD(MRMD_DEVICE_CHECK_NOT_NULLPTR_, __VA_ARGS__)
#define MRMD_DEVICE_CHECK_EQUAL(...) MACRO_OVERLOAD(MRMD_DEVICE_CHECK_EQUAL_, __VA_ARGS__)
#define MRMD_DEVICE_CHECK_NOT_EQUAL(...) MACRO_OVERLOAD(MRMD_DEVICE_CHECK_NOT_EQUAL_, __VA_ARGS__)
#define MRMD_DEVICE_CHECK_GREATER(...) MACRO_OVERLOAD(MRMD_DEVICE_CHECK_GREATER_, __VA_ARGS__)
#define MRMD_DEVICE_CHECK_LESS(...) MACRO_OVERLOAD(MRMD_DEVICE_CHECK_LESS_, __VA_ARGS__)
#define MRMD_DEVICE_CHECK_GREATEREQUAL(...) \
    MACRO_OVERLOAD(MRMD_DEVICE_CHECK_GREATEREQUAL_, __VA_ARGS__)
#define MRMD_DEVICE_CHECK_LESSEQUAL(...) MACRO_OVERLOAD(MRMD_DEVICE_CHECK_LESSEQUAL_, __VA_ARGS__)

#ifndef NDEBUG
#ifdef VERBOSE_ASSERT
#define MRMD_DEVICE_ASSERT_1(X) \
    {                           \
        MRMD_VERBOSE_CHECK_1(X) \
    }
#define MRMD_DEVICE_ASSERT_2(X, MSG) \
    {                                \
        MRMD_VERBOSE_CHECK_2(X, MSG) \
    }

#define MRMD_DEVICE_ASSERT_NULLPTR_1(X) \
    {                                   \
        MRMD_VERBOSE_CHECK_NULLPTR_1(X) \
    }
#define MRMD_DEVICE_ASSERT_NULLPTR_2(X, MSG) \
    {                                        \
        MRMD_VERBOSE_CHECK_NULLPTR_2(X, MSG) \
    }

#define MRMD_DEVICE_ASSERT_NOT_NULLPTR_1(X) \
    {                                       \
        MRMD_VERBOSE_CHECK_NOT_NULLPTR_1(X) \
    }
#define MRMD_DEVICE_ASSERT_NOT_NULLPTR_2(X, MSG) \
    {                                            \
        MRMD_VERBOSE_CHECK_NOT_NULLPTR_2(X, MSG) \
    }

#define MRMD_DEVICE_ASSERT_EQUAL_2(X, Y) \
    {                                    \
        MRMD_VERBOSE_CHECK_EQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_ASSERT_EQUAL_3(X, Y, MSG) \
    {                                         \
        MRMD_VERBOSE_CHECK_EQUAL_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_ASSERT_FLOAT_EQUAL_2(X, Y) \
    {                                          \
        MRMD_VERBOSE_CHECK_EQUAL_2(X, Y)       \
    }
#define MRMD_DEVICE_ASSERT_FLOAT_EQUAL_3(X, Y, MSG) \
    {                                               \
        MRMD_VERBOSE_CHECK_EQUAL_3(X, Y, MSG)       \
    }

#define MRMD_DEVICE_ASSERT_NOT_EQUAL_2(X, Y) \
    {                                        \
        MRMD_VERBOSE_CHECK_NOT_EQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_ASSERT_NOT_EQUAL_3(X, Y, MSG) \
    {                                             \
        MRMD_VERBOSE_CHECK_NOT_EQUAL_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_ASSERT_GREATER_2(X, Y) \
    {                                      \
        MRMD_VERBOSE_CHECK_GREATER_2(X, Y) \
    }
#define MRMD_DEVICE_ASSERT_GREATER_3(X, Y, MSG) \
    {                                           \
        MRMD_VERBOSE_CHECK_GREATER_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_ASSERT_LESS_2(X, Y) \
    {                                   \
        MRMD_VERBOSE_CHECK_LESS_2(X, Y) \
    }
#define MRMD_DEVICE_ASSERT_LESS_3(X, Y, MSG) \
    {                                        \
        MRMD_VERBOSE_CHECK_LESS_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_ASSERT_GREATEREQUAL_2(X, Y) \
    {                                           \
        MRMD_VERBOSE_CHECK_GREATEREQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_ASSERT_GREATEREQUAL_3(X, Y, MSG) \
    {                                                \
        MRMD_VERBOSE_CHECK_GREATEREQUAL_3(X, Y, MSG) \
    }

#define MRMD_DEVICE_ASSERT_LESSEQUAL_2(X, Y) \
    {                                        \
        MRMD_VERBOSE_CHECK_LESSEQUAL_2(X, Y) \
    }
#define MRMD_DEVICE_ASSERT_LESSEQUAL_3(X, Y, MSG) \
    {                                             \
        MRMD_VERBOSE_CHECK_LESSEQUAL_3(X, Y, MSG) \
    }
#else
#define MRMD_DEVICE_ASSERT_1(X) \
    {                           \
        MRMD_NORMAL_CHECK_1(X)  \
    }
#define MRMD_DEVICE_ASSERT_2(X, MSG) \
    {                                \
        MRMD_NORMAL_CHECK_2(X, MSG)  \
    }

#define MRMD_DEVICE_ASSERT_NULLPTR_1(X) \
    {                                   \
        MRMD_NORMAL_CHECK_NULLPTR_1(X)  \
    }
#define MRMD_DEVICE_ASSERT_NULLPTR_2(X, MSG) \
    {                                        \
        MRMD_NORMAL_CHECK_NULLPTR_2(X, MSG)  \
    }

#define MRMD_DEVICE_ASSERT_NOT_NULLPTR_1(X) \
    {                                       \
        MRMD_NORMAL_CHECK_NOT_NULLPTR_1(X)  \
    }
#define MRMD_DEVICE_ASSERT_NOT_NULLPTR_2(X, MSG) \
    {                                            \
        MRMD_NORMAL_CHECK_NOT_NULLPTR_2(X, MSG)  \
    }

#define MRMD_DEVICE_ASSERT_EQUAL_2(X, Y) \
    {                                    \
        MRMD_NORMAL_CHECK_EQUAL_2(X, Y)  \
    }
#define MRMD_DEVICE_ASSERT_EQUAL_3(X, Y, MSG) \
    {                                         \
        MRMD_NORMAL_CHECK_EQUAL_3(X, Y, MSG)  \
    }

#define MRMD_DEVICE_ASSERT_FLOAT_EQUAL_2(X, Y) \
    {                                          \
        MRMD_NORMAL_CHECK_EQUAL_2(X, Y)        \
    }
#define MRMD_DEVICE_ASSERT_FLOAT_EQUAL_3(X, Y, MSG) \
    {                                               \
        MRMD_NORMAL_CHECK_EQUAL_3(X, Y, MSG)        \
    }

#define MRMD_DEVICE_ASSERT_NOT_EQUAL_2(X, Y) \
    {                                        \
        MRMD_NORMAL_CHECK_NOT_EQUAL_2(X, Y)  \
    }
#define MRMD_DEVICE_ASSERT_NOT_EQUAL_3(X, Y, MSG) \
    {                                             \
        MRMD_NORMAL_CHECK_NOT_EQUAL_3(X, Y, MSG)  \
    }

#define MRMD_DEVICE_ASSERT_GREATER_2(X, Y) \
    {                                      \
        MRMD_NORMAL_CHECK_GREATER_2(X, Y)  \
    }
#define MRMD_DEVICE_ASSERT_GREATER_3(X, Y, MSG) \
    {                                           \
        MRMD_NORMAL_CHECK_GREATER_3(X, Y, MSG)  \
    }

#define MRMD_DEVICE_ASSERT_LESS_2(X, Y) \
    {                                   \
        MRMD_NORMAL_CHECK_LESS_2(X, Y)  \
    }
#define MRMD_DEVICE_ASSERT_LESS_3(X, Y, MSG) \
    {                                        \
        MRMD_NORMAL_CHECK_LESS_3(X, Y, MSG)  \
    }

#define MRMD_DEVICE_ASSERT_GREATEREQUAL_2(X, Y) \
    {                                           \
        MRMD_NORMAL_CHECK_GREATEREQUAL_2(X, Y)  \
    }
#define MRMD_DEVICE_ASSERT_GREATEREQUAL_3(X, Y, MSG) \
    {                                                \
        MRMD_NORMAL_CHECK_GREATEREQUAL_3(X, Y, MSG)  \
    }

#define MRMD_DEVICE_ASSERT_LESSEQUAL_2(X, Y) \
    {                                        \
        MRMD_NORMAL_CHECK_LESSEQUAL_2(X, Y)  \
    }
#define MRMD_DEVICE_ASSERT_LESSEQUAL_3(X, Y, MSG) \
    {                                             \
        MRMD_NORMAL_CHECK_LESSEQUAL_3(X, Y, MSG)  \
    }
#endif

#else

#define MRMD_DEVICE_ASSERT_1(X) \
    {                           \
    }
#define MRMD_DEVICE_ASSERT_2(X, MSG) \
    {                                \
    }

#define MRMD_DEVICE_ASSERT_NULLPTR_1(X) \
    {                                   \
    }
#define MRMD_DEVICE_ASSERT_NULLPTR_2(X, MSG) \
    {                                        \
    }

#define MRMD_DEVICE_ASSERT_NOT_NULLPTR_1(X) \
    {                                       \
    }
#define MRMD_DEVICE_ASSERT_NOT_NULLPTR_2(X, MSG) \
    {                                            \
    }

#define MRMD_DEVICE_ASSERT_EQUAL_2(X, Y) \
    {                                    \
    }
#define MRMD_DEVICE_ASSERT_EQUAL_3(X, Y, MSG) \
    {                                         \
    }

#define MRMD_DEVICE_ASSERT_NOT_EQUAL_2(X, Y) \
    {                                        \
    }
#define MRMD_DEVICE_ASSERT_NOT_EQUAL_3(X, Y, MSG) \
    {                                             \
    }

#define MRMD_DEVICE_ASSERT_GREATER_2(X, Y) \
    {                                      \
    }
#define MRMD_DEVICE_ASSERT_GREATER_3(X, Y, MSG) \
    {                                           \
    }

#define MRMD_DEVICE_ASSERT_LESS_2(X, Y) \
    {                                   \
    }
#define MRMD_DEVICE_ASSERT_LESS_3(X, Y, MSG) \
    {                                        \
    }

#define MRMD_DEVICE_ASSERT_GREATEREQUAL_2(X, Y) \
    {                                           \
    }
#define MRMD_DEVICE_ASSERT_GREATEREQUAL_3(X, Y, MSG) \
    {                                                \
    }

#define MRMD_DEVICE_ASSERT_LESSEQUAL_2(X, Y) \
    {                                        \
    }
#define MRMD_DEVICE_ASSERT_LESSEQUAL_3(X, Y, MSG) \
    {                                             \
    }

#endif

#define MRMD_DEVICE_ASSERT(...) MACRO_OVERLOAD(MRMD_DEVICE_ASSERT_, __VA_ARGS__)
#define MRMD_DEVICE_ASSERT_NULLPTR(...) MACRO_OVERLOAD(MRMD_DEVICE_ASSERT_EQUAL_, __VA_ARGS__)
#define MRMD_DEVICE_ASSERT_NOT_NULLPTR(...) \
    MACRO_OVERLOAD(MRMD_DEVICE_ASSERT_NOT_NULLPTR_, __VA_ARGS__)
#define MRMD_DEVICE_ASSERT_EQUAL(...) MACRO_OVERLOAD(MRMD_DEVICE_ASSERT_EQUAL_, __VA_ARGS__)
#define MRMD_DEVICE_ASSERT_NOT_EQUAL(...) MACRO_OVERLOAD(MRMD_DEVICE_ASSERT_NOT_EQUAL_, __VA_ARGS__)
#define MRMD_DEVICE_ASSERT_GREATER(...) MACRO_OVERLOAD(MRMD_DEVICE_ASSERT_GREATER_, __VA_ARGS__)
#define MRMD_DEVICE_ASSERT_LESS(...) MACRO_OVERLOAD(MRMD_DEVICE_ASSERT_LESS_, __VA_ARGS__)
#define MRMD_DEVICE_ASSERT_GREATEREQUAL(...) \
    MACRO_OVERLOAD(MRMD_DEVICE_ASSERT_GREATEREQUAL_, __VA_ARGS__)
#define MRMD_DEVICE_ASSERT_LESSEQUAL(...) MACRO_OVERLOAD(MRMD_DEVICE_ASSERT_LESSEQUAL_, __VA_ARGS__)