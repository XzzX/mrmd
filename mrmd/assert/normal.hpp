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

#define MRMD_NORMAL_CHECK_1(X) \
    {                          \
        assert(X);             \
    }
#define MRMD_NORMAL_CHECK_2(X, MSG) \
    {                               \
        assert(X);                  \
    }

#define MRMD_NORMAL_CHECK_NULLPTR_1(X) \
    {                                  \
        assert(X == nullptr);          \
    }
#define MRMD_NORMAL_CHECK_NULLPTR_2(X, MSG) \
    {                                       \
        assert(X == nullptr);               \
    }

#define MRMD_NORMAL_CHECK_NOT_NULLPTR_1(X) \
    {                                      \
        assert(X != nullptr);              \
    }
#define MRMD_NORMAL_CHECK_NOT_NULLPTR_2(X, MSG) \
    {                                           \
        assert(X != nullptr);                   \
    }

#define MRMD_NORMAL_CHECK_EQUAL_2(X, Y) \
    {                                   \
        assert(X == Y);                 \
    }
#define MRMD_NORMAL_CHECK_EQUAL_3(X, Y, MSG) \
    {                                        \
        assert(X == Y);                      \
    }

#define MRMD_NORMAL_CHECK_FLOAT_EQUAL_2(X, Y) \
    {                                         \
        assert(X == Y);                       \
    }
#define MRMD_NORMAL_CHECK_FLOAT_EQUAL_3(X, Y, MSG) \
    {                                              \
        assert(X == Y);                            \
    }

#define MRMD_NORMAL_CHECK_NOT_EQUAL_2(X, Y) \
    {                                       \
        assert(X != Y);                     \
    }
#define MRMD_NORMAL_CHECK_NOT_EQUAL_3(X, Y, MSG) \
    {                                            \
        assert(X != Y);                          \
    }

#define MRMD_NORMAL_CHECK_GREATER_2(X, Y) \
    {                                     \
        assert(X > Y);                    \
    }
#define MRMD_NORMAL_CHECK_GREATER_3(X, Y, MSG) \
    {                                          \
        assert(X > Y);                         \
    }

#define MRMD_NORMAL_CHECK_LESS_2(X, Y) \
    {                                  \
        assert(X < Y);                 \
    }
#define MRMD_NORMAL_CHECK_LESS_3(X, Y, MSG) \
    {                                       \
        assert(X < Y);                      \
    }

#define MRMD_NORMAL_CHECK_GREATEREQUAL_2(X, Y) \
    {                                          \
        assert(X >= Y);                        \
    }
#define MRMD_NORMAL_CHECK_GREATEREQUAL_3(X, Y, MSG) \
    {                                               \
        assert(X >= Y);                             \
    }

#define MRMD_NORMAL_CHECK_LESSEQUAL_2(X, Y) \
    {                                       \
        assert(X <= Y);                     \
    }
#define MRMD_NORMAL_CHECK_LESSEQUAL_3(X, Y, MSG) \
    {                                            \
        assert(X <= Y);                          \
    }

#define MRMD_NORMAL_CHECK(...) MACRO_OVERLOAD(MRMD_NORMAL_CHECK_, __VA_ARGS__)
#define MRMD_NORMAL_CHECK_NULLPTR(...) MACRO_OVERLOAD(MRMD_NORMAL_CHECK_EQUAL_, __VA_ARGS__)
#define MRMD_NORMAL_CHECK_NOT_NULLPTR(...) \
    MACRO_OVERLOAD(MRMD_NORMAL_CHECK_NOT_NULLPTR_, __VA_ARGS__)
#define MRMD_NORMAL_CHECK_EQUAL(...) MACRO_OVERLOAD(MRMD_NORMAL_CHECK_EQUAL_, __VA_ARGS__)
#define MRMD_NORMAL_CHECK_FLOAT_EQUAL(...) MACRO_OVERLOAD(MRMD_NORMAL_CHECK_EQUAL_, __VA_ARGS__)
#define MRMD_NORMAL_CHECK_NOT_EQUAL(...) MACRO_OVERLOAD(MRMD_NORMAL_CHECK_NOT_EQUAL_, __VA_ARGS__)
#define MRMD_NORMAL_CHECK_GREATER(...) MACRO_OVERLOAD(MRMD_NORMAL_CHECK_GREATER_, __VA_ARGS__)
#define MRMD_NORMAL_CHECK_LESS(...) MACRO_OVERLOAD(MRMD_NORMAL_CHECK_LESS_, __VA_ARGS__)
#define MRMD_NORMAL_CHECK_GREATEREQUAL(...) \
    MACRO_OVERLOAD(MRMD_NORMAL_CHECK_GREATEREQUAL_, __VA_ARGS__)
#define MRMD_NORMAL_CHECK_LESSEQUAL(...) MACRO_OVERLOAD(MRMD_NORMAL_CHECK_LESSEQUAL_, __VA_ARGS__)