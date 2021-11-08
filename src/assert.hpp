#pragma once

#define VERBOSE_ASSERT

#ifdef VERBOSE_ASSERT
#include "verbose_assert.hpp"
#else
#include "normal_assert.hpp"
#endif