#ifndef NOISY_ASSERT_H
#define NOISY_ASSERT_H

#include <iostream>
// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

#endif // NOISY_ASSERT_H