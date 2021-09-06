#ifndef NOISY_ASSERT_H
#define NOISY_ASSERT_H

#include <iostream>
// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

#define DEBUG_STATE 0         // Causes debug statements to be printed

// literally, "debug printf"
#define dprintf(debug, msg) if (debug!=0) printf(msg)

#endif // NOISY_ASSERT_H