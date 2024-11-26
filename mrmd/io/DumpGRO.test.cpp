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

#include "DumpGRO.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace io
{

bool compareFiles(FILE* f1, FILE* f2) {
  int N = 10000;
  char buf1[N];
  char buf2[N];

  do {
    size_t r1 = fread(buf1, 1, N, f1);
    size_t r2 = fread(buf2, 1, N, f2);

    if (r1 != r2 ||
        memcmp(buf1, buf2, r1)) {
      return 0;
    }
  } while (!feof(f1) || !feof(f2));

  return true;
}

TEST(DumpGRO, atoms)
{
    auto atoms = data::Atoms(10 * 2);
    auto subdomain = data::Subdomain();
    atoms.numLocalAtoms = 10;
    atoms.numGhostAtoms = 10;

    const std::vector<std::string> typeNames = {"At"};
    dumpGRO("testAtom.gro", atoms, subdomain, 0, "Test", "Atom", typeNames, true, true);

    FILE* testFile = std::fopen("testAtom.gro", "r");
    FILE* refFile = std::fopen("../../../tests/testData/refAtom.gro", "r");

    auto equalFiles = compareFiles(testFile, refFile);
    ASSERT_TRUE(equalFiles);
}
}  // namespace io
}  // namespace mrmd