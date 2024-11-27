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

void compareFiles(const std::string& filename1, const std::string& filename2)
{
    std::ifstream file1(filename1);
    std::ifstream file2(filename2);

    ASSERT_TRUE(file1.is_open());
    ASSERT_TRUE(file2.is_open());

    std::string line1, line2;
    while (std::getline(file1, line1) && std::getline(file2, line2))
    {
        ASSERT_EQ(line1, line2);
    }

    file1.close();
    file2.close();
}

TEST(DumpGRO, atoms)
{
    auto atoms = data::Atoms(10 * 2);
    auto subdomain = data::Subdomain();
    atoms.numLocalAtoms = 10;
    atoms.numGhostAtoms = 10;

    const std::vector<std::string> typeNames = {"At"};
    dumpGRO("testAtom.gro", atoms, subdomain, 0, "Test", "Atom", typeNames, true, true);

    compareFiles("testAtom.gro", "DumpGRO.test.gro");
}
}  // namespace io
}  // namespace mrmd