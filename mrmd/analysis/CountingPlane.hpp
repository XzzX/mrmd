// Copyright 2025 Sebastian Eibl
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

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd::analysis
{
/// Counts particles that cross a defined plane between two time steps.
class CountingPlane
{
    Point3D pointOnPlane_;
    Vector3D planeNormal_;
    ScalarView distanceToPlane_;

public:
    CountingPlane(const Point3D &pointOnPlane, const Vector3D &planeNormal);

    void startCounting(data::Atoms &atoms);
    int64_t stopCounting(data::Atoms &atoms);
};
}  // namespace mrmd::analysis